"""Identification engine: which molecule is each CHEMKIN species?

Evidence, combined as log-probabilities for every candidate structure c of a
species S:

    log P(c | data) = prior(c) + name(c) + sum_r  -ALPHA * d_r(c)  + const

* formula   - hard constraint from the thermo card (candidates must match)
* prior     - chemical plausibility of c (lewis + isomers.plausibility)
* name      - aliases, condensed formulas, LLNL patterns, N-/I-/S-/T- prefixes
* reactions - d_r(c) is the minimum number of sigma bonds broken + formed
              to run reaction r with S = c (all other species of r already
              resolved).  Elementary steps need 1-3 edits; a wrong
              structure needs more, so each reaction is a witness.

Candidates come from isomer enumeration (small formulas), the name, and
reaction-driven generation (structures that complete a reaction with few
bond edits - the main source for large species).  Resolution is iterative:
species above the confidence threshold are fixed, which turns more
reactions into witnesses for the rest (constraint propagation).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from .chemkin import Mechanism, Reaction, read_known_smiles
from .elements import hill_formula
from .isomers import enumerate_isomers, plausibility
from .mcd import generate_completions, sigma_distance, solve_reaction
from .molecule import Molecule
from .names import feature_bonus, name_candidates, name_features
from .smiles import SmilesError, canonical_smiles, parse_smiles

ALPHA = 1.5            # log-likelihood per extra sigma edit
D_MAX = 4              # edits considered "elementary"; beyond is uninformative
D_CAP_GAP = 3          # one reaction can move a candidate by at most this many edits
MAX_FORMULA_CANDIDATES = 80
DUPLICATE_PENALTY = 6.0  # a mechanism rarely lists the same molecule twice
LUMPED_WEIGHT = 0.25     # weight of a reaction with 3+ species on one side
GEN_PER_REACTION = 60    # reaction-built structures kept per witness reaction
BIG_SIDE_HEAVY = 14      # above this many heavy atoms on a side, search 2 deletions per side, not 3

# audit limits (per given species)
AUDIT_REACTIONS = 24     # witness reactions scored, elementary and small first
AUDIT_GEN_REACTIONS = 6  # of those, reactions used to build alternatives
AUDIT_CANDIDATES = 40    # alternatives scored besides the given structure
AUDIT_SECONDS = 20.0     # time budget; scoring stops after the current reaction

# Ground-state spin where Lewis theory (which pairs every electron it can)
# gets it wrong: Hund's rule for atoms and the classic triplet molecules.
GROUND_STATE_MULTIPLICITY = {
    "O2": 3, "CH2": 3, "O": 3, "C": 3, "S": 3, "NH": 3, "N": 4, "S2": 3, "SO": 3,
    "CH": 2, "Si": 3, "CN": 2, "C2O": 3, "HCCH": 1, "NCN": 3, "CCO": 3,
}


@dataclass
class Candidate:
    mol: Molecule
    key: str
    smiles: str
    sources: set = field(default_factory=set)
    log_prior: float = 0.0
    log_name: float = 0.0
    reasons: list = field(default_factory=list)
    distances: dict = field(default_factory=dict)   # reaction index -> sigma edits (None = > D_MAX)
    log_like: float = 0.0
    posterior: float = 0.0

    dup_penalty: float = 0.0
    log_thermo: float = 0.0
    thermo_raw: float = 0.0
    thermo_dh: float | None = None
    thermo_note: str = ""

    @property
    def score(self) -> float:
        return self.log_prior + self.log_name + self.log_like + self.log_thermo - self.dup_penalty


@dataclass
class SpeciesState:
    name: str
    formula: Counter | None
    multiplicity: int | None = None
    status: str = "unresolved"          # known | identified | unresolved | no-formula
    method: str = ""
    chosen: Candidate | None = None
    confidence: float = 0.0
    round: int = 0
    candidates: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    default_multiplicity: int | None = None
    truth: str | None = None            # for benchmarking

    @property
    def resolved(self) -> bool:
        return self.status in ("known", "identified")

    @property
    def formula_str(self) -> str:
        return hill_formula(self.formula) if self.formula else "?"

    def ranked(self) -> list[Candidate]:
        return sorted(self.candidates.values(), key=lambda c: -c.posterior)


def spin_from_name(name: str, formula: Counter | None) -> int | None:
    u = name.upper()
    if re.search(r"\(S\)$|^S-?CH2$|CH2S$|CH2\*$|SING|\(1D\)|1D$", u):
        return 1
    if re.search(r"\(T\)$|^T-?CH2$|TRIP", u):
        return 3
    return None


class Identifier:
    def __init__(self, mech: Mechanism, known: dict[str, str] | None = None,
                 use_names: bool = True, threshold: float = 0.9, verbose: bool = False,
                 max_rounds: int = 40, time_limit: float = 1800.0, thermo="auto",
                 exclude_models: set[str] | None = None):
        self.mech = mech
        if thermo == "auto":
            from .thermo import default_model
            thermo = default_model(exclude_models)
        self.thermo = thermo
        self.known = known or {}
        self.use_names = use_names
        self.threshold = threshold
        self.verbose = verbose
        self.max_rounds = max_rounds
        self.time_limit = time_limit
        self.species: dict[str, SpeciesState] = {}
        self.log: list[str] = []
        self._iso_cache: dict[str, list | None] = {}
        self._gen_done: set[tuple[str, int]] = set()
        self.solutions: dict[int, object] = {}
        self.inert = set()

    # ------------------------------------------------------------ helpers
    def say(self, msg: str) -> None:
        self.log.append(msg)
        if self.verbose:
            print(msg, flush=True)

    def mol_of(self, name: str) -> Molecule | None:
        st = self.species.get(name)
        if st is None or not st.resolved or st.chosen is None:
            return None
        return st.chosen.mol

    def _usable(self, r: Reaction) -> bool:
        return all(self.species.get(s) is not None and self.species[s].formula is not None
                   for s in r.species())

    # ------------------------------------------------------------ setup
    def setup(self) -> None:
        mech = self.mech
        for name in mech.species:
            f = mech.formula(name)
            if f is not None and "E" in f:
                f = None
            st = SpeciesState(name, Counter(f) if f else None)
            st.multiplicity = spin_from_name(name, st.formula)
            st.default_multiplicity = GROUND_STATE_MULTIPLICITY.get(hill_formula(f)) if f else None
            self.species[name] = st
        self._infer_missing_formulas()
        for name, smi in self.known.items():
            st = self.species.get(name)
            if st is None:
                continue
            try:
                mol = parse_smiles(smi)
            except SmilesError as e:
                st.notes.append(f"known SMILES unreadable: {e}")
                continue
            mol.label = name
            if st.formula is not None and Counter(mol.formula()) != st.formula:
                st.notes.append(f"known SMILES {smi} has formula {mol.formula_string()}, "
                                f"thermo says {st.formula_str} - ignored")
                continue
            if st.formula is None:
                st.formula = Counter(mol.formula())
            if st.multiplicity is None:
                st.multiplicity = mol.multiplicity
            c = self._make_candidate(st, mol, "user")
            st.candidates[c.key] = c
            st.chosen = c
            st.status = "known"
            st.method = "given"
            st.confidence = 1.0
        for st in self.species.values():
            if st.formula is None and not st.resolved:
                st.status = "no-formula"
        n_known = sum(1 for s in self.species.values() if s.status == "known")
        self.say(f"{len(self.species)} species, {len(self.mech.reactions)} reactions, "
                 f"{n_known} given as known")

    def _infer_missing_formulas(self) -> None:
        changed = True
        while changed:
            changed = False
            for r in self.mech.reactions:
                missing = [s for s in r.species() if self.species.get(s) and self.species[s].formula is None]
                if len(missing) != 1:
                    continue
                s = missing[0]
                lc = Counter(r.reactants).get(s, 0)
                pc = Counter(r.products).get(s, 0)
                if lc + pc != 1:
                    continue
                total = Counter()
                for x in r.reactants:
                    if x != s:
                        total.update(self.species[x].formula)
                for x in r.products:
                    if x != s:
                        total.subtract(self.species[x].formula)
                if lc:
                    total = Counter({k: -v for k, v in total.items()})
                total = Counter({k: v for k, v in total.items() if v})
                if total and all(v > 0 for v in total.values()):
                    self.species[s].formula = total
                    self.species[s].notes.append(f"formula {hill_formula(total)} inferred from reaction {r.index}")
                    changed = True

    def _make_candidate(self, st: SpeciesState, mol: Molecule, source: str) -> Candidate:
        mol = mol.copy()
        mol.label = st.name
        if st.multiplicity is not None:          # name hint or user SMILES
            mol.multiplicity = st.multiplicity
        elif st.default_multiplicity is not None:  # Hund's-rule ground state
            mol.multiplicity = st.default_multiplicity
        elif source not in ("user", "name"):
            mol.multiplicity = None                # let Lewis theory pair electrons
        key = mol.constitution_key()
        try:
            smi = canonical_smiles(mol)
        except Exception:  # pragma: no cover - drawing failure must not stop identification
            smi = mol.formula_string()
        lp, reasons = plausibility(mol)
        c = Candidate(mol, key, smi, {source}, lp, 0.0, list(reasons))
        t = self.mech.thermo.get(st.name)
        if self.thermo is not None and t is not None and t.h298 is not None and source != "user":
            try:
                c.thermo_raw, c.thermo_dh, c.thermo_note = self.thermo.log_likelihood(
                    mol, key, t.h298, t.s298)
                c.log_thermo = c.thermo_raw
            except Exception as e:  # noqa: BLE001 - thermo is optional evidence
                c.thermo_note = f"thermo estimate failed: {e}"
        return c

    def _add_candidate(self, st: SpeciesState, mol: Molecule, source: str,
                       name_weight: float = 0.0, why: str = "") -> Candidate | None:
        if st.formula is not None and Counter(mol.formula()) != st.formula:
            return None
        key = mol.constitution_key()
        c = st.candidates.get(key)
        if c is None:
            c = self._make_candidate(st, mol, source)
            if self.use_names:
                c.log_prior += feature_bonus(mol, name_features(st.name))
            st.candidates[key] = c
        c.sources.add(source)
        if name_weight > c.log_name:
            c.log_name = name_weight
            if why:
                c.reasons.append(why)
        return c

    def generate_initial_candidates(self) -> None:
        for st in self.species.values():
            if st.resolved or st.formula is None:
                continue
            if self.use_names:
                for mol, w, why in name_candidates(st.name, st.formula):
                    self._add_candidate(st, mol, "name", w, why)
            fkey = hill_formula(st.formula)
            if fkey not in self._iso_cache:
                self._iso_cache[fkey] = enumerate_isomers(st.formula)
            iso = self._iso_cache[fkey]
            if iso is not None:
                scored = sorted(iso, key=lambda m: -plausibility(m)[0])
                for mol in scored[:MAX_FORMULA_CANDIDATES]:
                    self._add_candidate(st, mol, "formula")
                if len(iso) <= MAX_FORMULA_CANDIDATES:
                    st.notes.append(f"{len(iso)} constitutional isomers for {fkey} (complete list)")
                else:
                    st.notes.append(f"{len(iso)} isomers for {fkey}; best {MAX_FORMULA_CANDIDATES} kept")
            else:
                st.notes.append(f"{fkey}: too many isomers to enumerate - reaction-driven candidates only")
            self._normalize_priors(st)

    def _normalize_priors(self, st):
        if not st.candidates:
            return
        top = max(c.log_prior for c in st.candidates.values())
        for c in st.candidates.values():
            c.log_prior -= top

    # ------------------------------------------------------------ evidence
    def witness_reactions(self, name: str) -> list[Reaction]:
        out = []
        for r in self.mech.reactions:
            if name not in r.reactants and name not in r.products:
                continue
            if name in r.reactants and name in r.products:
                continue
            others = [s for s in r.species() if s != name]
            if all(self.species[s].resolved for s in others if s in self.species):
                if all(s in self.species for s in others):
                    out.append(r)
        return out

    def _substituted(self, r: Reaction, name: str, mol: Molecule):
        def side(names):
            out = []
            for s in names:
                m = mol if s == name else self.mol_of(s)
                if m is None:
                    return None
                out.append(m)
            return out
        return side(r.reactants), side(r.products)

    def reaction_generated(self, st: SpeciesState, rxns: list[Reaction],
                           deadline: float | None = None) -> None:
        for r in rxns:
            if deadline is not None and time.time() > deadline:
                break
            tag = (st.name, r.index)
            if tag in self._gen_done:
                continue
            self._gen_done.add(tag)
            lc = r.reactants.count(st.name)
            pc = r.products.count(st.name)
            if lc + pc != 1:
                continue
            if lc:
                other = [self.mol_of(s) for s in r.products]
                same = [self.mol_of(s) for s in r.reactants if s != st.name]
            else:
                other = [self.mol_of(s) for s in r.reactants]
                same = [self.mol_of(s) for s in r.products if s != st.name]
            if any(m is None for m in other + same) or not other:
                continue
            heavy_other = sum(m.heavy_count() for m in other)
            max_side = 2 if heavy_other <= 12 else 1
            try:
                gens = generate_completions(other, same, max_edits=3, max_side=max_side,
                                            deadline=deadline)
            except RecursionError:
                continue
            # fewest-edit structures first; one reaction adds at most GEN_PER_REACTION
            ranked = sorted(gens.items(), key=lambda kv: kv[1][1])[:GEN_PER_REACTION]
            for key, (mol, edits) in ranked:
                if deadline is not None and time.time() > deadline:
                    break
                if Counter(mol.formula()) != st.formula:
                    continue
                c = self._add_candidate(st, mol, "reaction")
                if c is not None and f"completes reaction {r.index}" not in c.reasons:
                    c.reasons.append(f"completes reaction {r.index} with {edits} edits")
        self._normalize_priors(st)

    def score_species(self, st: SpeciesState, rxns: list[Reaction], max_candidates: int = 160,
                      deadline: float | None = None, must_keep: set | None = None):
        """Score candidates; returns (scored candidates, reactions actually used).

        Distances are filled reaction by reaction for all candidates, so when
        the deadline passes every candidate has been judged on the same
        reactions.
        """
        cands = list(st.candidates.values())
        if not cands:
            return [], []
        # keep the work bounded: score the plausible + name + reaction-derived ones
        if len(cands) > max_candidates:
            must_keep = must_keep or set()
            cands.sort(key=lambda c: (c.key not in must_keep,
                                      -(c.log_prior + c.log_name + (2 if "reaction" in c.sources else 0))))
            keep = cands[:max_candidates]
            for c in cands[max_candidates:]:
                c.posterior = 0.0
                c.log_like = 0.0
            cands = keep
        # thermo reliability: if no candidate fits the species' thermo, the
        # thermo card is suspect - keep only a quarter of its weight
        dhs = [c.thermo_dh for c in cands if c.thermo_dh is not None]
        factor = 0.25 if dhs and min(dhs) > 12.0 else 1.0
        if factor < 1 and not any("thermo unreliable" in n for n in st.notes):
            st.notes.append(f"thermo unreliable: no candidate within 12 kcal/mol (best {min(dhs):.1f})")
        for c in cands:
            c.log_thermo = c.thermo_raw * factor
        taken = self._taken(st)
        for c in cands:
            c.dup_penalty = DUPLICATE_PENALTY if (c.key, st.multiplicity) in taken else 0.0
            c.log_like = 0.0
        used = []
        for r in rxns:
            if deadline is not None and time.time() > deadline:
                break
            for c in cands:
                if r.index in c.distances:
                    continue
                sides = self._substituted(r, st.name, c.mol)
                d = None
                if sides[0] is not None and sides[1] is not None:
                    big = max(sum(m.heavy_count() for m in sides[0]),
                              sum(m.heavy_count() for m in sides[1]))
                    res = sigma_distance(sides[0], sides[1], max_total=D_MAX,
                                         max_side=2 if big > BIG_SIDE_HEAVY else 3)
                    d = None if res is None else res[0]
                c.distances[r.index] = d
            used.append(r)
        rxns = used
        # per reaction: relative penalty with a cap so one reaction can't dominate.
        # Lumped steps (3+ species on a side) and steps nobody explains in < 3
        # edits are weak witnesses: they favour whatever makes them look short.
        for r in rxns:
            ds = [c.distances.get(r.index) for c in cands]
            known_ds = [d for d in ds if d is not None]
            if not known_ds:
                continue      # no candidate explains it: lumped / non-elementary
            best = min(known_ds)
            w = 1.0 if max(len(r.reactants), len(r.products)) <= 2 else LUMPED_WEIGHT
            if best >= 3:
                w *= 0.5
            for c in cands:
                d = c.distances.get(r.index)
                gap = D_CAP_GAP if d is None else min(D_CAP_GAP, d - best)
                c.log_like -= ALPHA * w * gap
        scores = [c.score for c in cands]
        top = max(scores)
        z = sum(math.exp(s - top) for s in scores)
        for c in cands:
            c.posterior = math.exp(c.score - top) / z
        return cands, rxns

    def _taken(self, me: SpeciesState) -> set:
        out = set()
        for st in self.species.values():
            if st is not me and st.resolved and st.chosen is not None:
                out.add((st.chosen.key, st.multiplicity))
        return out

    # ------------------------------------------------------------ main loop
    def run(self) -> "Identifier":
        t0 = time.time()
        self.setup()
        self.generate_initial_candidates()
        thresholds = [self.threshold, 0.8, 0.65, 0.5]
        level = 0
        for rnd in range(1, self.max_rounds + 1):
            if time.time() - t0 > self.time_limit:
                self.say("time limit reached")
                break
            open_species = [s for s in self.species.values() if not s.resolved and s.formula is not None]
            if not open_species:
                break
            proposals = []
            for st in open_species:
                rxns = self.witness_reactions(st.name)
                if rxns:
                    self.reaction_generated(st, rxns)
                self.score_species(st, rxns)
                ranked = st.ranked()
                if not ranked:
                    continue
                top = ranked[0]
                second = ranked[1].posterior if len(ranked) > 1 else 0.0
                n_wit = sum(1 for r in rxns if top.distances.get(r.index) is not None)
                proposals.append((top.posterior, top.posterior - second, n_wit, st, top, len(rxns)))
            tau = thresholds[level]
            # below 0.8 a clear margin over the runner-up is also required, so
            # look-alike isomers (o- vs p-cresol) stay open instead of a coin flip
            accept = [p for p in proposals if p[0] >= tau and (tau >= 0.8 or p[1] >= 0.3)]
            claimed = self._taken(None)
            kept = []
            for p in sorted(accept, key=lambda p: (-p[0], -p[2])):
                k = (p[4].key, p[3].multiplicity)
                if k not in claimed:   # never hand one structure to two species
                    claimed.add(k)
                    kept.append(p)
            accept = kept
            if not accept:
                if level + 1 < len(thresholds):
                    level += 1
                    self.say(f"round {rnd}: nothing above {tau:.2f}, lowering threshold to {thresholds[level]:.2f}")
                    continue
                break
            # resolve the most certain first; ties prefer more witnesses
            for post, margin, n_wit, st, top, n_r in accept:
                st.chosen = top
                st.status = "identified"
                st.confidence = post
                st.round = rnd
                if n_wit and ("formula" in top.sources or "reaction" in top.sources) and not top.log_name:
                    st.method = f"reactions ({n_wit} witnesses)"
                elif top.log_name and n_wit:
                    st.method = f"name + reactions ({n_wit} witnesses)"
                elif top.log_name:
                    st.method = "name"
                elif len(st.candidates) == 1:
                    st.method = "only isomer of formula"
                else:
                    st.method = "chemical plausibility"
            self.say(f"round {rnd} (tau={tau:.2f}): identified "
                     + ", ".join(f"{p[3].name}={p[4].smiles}({p[0]:.2f})" for p in accept))
            if level > 0:
                level = 0   # new knowledge: go back to the strict threshold
        self.say(f"done in {time.time() - t0:.1f} s")
        return self

    # ------------------------------------------------------------ audit
    def _audit_reactions(self, name: str) -> list[Reaction]:
        """Witness reactions for an audit, most informative and cheapest first."""
        rxns = self.witness_reactions(name)

        def heavy(r):
            return sum(self.mol_of(s).heavy_count() for s in r.species()
                       if s != name and self.mol_of(s) is not None)

        rxns.sort(key=lambda r: (max(len(r.reactants), len(r.products)) > 2, heavy(r), r.index))
        return rxns

    def audit(self, min_gap: float = 3.0, verbose: bool = False,
              seconds: float = AUDIT_SECONDS) -> list[dict]:
        """Check given structures (``known``) against the reaction and thermo evidence.

        Every known species is put back in play with alternatives from its
        formula, name and reactions; the others stay fixed.  A finding is
        reported when an alternative beats the given structure by more than
        ``min_gap`` log units (e^3 = 20x more likely).

        Work per species is bounded (AUDIT_* limits); species whose budget
        ran out are listed in ``audit_partial`` with how many reactions were
        checked.
        """
        if not self.species:
            self.setup()
        findings = []
        self.audit_partial = []
        known = [st for st in self.species.values() if st.status == "known"]
        t_all = time.time()
        for k, st in enumerate(known, 1):
            t0 = time.time()
            deadline = t0 + seconds
            given = st.chosen
            st.status = "unresolved"
            saved = dict(st.candidates)
            saved_mult = st.multiplicity
            # alternatives take their spin from the name or the ground-state
            # table, not from the given SMILES (which is what is being checked)
            st.multiplicity = spin_from_name(st.name, st.formula)
            verdict = "ok"
            try:
                rxns_all = self._audit_reactions(st.name)
                rxns = rxns_all[:AUDIT_REACTIONS]
                if self.use_names:
                    for mol, w, why in name_candidates(st.name, st.formula):
                        self._add_candidate(st, mol, "name", w, why)
                fkey = hill_formula(st.formula)
                if fkey not in self._iso_cache:
                    self._iso_cache[fkey] = enumerate_isomers(st.formula)
                iso = self._iso_cache[fkey] or []
                for mol in sorted(iso, key=lambda m: -plausibility(m)[0])[:AUDIT_CANDIDATES]:
                    self._add_candidate(st, mol, "formula")
                self.reaction_generated(st, rxns[:AUDIT_GEN_REACTIONS], deadline=deadline)
                # the given structure is scored like any other candidate
                st.multiplicity = saved_mult
                g = self._make_candidate(st, given.mol, "formula")
                st.multiplicity = spin_from_name(st.name, st.formula)
                g.sources = {"given"}
                st.candidates[g.key] = g
                # a structure already given to another species is not offered as
                # the fix: two wrong entries would just point at each other
                others = {s.chosen.key for s in known if s is not st and s.chosen is not None}
                for key in [k for k in st.candidates if k in others and k != g.key]:
                    del st.candidates[key]
                self._normalize_priors(st)
                if len(st.candidates) == 1:
                    verdict = "only structure"
                    continue
                scored, used = self.score_species(
                    st, rxns, max_candidates=AUDIT_CANDIDATES + 1, deadline=deadline,
                    must_keep={g.key})
                if len(used) < len(rxns):
                    self.audit_partial.append({"species": st.name, "reactions_checked": len(used),
                                               "reactions_total": len(rxns_all)})
                    verdict = f"partial ({len(used)}/{len(rxns)} reactions)"
                best = max(scored, key=lambda c: c.score)
                gap = best.score - g.score
                if best.key != g.key and gap > min_gap:
                    rx = {r.index: r for r in self.mech.reactions}
                    diffs = [(rx[i].equation, g.distances.get(i), best.distances.get(i))
                             for i in best.distances if best.distances.get(i) != g.distances.get(i)]
                    findings.append({
                        "species": st.name, "formula": st.formula_str,
                        "given": self.known.get(st.name, given.smiles), "suggested": best.smiles,
                        "log_gap": round(gap, 2), "posterior_suggested": round(best.posterior, 3),
                        "given_thermo": g.thermo_note, "suggested_thermo": best.thermo_note,
                        "reactions": diffs[:12], "witnesses": len(used),
                    })
                    verdict = f"FLAG: evidence prefers {best.smiles} (log-odds {gap:+.1f})"
            finally:
                st.candidates = saved
                st.candidates[given.key] = given
                st.chosen = given
                st.status = "known"
                st.multiplicity = saved_mult
                if verbose:
                    print(f"[audit {k}/{len(known)}] {st.name} {self.known.get(st.name, given.smiles)}: {verdict} "
                          f"({time.time() - t0:.1f} s)", flush=True)
        self.audit_findings = findings
        self.say(f"audit: {len(known)} species in {time.time() - t_all:.0f} s, "
                 f"{len(findings)} flagged, {len(self.audit_partial)} partially checked")
        return findings

    # ------------------------------------------------------------ outputs
    def solve_all_reactions(self, limit: int | None = None) -> dict[int, object]:
        for r in self.mech.reactions[:limit]:
            R = [self.mol_of(s) for s in r.reactants]
            P = [self.mol_of(s) for s in r.products]
            if any(m is None for m in R + P):
                continue
            self.solutions[r.index] = solve_reaction(R, P, max_total=6, names_r=r.reactants,
                                                     names_p=r.products)
        return self.solutions

    def smiles_txt(self) -> str:
        lines = []
        for st in self.species.values():
            if st.resolved and st.chosen is not None:
                smi = st.chosen.smiles
                if st.multiplicity == 1 and st.chosen.mol.multiplicity == 1 and "[" in smi:
                    from .lewis import best_structure
                    if best_structure(st.chosen.mol).unpaired == 0 and _radical_in_low_spin(st.chosen.mol):
                        smi = "singlet" + smi
                lines.append(f"{st.name}\t{smi}")
        return "\n".join(lines) + "\n"

    def summary(self) -> dict:
        counts = Counter(s.status for s in self.species.values())
        return dict(counts)


def _radical_in_low_spin(mol):
    return mol.heavy_count() == 1 and mol.formula().get("H", 0) == 2


def identify(chem: str, thermo: str | None = None, known: str | None = None, **kw) -> Identifier:
    from .chemkin import parse_chemkin

    mech = parse_chemkin(chem, thermo)
    k = read_known_smiles(known) if known else {}
    return Identifier(mech, k, **kw).run()
