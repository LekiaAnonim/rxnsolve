"""Reaction solver: minimum chemical distance atom mapping and electron flow.

For a reaction  R1 + R2 + ...  ->  P1 + P2 + ...  we look for the atom-to-atom
mapping that needs the fewest sigma-bond edits (bonds broken + bonds formed).

Search ("meet in the middle"):  delete a bonds from the reactant side and b
bonds from the product side.  If the two remaining graphs are isomorphic,
the reaction is explained with a + b edits and the isomorphism *is* the atom
map.  Isomorphism of a disconnected graph = equal multisets of canonical
component keys, so every deletion state is stored as a sorted tuple of
fragment keys, and per-molecule deletion levels are cached and re-used for
every reaction the species appears in.

With the atom map fixed, the Lewis structures of both sides (best resonance
pair) give the pi-bond changes and the change of lone pairs, radicals and
formal charge on every atom.  The electron flow (fishhook or full arrows) is
then the smallest set of electron moves that turns the reactant electron
distribution into the product one.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

from .elements import hill_formula
from .lewis import LewisStructure, solve_lewis
from .molecule import Molecule, component_canon, isomorphism

MAX_SIDE_DEPTH = 3


# ====================================================================
# Per-molecule bond-deletion levels
# ====================================================================

class _State:
    __slots__ = ("deleted", "mol", "comps")

    def __init__(self, deleted, mol, comps):
        self.deleted = deleted      # tuple of (i, j) bonds removed, in molecule indices
        self.mol = mol              # molecule with those bonds removed
        self.comps = comps          # list of (atom list, key)

    @property
    def key(self):
        return tuple(sorted(k for _, k in self.comps))


class DeletionLevels:
    """All non-isomorphic results of deleting k sigma bonds from one molecule."""

    def __init__(self, mol: Molecule):
        self.mol = mol
        comps = [(c, component_canon(mol, c)[0]) for c in mol.components()]
        root = _State((), mol, comps)
        self.levels: list[dict[tuple, _State]] = [{root.key: root}]

    def level(self, k: int) -> dict[tuple, _State]:
        while len(self.levels) <= k:
            self._grow()
        return self.levels[k]

    def _grow(self):
        prev = self.levels[-1]
        nxt: dict[tuple, _State] = {}
        for st in prev.values():
            m = st.mol
            for ci, (comp, _key) in enumerate(st.comps):
                seen_h_parent = set()
                for i in comp:
                    for j in m.adj[i]:
                        if j < i:
                            continue
                        # bonds to equivalent terminal H: try only one per parent
                        hi = m.atoms[i].element == "H" and len(m.adj[i]) == 1
                        hj = m.atoms[j].element == "H" and len(m.adj[j]) == 1
                        if hi != hj:
                            parent = j if hi else i
                            if parent in seen_h_parent:
                                continue
                            seen_h_parent.add(parent)
                        child = m.copy()
                        child.remove_bond(i, j)
                        sub = _split_component(child, comp)
                        comps = st.comps[:ci] + sub + st.comps[ci + 1:]
                        key = tuple(sorted(k for _, k in comps))
                        if key not in nxt:
                            nxt[key] = _State(st.deleted + ((i, j),), child, comps)
        self.levels.append(nxt)


def _split_component(mol: Molecule, comp: list[int]):
    """Re-split one former component after a bond deletion."""
    compset = set(comp)
    seen = set()
    out = []
    for s in comp:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        part = []
        while stack:
            v = stack.pop()
            part.append(v)
            for u in mol.adj[v]:
                if u in compset and u not in seen:
                    seen.add(u)
                    stack.append(u)
        part.sort()
        out.append((part, component_canon(mol, part)[0]))
    return out


_LEVEL_CACHE: dict[str, DeletionLevels] = {}


def deletion_levels(mol: Molecule) -> DeletionLevels:
    k = mol.constitution_key()
    dl = _LEVEL_CACHE.get(k)
    if dl is None:
        dl = DeletionLevels(mol)
        _LEVEL_CACHE[k] = dl
    return dl


# ====================================================================
# Side states and sigma distance
# ====================================================================

_ISO_CACHE: dict[tuple[int, int], dict] = {}


def _translate(st: _State, mol: Molecule):
    """Deleted bonds of a cached state, renumbered onto ``mol``."""
    base = deletion_levels(mol).mol
    if base is mol:
        return st.deleted
    k = (id(base), id(mol))
    iso = _ISO_CACHE.get(k)
    if iso is None:
        iso = isomorphism(base, mol)
        _ISO_CACHE[k] = iso
    return [(iso[i], iso[j]) for i, j in st.deleted]


def _merge(keys):
    return tuple(sorted(itertools.chain.from_iterable(keys)))


def _compositions(total, parts):
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


_SIDE_CACHE: dict[tuple, dict] = {}


def side_states(mols: list[Molecule], depth: int) -> dict[tuple, tuple]:
    """Merged fragment-key tuples for a side at exactly ``depth`` deletions.

    Value: one representative choice = tuple of _State, one per molecule.
    """
    sig = (tuple(m.constitution_key() for m in mols), depth)
    hit = _SIDE_CACHE.get(sig)
    if hit is not None:
        return hit
    levels = [deletion_levels(m) for m in mols]
    out: dict[tuple, tuple] = {}
    for comp in _compositions(depth, len(mols)):
        per = []
        ok = True
        for dl, k in zip(levels, comp):
            lv = dl.level(k)
            if not lv:
                ok = False
                break
            per.append(list(lv.values()))
        if not ok:
            continue
        for combo in itertools.product(*per):
            key = _merge(st.key for st in combo)
            if key not in out:
                out[key] = combo
    if len(_SIDE_CACHE) > 200000:
        _SIDE_CACHE.clear()
    _SIDE_CACHE[sig] = out
    return out


def side_formula(mols) -> Counter:
    c = Counter()
    for m in mols:
        c.update(m.formula())
    return c


def sigma_distance(reactants: list[Molecule], products: list[Molecule],
                   max_total: int = 5, max_side: int = MAX_SIDE_DEPTH):
    """Minimum number of sigma bonds broken + formed, or None if > max_total.

    Returns (distance, a, b, key) with a bonds deleted on the left and b on
    the right.
    """
    if side_formula(reactants) != side_formula(products):
        return None
    for d in range(max_total + 1):
        for a in range(min(d, max_side) + 1):
            b = d - a
            if b > max_side:
                continue
            left = side_states(reactants, a)
            right = side_states(products, b)
            if len(left) > len(right):
                small, big = right, left
            else:
                small, big = left, right
            for key in small:
                if key in big:
                    return d, a, b, key
    return None


# ====================================================================
# Reaction-driven structure generation
# ====================================================================

_FREE_SIGMA = {"C": 4, "N": 3, "O": 2, "S": 2, "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1,
               "P": 3, "Si": 4, "B": 3}


def _free(mol: Molecule, i: int) -> int:
    return _FREE_SIGMA.get(mol.atoms[i].element, 0) - len(mol.adj[i])


def _fragments_for(combo, molecules, wanted: Counter):
    """Pull fragment molecules whose keys are in ``wanted`` out of a side state."""
    need = Counter(wanted)
    frags = []
    for st in combo:
        for atoms, key in st.comps:
            if need[key] > 0:
                need[key] -= 1
                sub, _ = st.mol.subgraph(atoms)
                # fragments are neutral: formal charges written in the parent's
                # SMILES ([C-]#[O+]) belong to its Lewis structure, not to the
                # pieces; the Lewis solver re-derives them for the new molecule
                for a in sub.atoms:
                    a.charge = 0
                    a.radicals = 0
                sub.charge = 0
                frags.append(sub)
    return frags if sum(need.values()) == 0 else None


def _join_two(a: Molecule, b: Molecule):
    fa = [i for i in range(a.n) if _free(a, i) > 0]
    fb = [j for j in range(b.n) if _free(b, j) > 0]
    seen = set()
    for i in fa:
        for j in fb:
            m, off = Molecule.union([a, b])
            m.add_bond(i, j + off[1])
            k = m.constitution_key()
            if k not in seen:
                seen.add(k)
                yield m


MAX_RING_CLOSURE_HEAVY = 20


def _ring_closures(mol: Molecule):
    """One new ring bond, only where chemistry can close one: between two
    radical centres (biradical -> ring, e.g. cyclic ether formation) or from
    a radical centre onto a pi-bonded atom (intramolecular addition).
    Arbitrary pairs of unsaturated carbons are skipped - in a large aromatic
    they number in the hundreds and never correspond to an elementary step.
    """
    from .lewis import solve_lewis

    if mol.heavy_count() > MAX_RING_CLOSURE_HEAVY:
        return
    structs = solve_lewis(mol, max_structures=1)
    if not structs:
        return
    s = structs[0]
    radical = {i for i in range(mol.n) if s.radicals[i] and mol.atoms[i].element != "H"}
    if not radical:
        return
    pi = {i for i in range(mol.n) if any(s.order(i, j) > 1 for j in mol.adj[i])}
    free = [i for i in range(mol.n) if _free(mol, i) > 0 and mol.atoms[i].element != "H"]
    seen = set()
    for x in range(len(free)):
        for y in range(x + 1, len(free)):
            i, j = free[x], free[y]
            if mol.has_bond(i, j):
                continue
            if not ((i in radical and (j in radical or j in pi)) or
                    (j in radical and i in pi)):
                continue
            m = mol.copy()
            m.add_bond(i, j)
            k = m.constitution_key()
            if k not in seen:
                seen.add(k)
                yield m


def generate_completions(other_side: list[Molecule], same_side_known: list[Molecule],
                         max_edits: int = 4, max_side: int = 2, max_fragments: int = 3,
                         ring_closure: bool = True, limit: int = 400,
                         deadline: float | None = None) -> dict[str, tuple[Molecule, int]]:
    """Structures X that make  other_side -> same_side_known + X  need <= max_edits.

    The unknown is assembled from the fragments left over after the known
    species are carved out of the opposite side, joined with the fewest new
    bonds (plus, optionally, one ring closure).
    """
    import time

    from .lewis import solve_lewis

    results: dict[str, tuple[Molecule, int]] = {}
    for a in range(0, min(max_edits, max_side) + 1):
        X = side_states(other_side, a)
        for b in range(0, min(max_edits - a, max_side) + 1):
            if same_side_known:
                Y = side_states(same_side_known, b)
            else:
                Y = {(): ()} if b == 0 else {}
            for ykey in Y:
                yc = Counter(ykey)
                for xkey, xcombo in X.items():
                    xc = Counter(xkey)
                    if any(xc[k] < v for k, v in yc.items()):
                        continue
                    left = xc - yc
                    m = sum(left.values())
                    if m == 0 or m > max_fragments:
                        continue
                    base = a + b + (m - 1)
                    if base > max_edits:
                        continue
                    if deadline is not None and time.time() > deadline:
                        return results
                    frags = _fragments_for(xcombo, other_side, left)
                    if not frags:
                        continue
                    pool = [frags[0]]
                    for f in frags[1:]:
                        pool = [j for p in pool for j in _join_two(p, f)]
                        if len(pool) > 200:
                            pool = pool[:200]
                    for mol in pool:
                        if deadline is not None and time.time() > deadline:
                            return results
                        _keep(results, mol, base, solve_lewis)
                        if ring_closure and base + 1 <= max_edits and base + 1 <= 3:
                            for ring in _ring_closures(mol):
                                _keep(results, ring, base + 1, solve_lewis)
                    if len(results) > limit:
                        return results
    return results


def _keep(results, mol, edits, solve_lewis):
    if not mol.is_connected():
        return
    k = mol.constitution_key()
    old = results.get(k)
    if old is not None and old[1] <= edits:
        return
    if not solve_lewis(mol, max_structures=1):
        return
    results[k] = (mol, edits)


# ====================================================================
# Full solution: atom map, bond changes, electrons, arrows, class
# ====================================================================

@dataclass
class Arrow:
    kind: str            # "fishhook" (1 e-) or "full" (2 e-)
    source: tuple        # ("atom", i) or ("bond", i, j)
    target: tuple

    def text(self, lab) -> str:
        def fmt(t):
            if t[0] == "atom":
                return lab(t[1])
            return f"{lab(t[1])}-{lab(t[2])} bond"
        sym = "⇀" if self.kind == "fishhook" else "→"
        return f"{fmt(self.source)} {sym} {fmt(self.target)}"


@dataclass
class ReactionSolution:
    reactants: list[Molecule]
    products: list[Molecule]
    balanced: bool
    distance: int | None = None
    atom_map: dict[int, int] = field(default_factory=dict)
    R: Molecule | None = None
    P: Molecule | None = None
    r_owner: list[int] = field(default_factory=list)
    p_owner: list[int] = field(default_factory=list)
    broken: list[tuple[int, int]] = field(default_factory=list)
    formed: list[tuple[int, int]] = field(default_factory=list)
    order_changes: list[tuple[int, int, int, int]] = field(default_factory=list)
    atom_changes: dict[int, dict] = field(default_factory=dict)
    moved: list[dict] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)
    reaction_class: str = "unsolved"
    r_struct: LewisStructure | None = None
    p_struct: LewisStructure | None = None
    notes: list[str] = field(default_factory=list)

    # ---- display helpers
    def label(self, i: int) -> str:
        """Atom label in reactant numbering, e.g. 'C2' (index within its molecule)."""
        el = self.R.atoms[i].element
        owner = self.r_owner[i]
        local = i - self._r_offsets[owner]
        return f"{el}{local + 1}[{self._names_r[owner]}]"

    def summary_lines(self) -> list[str]:
        if not self.balanced:
            return ["not atom-balanced"]
        if self.distance is None:
            return ["no mapping found within the edit limit"]
        out = [f"class: {self.reaction_class}",
               f"sigma edits: {self.distance} ({len(self.broken)} broken, {len(self.formed)} formed)"]
        for i, j in self.broken:
            out.append(f"  bond broken : {self.label(i)} - {self.label(j)}")
        for i, j in self.formed:
            out.append(f"  bond formed : {self.label(i)} - {self.label(j)}")
        for i, j, o1, o2 in self.order_changes:
            out.append(f"  bond order  : {self.label(i)} - {self.label(j)}  {o1} -> {o2}")
        for mv in self.moved:
            out.append(f"  atoms moved : {mv['atoms']} from {mv['from']} to {mv['to']}")
        for i, ch in sorted(self.atom_changes.items()):
            parts = [f"{k} {v:+d}" for k, v in ch.items() if v]
            if parts:
                out.append(f"  electrons   : {self.label(i)}  " + ", ".join(parts))
        for a in self.arrows:
            out.append(f"  arrow       : {a.text(self.label)}")
        return out


_LEWIS_CACHE: dict[tuple, list[LewisStructure]] = {}


def lewis_set(mol: Molecule) -> list[LewisStructure]:
    k = (mol.constitution_key(), mol.multiplicity)
    hit = _LEWIS_CACHE.get(k)
    if hit is not None and _same_layout(hit, mol):
        return hit
    target = None if mol.multiplicity is None else mol.multiplicity - 1
    res = solve_lewis(mol, unpaired=target, max_structures=12) if target is not None else []
    if not res:
        res = solve_lewis(mol, max_structures=12)
    _LEWIS_CACHE[k] = res
    return res


def _same_layout(structs, mol):
    # cached structures use atom indices of the molecule they were made for;
    # only re-use when bond set matches exactly
    if not structs:
        return True
    return set(structs[0].orders) == {(i, j) for i, j, _ in mol.bonds()}


def _union_structure(mols, offsets, structs) -> LewisStructure:
    orders, charges, lps, rads = {}, [], [], []
    for m, off, s in zip(mols, offsets, structs):
        for (i, j), o in s.orders.items():
            orders[(i + off, j + off)] = o
        charges += s.charges
        lps += s.lone_pairs
        rads += s.radicals
    return LewisStructure(orders, charges, lps, rads)


def solve_reaction(reactants: list[Molecule], products: list[Molecule],
                   max_total: int = 6, names_r=None, names_p=None) -> ReactionSolution:
    sol = ReactionSolution(reactants, products,
                           balanced=side_formula(reactants) == side_formula(products))
    sol._names_r = names_r or [m.label or m.formula_string() for m in reactants]
    sol._names_p = names_p or [m.label or m.formula_string() for m in products]
    R, r_off = Molecule.union(reactants)
    P, p_off = Molecule.union(products)
    sol.R, sol.P = R, P
    sol._r_offsets, sol._p_offsets = r_off, p_off
    sol.r_owner = [k for k, m in enumerate(reactants) for _ in range(m.n)]
    sol.p_owner = [k for k, m in enumerate(products) for _ in range(m.n)]
    if not sol.balanced:
        sol.reaction_class = "unbalanced"
        return sol
    res = sigma_distance(reactants, products, max_total=max_total)
    if res is None:
        sol.reaction_class = f"complex (> {max_total} sigma edits)"
        return sol
    d = res[0]
    sol.distance = d
    # among all mappings with the minimum number of edits, prefer the one that
    # moves hydrogens rather than breaking bonds between heavy atoms
    best = None
    for a in range(min(d, MAX_SIDE_DEPTH) + 1):
        b = d - a
        if b > MAX_SIDE_DEPTH:
            continue
        left_states = side_states(reactants, a)
        right_states = side_states(products, b)
        for key in left_states.keys() & right_states.keys():
            dr = [(i + off, j + off) for st, m, off in zip(left_states[key], reactants, r_off)
                  for (i, j) in _translate(st, m)]
            dp = [(i + off, j + off) for st, m, off in zip(right_states[key], products, p_off)
                  for (i, j) in _translate(st, m)]
            heavy = sum(1 for i, j in dr if R.atoms[i].element != "H" and R.atoms[j].element != "H") + \
                sum(1 for i, j in dp if P.atoms[i].element != "H" and P.atoms[j].element != "H")
            if best is None or heavy < best[0]:
                best = (heavy, dr, dp)
    _, del_r, del_p = best
    Rc = R.copy()
    for i, j in del_r:
        Rc.remove_bond(i, j)
    Pc = P.copy()
    for i, j in del_p:
        Pc.remove_bond(i, j)
    mapping = isomorphism(Rc, Pc)
    if mapping is None:  # should not happen
        sol.notes.append("internal: isomorphism failed")
        return sol
    inv = {v: k for k, v in mapping.items()}
    sol.atom_map = mapping
    sol.broken = [tuple(sorted(e)) for e in del_r]
    sol.formed = [tuple(sorted((inv[i], inv[j]))) for i, j in del_p]

    _electron_analysis(sol, reactants, products, r_off, p_off, mapping, inv)
    _moved_atoms(sol)
    sol.reaction_class = classify(sol)
    return sol


def _electron_analysis(sol, reactants, products, r_off, p_off, mapping, inv):
    r_sets = [lewis_set(m) or [None] for m in reactants]
    p_sets = [lewis_set(m) or [None] for m in products]
    if any(s == [None] for s in r_sets + p_sets):
        sol.notes.append("a species has no valid Lewis structure")
        return
    best = None
    combos_r = list(itertools.islice(itertools.product(*r_sets), 64))
    combos_p = list(itertools.islice(itertools.product(*p_sets), 64))
    R, P = sol.R, sol.P
    pairs = set()
    for i, j, _ in R.bonds():
        pairs.add((i, j))
    for i, j, _ in P.bonds():
        a, b = inv[i], inv[j]
        pairs.add((min(a, b), max(a, b)))
    for cr in combos_r:
        ur = _union_structure(reactants, r_off, cr)
        for cp in combos_p:
            up = _union_structure(products, p_off, cp)
            cost = 0
            for i, j in pairs:
                cost += abs(ur.order(i, j) - up.order(mapping[i], mapping[j]))
            atom_cost = sum(abs(ur.charges[i] - up.charges[mapping[i]]) for i in range(R.n))
            c = (cost, atom_cost)
            if best is None or c < best[0]:
                best = (c, ur, up)
    _, ur, up = best
    sol.r_struct, sol.p_struct = ur, up
    sigma_changed = set(sol.broken) | set(sol.formed)
    for i, j in sorted(pairs):
        o1 = ur.order(i, j)
        o2 = up.order(mapping[i], mapping[j])
        if o1 != o2 and (i, j) not in sigma_changed:
            sol.order_changes.append((i, j, o1, o2))
    for i in range(R.n):
        k = mapping[i]
        ch = {
            "lone_pairs": up.lone_pairs[k] - ur.lone_pairs[i],
            "radicals": up.radicals[k] - ur.radicals[i],
            "charge": up.charges[k] - ur.charges[i],
        }
        if any(ch.values()):
            sol.atom_changes[i] = ch
    sol.arrows = electron_flow(sol, ur, up, pairs)


def electron_flow(sol, ur, up, pairs) -> list[Arrow]:
    """Smallest set of electron moves consistent with the bookkeeping."""
    mapping = sol.atom_map
    events = []  # (i, j, sign) one unit of bond order; sign -1 break, +1 form
    for i, j in sorted(pairs):
        d = up.order(mapping[i], mapping[j]) - ur.order(i, j)
        for _ in range(abs(d)):
            events.append((i, j, 1 if d > 0 else -1))
    dn = {}
    for i in range(sol.R.n):
        k = mapping[i]
        dn[i] = (2 * up.lone_pairs[k] + up.radicals[k]) - (2 * ur.lone_pairs[i] + ur.radicals[i])
    if not events:
        return []
    splits = [(1, 1), (2, 0), (0, 2)]
    best = [None, None]
    n_ev = len(events)
    if n_ev > 9:
        splits_per = [[(1, 1)]] * n_ev
    else:
        splits_per = [splits] * n_ev

    def search(k, bal, chosen, het):
        if best[0] is not None and het >= best[0]:
            return
        if k == n_ev:
            if all(v == 0 for v in bal.values()):
                best[0], best[1] = het, list(chosen)
            return
        i, j, sign = events[k]
        for si, sj in splits_per[k]:
            # a break hands electrons to the atom (nonbonding count rises);
            # a formation takes them from the atom (nonbonding count falls)
            di, dj = (si, sj) if sign < 0 else (-si, -sj)
            bal[i] -= di
            bal[j] -= dj
            chosen.append((si, sj))
            search(k + 1, bal, chosen, het + (0 if si == sj else 1))
            chosen.pop()
            bal[i] += di
            bal[j] += dj

    # bal[i] = remaining change in nonbonding electrons to explain
    bal = dict(dn)
    search(0, bal, [], 0)
    if best[1] is None:
        sol.notes.append("electron flow: no exact two-centre decomposition (charge transfer?)")
        return []
    # build per-atom pools of incoming (from breaks) and outgoing (into forms) electrons
    incoming: dict[int, list] = {}
    outgoing: dict[int, list] = {}
    for (i, j, sign), (si, sj) in zip(events, best[1]):
        bond = ("bond", min(i, j), max(i, j))
        for atom, ne in ((i, si), (j, sj)):
            if ne == 0:
                continue
            if sign < 0:
                incoming.setdefault(atom, []).append((bond, ne))
            else:
                outgoing.setdefault(atom, []).append((bond, ne))
    arrows = []
    for atom in sorted(set(incoming) | set(outgoing)):
        ins = list(incoming.get(atom, []))
        outs = list(outgoing.get(atom, []))
        # pass electrons straight from a broken bond into a formed bond
        while ins and outs:
            (b_in, n_in), (b_out, n_out) = ins[0], outs[0]
            n = min(n_in, n_out)
            arrows.append(Arrow("full" if n == 2 else "fishhook", b_in, b_out))
            ins[0] = (b_in, n_in - n)
            outs[0] = (b_out, n_out - n)
            if ins[0][1] == 0:
                ins.pop(0)
            if outs[0][1] == 0:
                outs.pop(0)
        for b_in, n in ins:
            arrows.append(Arrow("full" if n == 2 else "fishhook", b_in, ("atom", atom)))
        for b_out, n in outs:
            arrows.append(Arrow("full" if n == 2 else "fishhook", ("atom", atom), b_out))
    return arrows


def _moved_atoms(sol):
    R, owner_r, owner_p = sol.R, sol.r_owner, sol.p_owner
    mp = sol.atom_map
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(R.n):
        groups.setdefault((owner_r[i], owner_p[mp[i]]), []).append(i)
    # core of each reactant molecule = its largest group (heavy atoms weigh more)
    core = {}
    for (r, p), atoms in groups.items():
        w = sum(10 if R.atoms[a].element != "H" else 1 for a in atoms)
        if r not in core or w > core[r][1]:
            core[r] = (p, w)
    for (r, p), atoms in sorted(groups.items()):
        if core[r][0] == p:
            continue
        hosts = [rr for rr, (pp, _) in core.items() if pp == p and rr != r]
        frag = Counter(R.atoms[a].element for a in atoms)
        sol.moved.append({
            "atoms": hill_formula(frag),
            "indices": atoms,
            "from": sol._names_r[r],
            "to": (sol._names_r[hosts[0]] if hosts else sol._names_p[p]),
            "product": sol._names_p[p],
        })


def classify(sol: ReactionSolution) -> str:
    nr, np_ = len(sol.reactants), len(sol.products)
    nb, nf = len(sol.broken), len(sol.formed)
    R = sol.R
    pi_up = sum(1 for _, _, o1, o2 in sol.order_changes if o2 > o1)
    pi_down = sum(1 for _, _, o1, o2 in sol.order_changes if o2 < o1)
    moved_h = [m for m in sol.moved if m["atoms"] == "H"]

    def h_involved(bonds):
        return [e for e in bonds if "H" in (R.atoms[e[0]].element, R.atoms[e[1]].element)]

    if sol.distance == 0:
        if sol.order_changes or sol.atom_changes:
            return "spin / electronic state change"
        return "no change (third-body or identity)"
    if nb == 0 and nf == 1:
        if nr == 2 and np_ == 1:
            return "radical addition to multiple bond" if pi_down else "radical recombination"
        return "ring closure" if nr == np_ else "bond formation"
    if nb == 1 and nf == 0:
        if nr == 1 and np_ == 2:
            return "beta-scission" if pi_up else "bond fission (homolysis)"
        return "ring opening" if nr == np_ else "bond cleavage"
    if nb == 1 and nf == 1:
        if nr == 2 and np_ == 2:
            if moved_h:
                return "disproportionation" if pi_up else "H-abstraction"
            if sol.moved:
                return f"{sol.moved[0]['atoms']}-atom transfer"
            return "metathesis"
        if nr == 1 and np_ == 1:
            if moved_h or h_involved(sol.broken) and h_involved(sol.formed):
                return "intramolecular H-migration"
            return "isomerization"
        if nr == 1 and np_ == 2:
            return "intramolecular substitution / elimination"
        if nr == 2 and np_ == 1:
            return "addition with rearrangement"
    if nb >= 2 and nf >= 1 and nr == 1 and np_ == 2:
        if h_involved(sol.broken) and pi_up:
            return "concerted elimination"
        return "molecular elimination"
    if nf >= 2 and nb >= 1 and nr == 2 and np_ == 1:
        return "insertion"
    if nr == 1 and np_ == 1:
        return f"rearrangement ({nb} broken, {nf} formed)"
    return f"multi-bond reaction ({nb} broken, {nf} formed)"
