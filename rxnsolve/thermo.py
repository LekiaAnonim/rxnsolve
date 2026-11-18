"""Thermochemistry evidence: a thermo library and a group-additivity model.

Both are learned from mechanisms whose species are already identified (the
RMG-models folders: SMILES.txt + the NASA polynomials of the same model).

* library    - constitution key -> list of (H298, S298) seen across models.
               A candidate whose library thermo matches the species' NASA
               thermo is very likely right (what RMG's importer does with
               thermo libraries).
* additivity - Benson-style groups (central atom type, H count, radical,
               neighbour types) with back-off levels, fitted by ridge
               regression to H298.  Gives an estimate for any candidate, so
               ring/aromatic/radical-site isomers are separated even when
               reactions cannot tell them apart.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict

from .chemkin import _parse_thermo_lines, _read, Mechanism
from .isomers import aromatic_rings
from .lewis import smallest_rings, solve_lewis
from .molecule import Molecule
from .smiles import parse_smiles

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LIBRARY_FILE = os.path.join(DATA_DIR, "thermo_library.json")
# heat capacity is compared at these temperatures (K)
CP_TEMPS = (300.0, 500.0, 800.0, 1000.0, 1500.0)
R_CAL = 1.987204  # cal/mol/K
# fitted group values, so group additivity still works when the library is missing
GA_FILE = os.path.join(DATA_DIR, "group_additivity.json")
SCP_FILE = os.path.join(DATA_DIR, "entropy_cp_additivity.json")
SIGMA_S = 2.0    # cal/mol/K: held-out S298 error median 1.1, mean 2.1
SIGMA_CP = 1.0   # cal/mol/K: held-out Cp error median 0.4-0.65 over 300-1500 K


# ====================================================================
# Library building
# ====================================================================

def _import_args(folder: str) -> dict:
    script = os.path.join(folder, "import.sh")
    if not os.path.exists(script):
        return {}
    text = open(script, encoding="latin-1").read()
    out = {}
    for flag in ("species", "reactions", "thermo", "known"):
        m = re.search(rf"--{flag}\s+(\S+)", text)
        if m:
            out[flag] = os.path.join(folder, m.group(1))
    return out


_THERMO_NAME = re.compile(r"therm", re.I)
_MECH_NAME = re.compile(r"(mech|kinetic|chem|reaction)", re.I)


def find_model_inputs(folder: str) -> dict:
    """Mechanism, thermo and SMILES files of one model folder.

    import.sh (the RMG importer layout) is read first; when a file it names
    does not exist (a renamed file, a name with a space), the folder is
    searched: *.inp or *mech*/*kinetic* for the mechanism, *therm* for thermo.
    """
    args = _import_args(folder)
    files = sorted(f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    text = [f for f in files if f.lower().endswith((".inp", ".txt", ".dat", ".mech", ".ck", ".chemkin"))]

    def pick(pred):
        for f in text:
            if pred(f):
                return os.path.join(folder, f)
        return None

    chem = args.get("reactions") or args.get("species")
    if not chem or not os.path.exists(chem):
        chem = (pick(lambda f: f.lower().endswith(".inp") and not _THERMO_NAME.search(f))
                or pick(lambda f: _MECH_NAME.search(f) and not _THERMO_NAME.search(f)
                        and not re.search(r"tran", f, re.I) and f != "SMILES.txt"))
    thermo = args.get("thermo")
    if not thermo or not os.path.exists(thermo):
        thermo = pick(lambda f: _THERMO_NAME.search(f) and not re.search(r"tran", f, re.I))
    known = os.path.join(folder, "SMILES.txt")
    return {"chem": chem, "thermo": thermo, "known": known if os.path.exists(known) else None,
            "has_import": "reactions" in args or "species" in args}


def build_library(models_root: str, verbose: bool = False) -> dict:
    """Scan RMG-models style folders; return {"records": [...]}."""
    records = []
    for dirpath, _dirs, files in os.walk(models_root):
        if "SMILES.txt" not in files or "import.sh" not in files:
            continue
        args = find_model_inputs(dirpath)
        args["reactions"] = args["chem"]
        model = os.path.relpath(dirpath, models_root)
        mech = Mechanism()
        for key in ("thermo", "reactions"):
            p = args.get(key)
            if p and os.path.exists(p):
                try:
                    _parse_thermo_lines(_read(p), mech)
                except Exception:  # noqa: BLE001 - a broken file must not stop the scan
                    pass
        n = 0
        for raw in _read(os.path.join(dirpath, "SMILES.txt")):
            parts = raw.split("!")[0].split(None, 1)
            if len(parts) != 2:
                continue
            name, smi = parts[0], parts[1].strip()
            t = mech.thermo.get(name)
            if t is None or t.h298 is None:
                continue
            try:
                mol = parse_smiles(smi)
            except Exception:  # noqa: BLE001
                continue
            if Counter(mol.formula()) != Counter({k: v for k, v in t.composition.items() if v}):
                continue
            h, s = t.h298, t.s298
            if not (-400 < h < 400) or not (0 < s < 400):
                continue
            try:
                cp = [round(t.cp_r(T) * R_CAL, 3) for T in CP_TEMPS]
            except (IndexError, OverflowError):
                cp = None
            records.append({"model": model, "name": name, "smiles": smi,
                            "key": mol.constitution_key(), "mult": mol.multiplicity,
                            "formula": mol.formula_string(),
                            "h298": round(h, 3), "s298": round(s, 3), "cp": cp})
            n += 1
        if verbose:
            print(f"{model}: {n} species with thermo")
    features = {}
    for r in records:
        if r["key"] in features:
            continue
        try:
            features[r["key"]] = dict(group_features(parse_smiles(r["smiles"])))
        except Exception:  # noqa: BLE001
            continue
    return {"records": records, "features": features}


def save_library(lib: dict, path: str = LIBRARY_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(lib, fh)


def load_library(path: str = LIBRARY_FILE) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ====================================================================
# Group features
# ====================================================================

def _atom_types(mol: Molecule, s, arom_atoms: set):
    types = {}
    for i, a in enumerate(mol.atoms):
        if a.element == "H":
            continue
        nd = sum(1 for j in mol.adj[i] if s.order(i, j) == 2)
        nt = sum(1 for j in mol.adj[i] if s.order(i, j) == 3)
        co = any(s.order(i, j) == 2 and mol.atoms[j].element == "O" for j in mol.adj[i])
        if i in arom_atoms:
            t = a.element + "b"
        elif co and a.element == "C":
            t = "CO"
        else:
            t = a.element + "d" * nd + "t" * nt
        types[i] = t
    return types


def _aromatic_atoms(mol: Molecule, s) -> set:
    out = set()
    seen = set()
    for start in range(mol.n):
        if mol.atoms[start].element == "H":
            continue
        stack = [(start, [start])]
        while stack:
            v, path = stack.pop()
            if len(path) == 6:
                if start in mol.adj[v]:
                    cyc = frozenset(path)
                    if cyc in seen:
                        continue
                    seen.add(cyc)
                    ring = path + [start]
                    doubles = sum(1 for x, y in zip(ring, ring[1:]) if s.order(x, y) == 2)
                    if doubles == 3:
                        out.update(path)
                continue
            for u in mol.adj[v]:
                if u not in path and u > start and mol.atoms[u].element != "H":
                    stack.append((u, path + [u]))
    return out


_FEATURES: dict = {}


def group_features(mol: Molecule) -> Counter:
    """Sparse feature vector (feature -> count) for group additivity (cached per structure)."""
    k = (mol.constitution_key(), mol.multiplicity)
    hit = _FEATURES.get(k)
    if hit is None:
        hit = _group_features(mol)
        _FEATURES[k] = hit
    return Counter(hit)


def _group_features(mol: Molecule) -> Counter:
    target = None if mol.multiplicity is None else mol.multiplicity - 1
    structs = solve_lewis(mol, unpaired=target, max_structures=4) if target is not None else []
    if not structs:
        structs = solve_lewis(mol, max_structures=4)
    f = Counter()
    if not structs:
        f["invalid"] = 1
        return f
    s = structs[0]
    arom = _aromatic_atoms(mol, s)
    types = _atom_types(mol, s, arom)
    for i, own in types.items():
        r = s.radicals[i]
        q = s.charges[i]
        lp_extra = ""
        own_s = own + ("." * r) + (f"{q:+d}" if q else "")
        nh = sum(1 for j in mol.adj[i] if mol.atoms[j].element == "H")
        nbt = sorted(types[j] for j in mol.adj[i] if j in types)
        nbe = sorted(mol.atoms[j].element for j in mol.adj[i] if j in types)
        f[f"L1|{own_s}|H{nh}|{','.join(nbt)}"] += 1
        f[f"L2|{own_s}|H{nh}|{','.join(nbe)}"] += 1
        f[f"L3|{own_s}|H{nh}|d{len(nbt)}"] += 1
        f[f"L4|{mol.atoms[i].element}{'.' * r}"] += 1
    ring = smallest_rings(mol)
    for k in (3, 4, 5, 6, 7):
        cnt = sum(1 for x in ring if x == k)
        if cnt:
            f[f"ring{k}"] += cnt / k
    if arom:
        f["aromatic"] += len(arom) / 6.0
    if mol.heavy_count() == 0:
        f[f"Honly{mol.n}"] += 1
    f["const"] += 1
    return f


_REG = {"L1": 2.0, "L2": 1.0, "L3": 0.5, "L4": 0.05, "ring": 0.5, "aromatic": 0.5,
        "const": 0.01, "Honly": 0.01, "invalid": 0.01}


def _reg(feat: str) -> float:
    for k, v in _REG.items():
        if feat.startswith(k):
            return v
    return 1.0


class GroupAdditivity:
    def __init__(self, weights: dict[str, float] | None = None, rms: float = 4.0):
        self.w = weights or {}
        self.rms = rms

    def save(self, path: str = GA_FILE, **meta) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"rms": round(self.rms, 4), **meta,
                       "weights": {k: round(v, 6) for k, v in sorted(self.w.items())}}, fh, indent=0)

    @classmethod
    def load(cls, path: str = GA_FILE) -> "GroupAdditivity | None":
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            d = json.load(fh)
        ga = cls(d["weights"], d.get("rms", 4.0))
        ga.meta = {k: v for k, v in d.items() if k != "weights"}
        return ga

    def predict(self, mol: Molecule) -> float:
        return sum(self.w.get(k, 0.0) * v for k, v in group_features(mol).items())

    @classmethod
    def fit(cls, samples: list[tuple[Counter, float]], iters: int = 400, verbose=False):
        """Ridge regression by conjugate gradient on the normal equations."""
        feats = sorted({k for f, _ in samples for k in f})
        idx = {k: i for i, k in enumerate(feats)}
        rows = [([(idx[k], v) for k, v in f.items()], y) for f, y in samples]
        n = len(feats)
        lam = [_reg(k) for k in feats]

        def A(x):
            out = [lam[i] * x[i] for i in range(n)]
            for cols, _ in rows:
                dot = sum(x[c] * v for c, v in cols)
                for c, v in cols:
                    out[c] += v * dot
            return out

        b = [0.0] * n
        for cols, y in rows:
            for c, v in cols:
                b[c] += v * y
        x = [0.0] * n
        r = b[:]
        p = r[:]
        rs = sum(v * v for v in r)
        for it in range(iters):
            Ap = A(p)
            denom = sum(pi * api for pi, api in zip(p, Ap))
            if denom == 0:
                break
            alpha = rs / denom
            x = [xi + alpha * pi for xi, pi in zip(x, p)]
            r = [ri - alpha * api for ri, api in zip(r, Ap)]
            rs_new = sum(v * v for v in r)
            if rs_new < 1e-10:
                break
            p = [ri + (rs_new / rs) * pi for ri, pi in zip(r, p)]
            rs = rs_new
        w = {feats[i]: x[i] for i in range(n) if abs(x[i]) > 1e-9}
        err = [sum(w.get(k, 0) * v for k, v in f.items()) - y for f, y in samples]
        rms = math.sqrt(sum(e * e for e in err) / max(1, len(err)))
        if verbose:
            print(f"group additivity: {len(samples)} species, {n} features, train RMS {rms:.2f} kcal/mol")
        return cls(w, rms)


# ====================================================================
# Thermo evidence for the identifier
# ====================================================================

class ThermoModel:
    """Library + group additivity, optionally excluding some models."""

    def __init__(self, library: dict | None, exclude_models: set[str] | None = None, verbose=False,
                 ga: "GroupAdditivity | None" = None, scp: "EntropyCpModel | None" = None):
        exclude_models = exclude_models or set()
        if library is None:
            # group additivity only: saved group values, no records to match
            self.by_key, self._index, self._hs = defaultdict(list), defaultdict(list), {}
            self.ga = ga
            self.scp = scp
            self.n_records = 0
            self.n_species = (getattr(ga, "meta", {}) or {}).get("n_species", 0)
            return
        recs = [r for r in library["records"] if r["model"] not in exclude_models
                and not any(r["model"].startswith(e + os.sep) for e in exclude_models)]
        self.by_key: dict[str, list[tuple[float, float]]] = defaultdict(list)
        smiles_of = {}
        for r in recs:
            self.by_key[r["key"]].append((r["h298"], r["s298"]))
            smiles_of.setdefault(r["key"], r["smiles"])
        # per-formula sorted index for "which structures carry this exact thermo?"
        self._index: dict[str, list] = defaultdict(list)
        for r in recs:
            f = r.get("formula")
            if f is None:
                try:
                    f = parse_smiles(r["smiles"]).formula_string()
                except Exception:  # noqa: BLE001
                    continue
            self._index[f].append((r["h298"], r["s298"], r["key"]))
        for f in self._index:
            self._index[f].sort()
        self._hs = {f: [x[0] for x in v] for f, v in self._index.items()}
        self.n_records = len(recs)
        self._recs = recs
        if ga is not None:
            self.ga = ga
            self.n_species = len(self.by_key)
            self.scp = scp
            return
        # one training sample per constitution: median H298
        cached = library.get("features", {})
        samples = []
        for key, vals in self.by_key.items():
            hs = sorted(v[0] for v in vals)
            h = hs[len(hs) // 2]
            feats = cached.get(key)
            if feats is None:
                try:
                    feats = group_features(parse_smiles(smiles_of[key]))
                except Exception:  # noqa: BLE001
                    continue
            samples.append((Counter(feats), h))
        self.ga = GroupAdditivity.fit(samples, verbose=verbose)
        self.n_species = len(samples)
        self.scp = scp if scp is not None else EntropyCpModel.fit_records(recs, cached, smiles_of, verbose)

    def library_match(self, key: str, h: float, s: float | None):
        vals = self.by_key.get(key)
        if not vals:
            return None
        best = min(vals, key=lambda v: abs(v[0] - h) + (0 if s is None else 0.3 * abs(v[1] - s)))
        return best

    def thermo_votes(self, formula: str, h: float, s: float | None,
                     dh: float = 0.6, ds: float = 1.2) -> Counter:
        """Same-formula library records whose thermo matches (h, s), per structure.

        A structure seen with this thermo in 70 models outvotes one seen in a
        single (possibly misidentified) model.
        """
        import bisect
        hs = self._hs.get(formula)
        if not hs:
            return Counter()
        lo = bisect.bisect_left(hs, h - dh)
        hi = bisect.bisect_right(hs, h + dh)
        votes = Counter()
        for hh, ss, key in self._index[formula][lo:hi]:
            if s is None or abs(ss - s) <= ds:
                votes[key] += 1
        return votes

    def library_median(self, key: str) -> float | None:
        vals = self.by_key.get(key)
        if not vals:
            return None
        hs = sorted(v[0] for v in vals)
        return hs[len(hs) // 2]

    def log_likelihood(self, mol: Molecule, key: str, h_mech: float, s_mech: float | None):
        """(log-likelihood, |dH|, explanation) of the species' NASA thermo given mol.

        Robust: a Gaussian on dH mixed with a 0.5 % floor for "the mechanism's
        thermo is simply wrong" (penalty saturates at log 0.005 = -5.3); a
        species whose thermo fits no candidate at all is down-weighted by the
        identifier.
        """
        votes = self.thermo_votes(mol.formula_string(), h_mech, s_mech)
        total = sum(votes.values())
        share = votes.get(key, 0) / total if total else 0.0
        if share >= 0.25:
            return 3.0 * share, 0.0, (f"thermo library match: {votes[key]} of {total} records "
                                     f"with this thermo have this structure")
        med = self.library_median(key)
        if med is not None and len(self.by_key[key]) >= 3:
            dh = med - h_mech
            note = f"library H298 (median of {len(self.by_key[key])}) differs by {dh:+.1f} kcal/mol"
            if share > 0:
                note += f"; minority match ({votes[key]} of {total} records)"
            return _robust(dh, 2.5), abs(dh), note
        pred = self.ga.predict(mol)
        dh = pred - h_mech
        sigma = max(3.0, self.ga.rms * 0.8)
        return _robust(dh, sigma), abs(dh), \
            f"group additivity H298 {pred:.1f} vs mechanism {h_mech:.1f} ({dh:+.1f})"


def _robust(dh: float, sigma: float, eps: float = 0.005) -> float:
    return math.log((1 - eps) * math.exp(-0.5 * (dh / sigma) ** 2) + eps)


class EntropyCpModel:
    """S298 and Cp(T) from structure: group additivity plus spin degeneracy.

    Rings lock bonds that rotate freely in chains, so a cyclic isomer has a
    lower entropy (often 5-15 cal/mol/K) and a different Cp(T) curve. The
    species' own thermo card gives S298 and Cp from 300 to 5000 K; only
    H298 was used before. Nothing here looks up another model's polynomial:
    the group values are fitted once, the comparison is with this card.
    S = sum of groups (fitted on S298 - R ln(2S+1)) + R ln(2S+1).
    """

    def __init__(self, gs: GroupAdditivity, gcs: list, support: dict | None = None):
        self.gs, self.gcs = gs, gcs
        self.support = support or {}    # specific (L1) group -> structures in the fit containing it
        self.meta = {}

    @classmethod
    def fit_records(cls, recs, cached_features, smiles_of, verbose=False) -> "EntropyCpModel | None":
        by = defaultdict(list)
        for r in recs:
            if r.get("cp"):
                by[r["key"]].append(r)
        if not by:
            return None

        def median(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2]

        rows = []
        for key, rs in by.items():
            feats = cached_features.get(key)
            if feats is None:
                try:
                    feats = group_features(parse_smiles(smiles_of.get(key, rs[0]["smiles"])))
                except Exception:  # noqa: BLE001
                    continue
            mult = rs[0].get("mult") or 1
            rows.append((Counter(feats), median([x["s298"] for x in rs]) - R_CAL * math.log(mult),
                         [median([x["cp"][i] for x in rs]) for i in range(len(CP_TEMPS))]))
        gs = GroupAdditivity.fit([(f, s_) for f, s_, _ in rows], verbose=verbose)
        gcs = [GroupAdditivity.fit([(f, c[i]) for f, _, c in rows]) for i in range(len(CP_TEMPS))]
        support = Counter(k for f, _, _ in rows for k in f if k.startswith("L1|"))
        return cls(gs, gcs, dict(support))

    def save(self, path: str = SCP_FILE, **meta) -> None:
        with open(path, "w") as fh:
            json.dump({**meta, "temperatures": list(CP_TEMPS), "support": self.support,
                       "s": {"rms": round(self.gs.rms, 4), "weights": {k: round(v, 6) for k, v in sorted(self.gs.w.items())}},
                       "cp": [{"rms": round(g.rms, 4), "weights": {k: round(v, 6) for k, v in sorted(g.w.items())}}
                              for g in self.gcs]}, fh, indent=0)

    @classmethod
    def load(cls, path: str = SCP_FILE) -> "EntropyCpModel | None":
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            d = json.load(fh)
        m = cls(GroupAdditivity(d["s"]["weights"], d["s"]["rms"]),
                [GroupAdditivity(g["weights"], g["rms"]) for g in d["cp"]], d.get("support", {}))
        m.meta = {k: v for k, v in d.items() if k not in ("s", "cp", "support")}
        return m

    def predict(self, mol: Molecule, multiplicity: int) -> tuple[float, list[float]]:
        s298 = self.gs.predict(mol) + R_CAL * math.log(max(1, multiplicity))
        return s298, [g.predict(mol) for g in self.gcs]

    def min_support(self, mol: Molecule) -> int:
        """How many structures in the fit share this structure's rarest specific group."""
        l1 = [k for k in group_features(mol) if k.startswith("L1|")]
        if not l1 or not self.support:
            return 10**6
        return min(self.support.get(k, 0) for k in l1)

    def log_likelihood(self, mol: Molecule, s_mech: float, cp_mech: list[float], multiplicity: int):
        """(log-likelihood, z, note).

        Uncertainty follows what the fit knows: sigma widens up to 3x when the
        structure has groups the fit saw in few structures (carbenes, benzyne, strained
        rings), and deviations within one sigma count as no evidence, so the
        1-2 cal/mol/K differences between positional isomers do not decide
        anything. Robust floor log 0.005 = -5.3, like the H298 term.
        """
        s_pred, cp_pred = self.predict(mol, multiplicity)
        # a group seen in fewer than 10 structures is poorly fitted: widen sigma up to 3x
        widen = 1.0 + 2.0 * max(0.0, (10 - self.min_support(mol)) / 10.0)
        zs = (s_pred - s_mech) / (SIGMA_S * widen)
        zc2 = sum(((p - c) / (SIGMA_CP * widen)) ** 2 for p, c in zip(cp_pred, cp_mech)) / len(cp_mech)
        z = math.sqrt(zs * zs + zc2)
        excess = max(0.0, z - 1.0)
        ll = math.log(0.995 * math.exp(-0.5 * excess * excess) + 0.005)
        dcp = max(abs(p - c) for p, c in zip(cp_pred, cp_mech))
        note = (f"S298 {s_pred:.1f} vs mechanism {s_mech:.1f} ({s_pred - s_mech:+.1f}); "
                f"Cp within {dcp:.1f} cal/mol/K over {int(CP_TEMPS[0])}-{int(CP_TEMPS[-1])} K"
                + (f"; uncertainty x{widen:.1f} (rare groups in the fit)" if widen > 1.05 else ""))
        return ll, z, note


_DEFAULT = {}


def save_group_additivity(library: dict, path: str = GA_FILE, verbose: bool = False) -> ThermoModel:
    """Fit group additivity on the whole library and store the group values."""
    import time as _time
    m = ThermoModel(library, verbose=verbose)
    meta = dict(n_species=m.n_species, n_records=m.n_records, built=_time.strftime("%Y-%m-%d"))
    m.ga.save(path, **meta)
    if m.scp is not None:
        m.scp.save(SCP_FILE if path == GA_FILE else path.replace(".json", "_s_cp.json"), **meta)
    return m


def default_model(exclude_models: set[str] | None = None, verbose=False) -> ThermoModel | None:
    """Thermo evidence for a run.

    library present           -> library votes + group additivity fitted on it
                                 (refitted without ``exclude_models`` when given)
    library missing, GA file  -> group additivity only, from the saved values
    neither                   -> None: thermochemistry is not used
    """
    key = tuple(sorted(exclude_models or ()))
    if key in _DEFAULT:
        return _DEFAULT[key]
    lib = load_library()
    if lib is None:
        ga = GroupAdditivity.load()
        if ga is None:
            return None
        m = ThermoModel(None, ga=ga, scp=EntropyCpModel.load())
    else:
        saved = None if exclude_models else GroupAdditivity.load()
        saved_scp = None if exclude_models else EntropyCpModel.load()
        if saved is not None and saved.meta.get("n_records") == len(lib["records"]) \
                and saved_scp is not None and saved_scp.meta.get("n_records") == len(lib["records"]):
            m = ThermoModel(lib, ga=saved, scp=saved_scp)      # same library: reuse the saved fits
        else:
            m = ThermoModel(lib, set(exclude_models or ()), verbose=verbose)
    _DEFAULT[key] = m
    return m
