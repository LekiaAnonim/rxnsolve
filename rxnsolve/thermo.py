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
# fitted group values, so group additivity still works when the library is missing
GA_FILE = os.path.join(DATA_DIR, "group_additivity.json")


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
            records.append({"model": model, "name": name, "smiles": smi,
                            "key": mol.constitution_key(), "mult": mol.multiplicity,
                            "formula": mol.formula_string(),
                            "h298": round(h, 3), "s298": round(s, 3)})
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


def group_features(mol: Molecule) -> Counter:
    """Sparse feature vector (feature -> count) for group additivity."""
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
                 ga: "GroupAdditivity | None" = None):
        exclude_models = exclude_models or set()
        if library is None:
            # group additivity only: saved group values, no records to match
            self.by_key, self._index, self._hs = defaultdict(list), defaultdict(list), {}
            self.ga = ga
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
        if ga is not None:
            self.ga = ga
            self.n_species = len(self.by_key)
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


_DEFAULT = {}


def save_group_additivity(library: dict, path: str = GA_FILE, verbose: bool = False) -> ThermoModel:
    """Fit group additivity on the whole library and store the group values."""
    import time as _time
    m = ThermoModel(library, verbose=verbose)
    m.ga.save(path, n_species=m.n_species, n_records=m.n_records, built=_time.strftime("%Y-%m-%d"))
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
        m = ThermoModel(None, ga=ga)
    else:
        saved = None if exclude_models else GroupAdditivity.load()
        if saved is not None and saved.meta.get("n_records") == len(lib["records"]):
            m = ThermoModel(lib, ga=saved)      # same library: reuse the saved fit
        else:
            m = ThermoModel(lib, set(exclude_models or ()), verbose=verbose)
    _DEFAULT[key] = m
    return m
