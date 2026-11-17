"""Structure hints from CHEMKIN species names.

Three sources, all *soft* evidence (the formula from the thermo card and the
reactions always decide):

1. ``ALIASES``   - names used across GRI-Mech, USC Mech, AramcoMech, LLNL.
2. condensed structural formulas - CH3CH2CHO, HOCH2O, CH3OCH3, (CH3)2CHOH.
   Every inline O/S without hydrogens is tried both in the chain (ether)
   and as a pendant on the previous atom (carbonyl).
3. LLNL n-alkane patterns - C7H15-2, C7H15O2-3, C7H14OOH2-4, C7H14O1-3,
   C7H14-2, NC7KET24, plus N-/I-/S-/T-/P-/A- prefix features.
"""

from __future__ import annotations

import re
from collections import Counter

from .lewis import solve_lewis
from .molecule import Molecule
from .smiles import parse_smiles

ALIASES = {
    # H/O
    "H": "[H]", "H2": "[H][H]", "O": "[O]", "O2": "[O][O]", "OH": "[OH]", "H2O": "O",
    "HO2": "[O]O", "H2O2": "OO", "O3": "[O-][O+]=O", "O(1D)": "singlet[O]", "O1D": "singlet[O]",
    "OH*": "[OH]", "OHV": "[OH]", "OH(A)": "[OH]",
    # inert
    "N2": "N#N", "AR": "[Ar]", "HE": "[He]", "NE": "[Ne]", "KR": "[Kr]", "XE": "[Xe]",
    # C1
    "C": "[C]", "CH": "[CH]", "CH2": "triplet[CH2]", "CH2(S)": "singlet[CH2]",
    "CH2S": "singlet[CH2]", "SCH2": "singlet[CH2]", "CH2*": "singlet[CH2]", "CH2SING": "singlet[CH2]",
    "CH2(T)": "triplet[CH2]", "TCH2": "triplet[CH2]", "CH3": "[CH3]", "CH4": "C",
    "CO": "[C-]#[O+]", "CO2": "O=C=O", "HCO": "[CH]=O", "CH2O": "C=O", "HCHO": "C=O",
    "CH2OH": "[CH2]O", "CH3O": "C[O]", "CH3OH": "CO", "CH3O2": "CO[O]", "CH3OO": "CO[O]",
    "CH3O2H": "COO", "CH3OOH": "COO", "HOCO": "O=[C]O", "OCHO": "[O]C=O",
    "HOCHO": "OC=O", "HCOOH": "OC=O", "HOCH2O": "OC[O]", "HCOH": "O[CH]", "CHOH": "O[CH]",
    "CH2OOH": "[CH2]OO", "HOCH2O2": "OCO[O]", "HOCH2O2H": "OCOO", "HCO3": "O=CO[O]",
    "HOOCHO": "O=COO", "OCHO2": "O=CO[O]",
    # C2
    "C2": "[C]#[C]", "C2H": "[C]#C", "C2H2": "C#C", "H2CC": "[C]=C", "C2H3": "[CH]=C",
    "C2H4": "C=C", "C2H5": "C[CH2]", "C2H6": "CC", "HCCO": "[CH]=C=O", "CH2CO": "C=C=O",
    "HCCOH": "C#CO", "CH2CHO": "[CH2]C=O", "CH3CHO": "CC=O", "CH3HCO": "CC=O",
    "CH3CO": "C[C]=O", "C2H5OH": "CCO", "CH3CH2OH": "CCO", "PC2H4OH": "[CH2]CO",
    "SC2H4OH": "C[CH]O", "CH2CH2OH": "[CH2]CO", "CH3CHOH": "C[CH]O", "CH3CH2O": "CC[O]",
    "C2H5O": "CC[O]", "C2H5O2": "CCO[O]", "C2H5O2H": "CCOO", "C2H5OOH": "CCOO",
    "CH3OCH3": "COC", "CH3OCH2": "CO[CH2]", "CH3OCH2O2": "COCO[O]", "C2H4O": "CC=O",
    "C2H3OH": "C=CO", "CH2CHOH": "C=CO", "C2H4O1-2": "C1CO1", "C2H3O1-2": "[CH]1CO1",
    "OCHCHO": "O=CC=O", "CH3CO3": "CC(=O)O[O]", "CH3CO3H": "CC(=O)OO", "CH3COOH": "CC(=O)O",
    "CH3OCHO": "COC=O", "CH2OCHO": "[CH2]OC=O", "CH3OCO": "CO[C]=O", "HCOCH2": "[CH2]C=O",
    "CH2CH": "[CH]=C", "C2H5CHO": "CCC=O", "C2H5CO": "CC[C]=O", "C2O": "[C]=C=O",
    # C3
    "C3H8": "CCC", "NC3H7": "[CH2]CC", "IC3H7": "C[CH]C", "C3H7-1": "[CH2]CC",
    "C3H7-2": "C[CH]C", "N-C3H7": "[CH2]CC", "I-C3H7": "C[CH]C", "C3H6": "C=CC",
    "AC3H5": "C=C[CH2]", "C3H5-A": "C=C[CH2]", "SC3H5": "CC=[CH]",
    "C3H5-S": "CC=[CH]", "TC3H5": "C=[C]C", "C3H5-T": "C=[C]C", "CH2CHCH2": "C=C[CH2]",
    "PC3H4": "C#CC", "C3H4-P": "C#CC", "AC3H4": "C=C=C", "C3H4-A": "C=C=C",
    "CC3H4": "C1=CC1", "C3H4-C": "C1=CC1", "C3H3": "C#C[CH2]", "H2CCCH": "C#C[CH2]",
    "C3H2": "[CH]=C=[C]", "HCCCHO": "C#CC=O", "C3H6O": "CC(C)=O", "CH3COCH3": "CC(C)=O",
    "CH3COCH2": "CC(=O)[CH2]", "C2H3CHO": "C=CC=O", "C2H3CO": "C=C[C]=O",
    "NC3H7OH": "CCCO", "IC3H7OH": "CC(C)O",
    # C4
    "C4H2": "C#CC#C", "C4H10": "CCCC", "NC4H10": "CCCC", "IC4H10": "CC(C)C",
    "PC4H9": "[CH2]CCC", "NC4H9": "[CH2]CCC", "SC4H9": "C[CH]CC", "IC4H9": "[CH2]C(C)C",
    "TC4H9": "C[C](C)C", "C4H9-1": "[CH2]CCC", "C4H9-2": "C[CH]CC",
    "C4H8-1": "C=CCC", "C4H81": "C=CCC", "C4H8-2": "CC=CC", "C4H82": "CC=CC",
    "IC4H8": "C=C(C)C", "C4H6": "C=CC=C", "C4H6-13": "C=CC=C", "C4H612": "C=C=CC",
    "C4H6-12": "C=C=CC", "C4H6-1": "C#CCC", "C4H6-2": "CC#CC", "C4H4": "C=CC#C",
    "C4H3": "[CH]=CC#C", "NC4H3": "[CH]=CC#C", "IC4H3": "C=[C]C#C", "C4H5-N": "[CH]=CC=C",
    "NC4H5": "[CH]=CC=C", "IC4H5": "C=[C]C=C", "C4H5-I": "C=[C]C=C",
    "C4H71-3": "C=C[CH]C", "C4H7-1": "C=CC[CH2]",
    "IC4H7": "C=C(C)[CH2]", "C4H8O": "CCCC=O", "NC3H7CHO": "CCCC=O", "IC3H7CHO": "CC(C)C=O",
    "C4H9OH": "CCCCO", "NC4H9OH": "CCCCO",
    # aromatics
    "C6H6": "c1ccccc1", "A1": "c1ccccc1", "BENZENE": "c1ccccc1", "C6H5": "[c]1ccccc1",
    "A1-": "[c]1ccccc1", "C6H5O": "[O]c1ccccc1", "C6H5OH": "Oc1ccccc1", "A1OH": "Oc1ccccc1",
    "C7H8": "Cc1ccccc1", "A1CH3": "Cc1ccccc1", "C6H5CH3": "Cc1ccccc1", "TOLUENE": "Cc1ccccc1",
    "C6H5CH2": "[CH2]c1ccccc1", "A1CH2": "[CH2]c1ccccc1", "C7H7": "[CH2]c1ccccc1",
    "C6H5CHO": "O=Cc1ccccc1", "A1CHO": "O=Cc1ccccc1", "C6H5CO": "O=[C]c1ccccc1",
    "C6H5CH2OH": "OCc1ccccc1", "C10H8": "c1ccc2ccccc2c1", "A2": "c1ccc2ccccc2c1",
    "C5H5": "[CH]1C=CC=C1", "C5H6": "C1=CCC=C1",
    "C6H4O2": "O=C1C=CC(=O)C=C1", "C6H5C2H": "C#Cc1ccccc1", "A1C2H": "C#Cc1ccccc1",
    "C6H5C2H3": "C=Cc1ccccc1", "A1C2H3": "C=Cc1ccccc1", "STYRENE": "C=Cc1ccccc1",
    # nitrogen
    "N": "[N]", "NH": "[NH]", "NH2": "[NH2]", "NH3": "N", "NNH": "[N]=N", "NO": "[N]=O",
    "NO2": "[O]N=O", "N2O": "[N-]=[N+]=O", "HNO": "N=O", "HON": "[N]O", "HONO": "ON=O",
    "HNO2": "O=[NH+][O-]", "HNOH": "[NH]O", "NH2OH": "NO", "N2H2": "N=N", "N2H3": "[NH]N",
    "N2H4": "NN", "H2NN": "[N-]=[NH2+]", "CN": "[C]#N", "HCN": "C#N", "HNC": "[C-]#[NH+]",
    "H2CN": "C=[N]", "HCNN": "[CH]=[N+]=[N-]", "HCNO": "C#[N+][O-]", "HOCN": "N#CO",
    "HNCO": "N=C=O", "NCO": "[N]=C=O", "CH3CN": "CC#N", "CH2CN": "[CH2]C#N",
    "NO3": "[O][N+](=O)[O-]", "HNO3": "O[N+](=O)[O-]",
    "CH3NO2": "C[N+](=O)[O-]", "CH3ONO": "CON=O", "CH3NO": "CN=O",
    # sulfur / halogen
    "SO2": "O=S=O", "SO": "[S]=O", "H2S": "S", "HS": "[SH]", "S": "[S]", "SO3": "O=S(=O)=O",
    "HCL": "Cl", "CL": "[Cl]", "CL2": "ClCl", "HBR": "Br", "BR": "[Br]", "BR2": "BrBr",
    "HF": "F", "F": "[F]", "F2": "FF", "CLO": "[O]Cl", "HOCL": "OCl",
}
ALIASES = {k.upper(): v for k, v in ALIASES.items()}

_ELEM_TOKENS = ["CL", "BR", "AR", "HE", "NE", "C", "H", "O", "N", "S", "F"]


def normalize(name: str) -> str:
    return name.strip().upper()


# ====================================================================
# 1. aliases
# ====================================================================

def alias_structure(name: str) -> Molecule | None:
    s = ALIASES.get(normalize(name))
    if not s:
        return None
    mol = parse_smiles(s)
    mol.label = name
    return mol


# ====================================================================
# 2. condensed structural formulas
# ====================================================================

def _tokenize(name: str):
    s = normalize(name)
    for grp, tag in _GROUPS:
        s = s.replace(grp, "{" + tag + "}")
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            end = s.index("}", i)
            tokens.append((s[i + 1:end], 1))
            i = end + 1
            continue
        if ch in "()":
            tokens.append((ch, 1))
            i += 1
            m = re.match(r"\d+", s[i:]) if ch == ")" else None
            if m:
                tokens[-1] = (")", int(m.group()))
                i += m.end()
            continue
        el = None
        for t in _ELEM_TOKENS:
            if s.startswith(t, i):
                el = t
                break
        if el is None:
            return None
        i += len(el)
        m = re.match(r"\d+", s[i:])
        n = 1
        if m:
            n = int(m.group())
            i += m.end()
        tokens.append((el, n))
    return tokens


def _el(t):
    return {"CL": "Cl", "BR": "Br", "AR": "Ar", "HE": "He", "NE": "Ne"}.get(t, t)


def condensed_structures(name: str, max_variants: int = 64) -> list[Molecule]:
    """Structures implied by a condensed formula such as CH3CH2CHO."""
    tokens = _tokenize(name)
    if not tokens:
        return []
    # Expand: heavy atom with count > 1 only allowed for O/S chains (O2 -> O-O)
    seq = []   # list of dict(el, h, branch_open/close markers)
    pending_h = 0
    for t, n in tokens:
        if t == "H":
            if seq and seq[-1]["kind"] == "atom":
                seq[-1]["h"] += n
            else:
                pending_h += n
            continue
        if t in "()":
            seq.append({"kind": t, "n": n})
            continue
        if t in _GROUP_BUILDERS:
            seq.append({"kind": "atom", "el": t, "h": 0})
            continue
        if n > 1 and t not in ("O", "S"):
            return []          # molecular formula (C2H5), not condensed
        for k in range(n):
            seq.append({"kind": "atom", "el": _el(t), "h": 0})
        if pending_h:
            # leading H (HOCH2, HCO) belongs to the first heavy atom
            first = next(x for x in seq if x["kind"] == "atom")
            first["h"] += pending_h
            pending_h = 0
    if pending_h:
        return []
    atoms = [x for x in seq if x["kind"] == "atom"]
    if not atoms or len(atoms) > 14:
        return []
    # candidate pendant positions: O/S without H, not first, not last
    pend = [k for k, x in enumerate(seq) if x["kind"] == "atom" and x["el"] in ("O", "S")
            and x["h"] == 0]
    results = {}
    n_var = 0
    for mask in range(1 << len(pend)):
        if n_var >= max_variants:
            break
        n_var += 1
        as_pendant = {pend[b] for b in range(len(pend)) if mask >> b & 1}
        mol = _assemble(seq, as_pendant)
        if mol is None:
            continue
        variants = [(0, mol)] + [(1, ring) for ring in _ring_variants(mol)]
        for n_ring, var in variants:
            structs = solve_lewis(var, max_structures=1)
            if not structs:
                continue
            var.label = name
            cost = structs[0].cost[:3] + (n_ring,)
            k = var.constitution_key()
            if k not in results or cost < results[k][0]:
                results[k] = (cost, var)
    if not results:
        return []
    best = min(c for c, _ in results.values())
    # keep the interpretations with the best electron structure
    return [m for c, m in results.values() if c == best]


def _assemble(seq, as_pendant):
    """Build the chain; X(Y)Z puts Y as a branch on X.  Leading or repeated
    groups such as (CH3)2CH are left to the alias table."""
    mol = Molecule()
    prev = None           # atom the next chain atom bonds to
    stack = []
    for k, x in enumerate(seq):
        kind = x["kind"]
        if kind == "(":
            if prev is None:
                return None
            stack.append(prev)
            continue
        if kind == ")":
            if not stack or x.get("n", 1) > 1:
                return None
            prev = stack.pop()
            continue
        el = x["el"]
        if el in _GROUP_BUILDERS:
            idx = _GROUP_BUILDERS[el](mol)
        else:
            idx = mol.add_atom(el)
            for _ in range(x["h"]):
                h = mol.add_atom("H")
                mol.add_bond(idx, h)
        if prev is not None:
            mol.add_bond(prev, idx)
        if k in as_pendant and prev is not None:
            pass  # pendant: chain continues from prev
        else:
            prev = idx
    return mol


def _ring_variants(mol):
    """One extra bond between two non-adjacent heavy atoms that still have free
    valence: CH2OCH2 -> oxirane, CH3CHOCH2 -> methyloxirane."""
    cap = {"C": 4, "O": 2, "N": 3, "S": 2}
    free = [i for i, a in enumerate(mol.atoms)
            if a.element in cap and len(mol.adj[i]) < cap[a.element]]
    for x in range(len(free)):
        for y in range(x + 1, len(free)):
            i, j = free[x], free[y]
            if mol.has_bond(i, j):
                continue
            m = mol.copy()
            m.add_bond(i, j)
            yield m


def _add_phenyl(mol):
    ring = [mol.add_atom("C") for _ in range(6)]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        mol.add_bond(a, b)
    for c in ring[1:]:
        h = mol.add_atom("H")
        mol.add_bond(c, h)
    return ring[0]


def _add_chain(mol, hs):
    """Carbon chain with the given H counts; returns the attachment atom (first)."""
    cs = [mol.add_atom("C") for _ in hs]
    for a, b in zip(cs, cs[1:]):
        mol.add_bond(a, b)
    for c, n in zip(cs, hs):
        for _ in range(n):
            mol.add_bond(c, mol.add_atom("H"))
    return cs[0]


# multi-carbon groups that are unambiguous inside condensed names
_GROUP_BUILDERS = {
    "Ph": _add_phenyl,
    "Et": lambda m: _add_chain(m, [2, 3]),     # C2H5- : -CH2CH3
    "Vi": lambda m: _add_chain(m, [1, 2]),     # C2H3- : -CH=CH2
}
_GROUPS = [("C6H5", "Ph"), ("C2H5", "Et"), ("C2H3", "Vi")]


# ====================================================================
# 3. LLNL n-alkane naming patterns
# ====================================================================

def _chain(n):
    mol = Molecule()
    cs = [mol.add_atom("C") for _ in range(n)]
    for a, b in zip(cs, cs[1:]):
        mol.add_bond(a, b)
    return mol, cs


def _fill_h(mol, keep_free: dict[int, int] | None = None):
    """Saturate carbons with H, leaving keep_free[c] valences open."""
    keep_free = keep_free or {}
    for i in range(mol.n):
        if mol.atoms[i].element == "C":
            need = 4 - len(mol.adj[i]) - keep_free.get(i, 0)
            for _ in range(max(0, need)):
                h = mol.add_atom("H")
                mol.add_bond(i, h)
        elif mol.atoms[i].element == "O":
            need = 2 - len(mol.adj[i]) - keep_free.get(i, 0)
            for _ in range(max(0, need)):
                h = mol.add_atom("H")
                mol.add_bond(i, h)


def llnl_structures(name: str) -> list[Molecule]:
    s = normalize(name).replace("N-", "N").replace("_", "-")
    out = []
    m = re.fullmatch(r"N?C(\d+)H(\d+)-(\d+)", s)             # alkyl or alkene
    if m:
        n, h, k = (int(x) for x in m.groups())
        mol, cs = _chain(n)
        if h == 2 * n + 1 and 1 <= k <= n:
            _fill_h(mol, {cs[k - 1]: 1})
            out.append(mol)
        elif h == 2 * n and 1 <= k < n:
            _fill_h(mol, {cs[k - 1]: 1, cs[k]: 1})
            out.append(mol)
    m = re.fullmatch(r"N?C(\d+)H(\d+)O2-(\d+)", s)            # RO2
    if m:
        n, h, k = (int(x) for x in m.groups())
        if h == 2 * n + 1 and 1 <= k <= n:
            mol, cs = _chain(n)
            o1 = mol.add_atom("O")
            o2 = mol.add_atom("O")
            mol.add_bond(cs[k - 1], o1)
            mol.add_bond(o1, o2)
            _fill_h(mol, {o2: 1})
            out.append(mol)
    m = re.fullmatch(r"N?C(\d+)H(\d+)O-(\d+)", s)             # RO
    if m:
        n, h, k = (int(x) for x in m.groups())
        if h == 2 * n + 1 and 1 <= k <= n:
            mol, cs = _chain(n)
            o1 = mol.add_atom("O")
            mol.add_bond(cs[k - 1], o1)
            _fill_h(mol, {o1: 1})
            out.append(mol)
    m = re.fullmatch(r"N?C(\d+)H(\d+)OOH(\d+)-(\d+)", s)      # QOOH
    if m:
        n, h, a, b = (int(x) for x in m.groups())
        if h == 2 * n and 1 <= a <= n and 1 <= b <= n and a != b:
            mol, cs = _chain(n)
            o1 = mol.add_atom("O")
            o2 = mol.add_atom("O")
            mol.add_bond(cs[a - 1], o1)
            mol.add_bond(o1, o2)
            _fill_h(mol, {cs[b - 1]: 1})
            out.append(mol)
    m = re.fullmatch(r"N?C(\d+)H(\d+)O(\d+)-(\d+)", s)        # cyclic ether
    if m:
        n, h, a, b = (int(x) for x in m.groups())
        if h == 2 * n and 1 <= a < b <= n:
            mol, cs = _chain(n)
            o = mol.add_atom("O")
            mol.add_bond(cs[a - 1], o)
            mol.add_bond(cs[b - 1], o)
            _fill_h(mol)
            out.append(mol)
    m = re.fullmatch(r"N?C(\d+)KET(\d)(\d)", s)               # ketohydroperoxide
    if m:
        n, a, b = (int(x) for x in m.groups())
        if 1 <= a <= n and 1 <= b <= n and a != b:
            mol, cs = _chain(n)
            ok = mol.add_atom("O")
            mol.add_bond(cs[a - 1], ok)
            o1 = mol.add_atom("O")
            o2 = mol.add_atom("O")
            mol.add_bond(cs[b - 1], o1)
            mol.add_bond(o1, o2)
            _fill_h(mol, {cs[a - 1]: 1, ok: 1})
            out.append(mol)
    for mol in out:
        mol.label = name
    return out


# ====================================================================
# Prefix features (scoring hints, not structures)
# ====================================================================

def name_features(name: str) -> dict:
    s = normalize(name)
    f = {}
    m = re.match(r"^([NISTPA])-?C\d", s)
    if m:
        f["prefix"] = m.group(1)
    return f


def feature_bonus(mol: Molecule, features: dict) -> float:
    """Log-bonus for a candidate that agrees with name prefix features."""
    p = features.get("prefix")
    if not p:
        return 0.0
    from .lewis import solve_lewis
    structs = solve_lewis(mol, max_structures=1)
    if not structs:
        return 0.0
    s = structs[0]
    cnb = {i: sum(1 for j in mol.adj[i] if mol.atoms[j].element == "C")
           for i in range(mol.n) if mol.atoms[i].element == "C"}
    branch = sum(1 for c, k in cnb.items() if k >= 3)
    rad_c = [i for i in cnb if s.radicals[i]]
    rings = mol.n_bonds() - mol.n + 1
    bonus = 0.0
    if p == "N":
        bonus += 1.0 if branch == 0 and rings == 0 else -1.0
        if rad_c:
            bonus += 1.0 if all(cnb[i] <= 1 for i in rad_c) else -0.5
    elif p == "I":
        bonus += 1.0 if branch == 1 else -1.0
    elif p == "S":
        bonus += 1.5 if rad_c and all(cnb[i] == 2 for i in rad_c) else -1.0
    elif p == "T":
        bonus += 1.5 if rad_c and all(cnb[i] == 3 for i in rad_c) else -1.0
    elif p == "P":
        bonus += 1.0 if rad_c and all(cnb[i] <= 1 for i in rad_c) else -0.5
    elif p == "A":
        # allyl radical / allene: radical next to C=C
        allylic = any(any(s.order(j, k) == 2 for k in mol.adj[j]) for i in rad_c for j in mol.adj[i])
        bonus += 1.0 if allylic else -0.5
    return bonus


def name_candidates(name: str, formula: dict | None = None) -> list[tuple[Molecule, float, str]]:
    """(structure, log-weight, reason) suggested by the name, matching the formula."""
    out = []
    seen = set()

    def add(mol, w, why):
        if formula is not None and Counter(mol.formula()) != Counter({k: v for k, v in formula.items() if v}):
            return
        k = mol.constitution_key()
        if k in seen:
            return
        seen.add(k)
        out.append((mol, w, why))

    a = alias_structure(name)
    if a is not None:
        add(a, 4.0, "known name")
    for mol in llnl_structures(name):
        add(mol, 3.0, "LLNL naming pattern")
    for mol in condensed_structures(name):
        add(mol, 2.5, "condensed formula")
    return out
