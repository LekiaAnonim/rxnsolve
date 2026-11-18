"""SMILES reader and writer (no RDKit needed).

Reader: organic subset, bracket atoms (charge, H count, isotopes and
stereo marks are read and ignored), branches, ring closures (``1``..``9``,
``%nn``), bond symbols ``- = # $ : / \\ .`` and aromatic lower-case atoms.
RMG-style spin prefixes ``singlet``/``triplet``/``quintet`` are understood.

Writer: produces a Kekule SMILES from a Lewis structure, so radicals, lone
pairs and formal charges come out as bracket atoms (``[CH3]``, ``[C-]#[O+]``).
"""

from __future__ import annotations

import re

from .elements import ORGANIC_SUBSET, SMILES_VALENCES, standard_valence
from .molecule import Molecule, component_canon

_SPIN_PREFIX = re.compile(r"^\s*(singlet|doublet|triplet|quartet|quintet)\s*", re.I)
_SPIN = {"singlet": 1, "doublet": 2, "triplet": 3, "quartet": 4, "quintet": 5}
_TOKEN = re.compile(
    r"(\[[^\]]+\])|(Cl|Br|B|C|N|O|P|S|F|I|b|c|n|o|p|s|\*)|(%\d\d|\d)|([-=#$:/\\.])|([()])"
)
_BRACKET = re.compile(
    r"^\[(\d+)?([A-Z][a-z]?|[bcnops]|se|as)(@{0,2}|@[A-Z]{2}\d*)?(H\d*)?([+-]+\d*)?(?::\d+)?\]$"
)
_AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S", "se": "Se", "as": "As"}


class SmilesError(ValueError):
    pass


def parse_smiles(text: str) -> Molecule:
    s = text.strip()
    mult = None
    m = _SPIN_PREFIX.match(s)
    if m:
        mult = _SPIN[m.group(1).lower()]
        s = s[m.end():]
    s = s.split()[0] if s else s
    if not s:
        raise SmilesError("empty SMILES")

    mol = Molecule()
    aromatic: set[int] = set()
    arom_bonds: set[frozenset] = set()
    bracket: dict[int, int | None] = {}   # atom -> explicit H count (None if organic)
    prev: int | None = None
    stack: list[int | None] = []
    rings: dict[str, tuple[int, str | None]] = {}
    pending_bond: str | None = None
    pos = 0

    def bond_order(sym, a, b):
        if sym is None:
            return 15 if (a in aromatic and b in aromatic) else 1
        return {"-": 1, "/": 1, "\\": 1, "=": 2, "#": 3, "$": 4, ":": 15}[sym]

    def connect(a, b, sym):
        o = bond_order(sym, a, b)
        if o == 15:
            arom_bonds.add(frozenset((a, b)))
            o = 1
        mol.add_bond(a, b, o)

    while pos < len(s):
        tm = _TOKEN.match(s, pos)
        if not tm:
            raise SmilesError(f"cannot parse {text!r} at position {pos}")
        pos = tm.end()
        br, org, ring, bond, paren = tm.groups()
        if br or org:
            if br:
                bm = _BRACKET.match(br)
                if not bm:
                    raise SmilesError(f"bad bracket atom {br}")
                _iso, sym, _chir, hs, chg = bm.groups()
                arom = sym in _AROMATIC
                el = _AROMATIC.get(sym, sym)
                hcount = 0 if not hs else (int(hs[1:]) if len(hs) > 1 else 1)
                charge = 0
                if chg:
                    if len(chg) > 1 and chg[1:].isdigit():
                        charge = int(chg[1:]) * (1 if chg[0] == "+" else -1)
                    else:
                        charge = chg.count("+") - chg.count("-")
                idx = mol.add_atom(el, charge)
                bracket[idx] = hcount
            else:
                if org == "*":
                    raise SmilesError("wildcard atoms are not supported")
                arom = org in _AROMATIC
                el = _AROMATIC.get(org, org)
                idx = mol.add_atom(el)
                bracket[idx] = None
            if arom:
                aromatic.add(idx)
            if prev is not None:
                connect(prev, idx, pending_bond)
            pending_bond = None
            prev = idx
        elif ring:
            num = ring
            if prev is None:
                raise SmilesError("ring closure before any atom")
            if num in rings:
                other, sym = rings.pop(num)
                connect(other, prev, pending_bond or sym)
            else:
                rings[num] = (prev, pending_bond)
            pending_bond = None
        elif bond:
            if bond == ".":
                prev = None
                pending_bond = None
            else:
                pending_bond = bond
        elif paren == "(":
            stack.append(prev)
        elif paren == ")":
            if not stack:
                raise SmilesError("unbalanced parentheses")
            prev = stack.pop()
    if rings:
        raise SmilesError(f"unclosed ring(s) {sorted(rings)} in {text!r}")

    # implicit / explicit hydrogens and radical counts
    heavy = list(range(mol.n))
    total_radicals = 0
    for i in heavy:
        atom = mol.atoms[i]
        el = atom.element
        n_arom = sum(1 for j in mol.adj[i] if frozenset((i, j)) in arom_bonds)
        order_sum = sum(o for j, o in mol.adj[i].items()
                        if frozenset((i, j)) not in arom_bonds)
        hexp = bracket[i]
        if hexp is None:  # organic subset
            used = order_sum + n_arom
            if i in aromatic and el in ("C", "N", "B", "P") and n_arom:
                used += 1
            vals = SMILES_VALENCES.get(el, (standard_valence(el),))
            target = next((v for v in vals if v >= used), None)
            nh = max(0, (target if target is not None else used) - used)
            radicals = 0
        else:
            nh = hexp
            used = order_sum + 1.5 * n_arom + nh
            std = standard_valence(el, atom.charge)
            if el in ("S", "P", "N") and atom.charge == 0 and used > std:
                vals = SMILES_VALENCES.get(el, (std,))
                std = next((v for v in vals if v >= used), std)
            radicals = max(0, int(round(std - used)))
        atom.radicals = radicals
        total_radicals += radicals
        for _ in range(nh):
            h = mol.add_atom("H")
            mol.add_bond(i, h, 1)
    mol.charge = sum(a.charge for a in mol.atoms)
    if mult is None:
        mult = total_radicals + 1
    mol.multiplicity = mult
    # True only when the text said singlet/triplet/...; otherwise the spin is
    # RMG's bracket convention (every unfilled valence counted as a radical)
    mol.spin_explicit = m is not None
    mol.label = text.strip()
    return mol


# ====================================================================
# Writer
# ====================================================================

def _num(n: int) -> str:
    return str(n) if n < 10 else f"%{n}"


def _atom_text(mol, i, structure, skel_nbrs, n_h):
    el = mol.atoms[i].element
    q = structure.charges[i]
    r = structure.radicals[i]
    if el in ORGANIC_SUBSET and q == 0 and r == 0:
        used = sum(structure.order(i, j) for j in skel_nbrs)
        target = next((v for v in SMILES_VALENCES.get(el, ()) if v >= used), None)
        if target is not None and target - used == n_h:
            return el
    h = "" if n_h == 0 else ("H" if n_h == 1 else f"H{n_h}")
    c = ""
    if q:
        c = ("+" if q > 0 else "-") if abs(q) == 1 else f"{'+' if q > 0 else '-'}{abs(q)}"
    return f"[{el}{h}{c}]"


def _component_smiles(mol, comp, structure):
    _key, order, hmap = component_canon(mol, comp)
    rank = {v: k for k, v in enumerate(order)}
    nodeset = set(order)
    nb = {v: sorted((u for u in mol.adj[v] if u in nodeset), key=rank.__getitem__) for v in order}
    start = min(order, key=lambda v: (len(nb[v]) > 1, rank[v]))

    parent = {start: None}
    visit: list[int] = []
    seen: set[int] = set()
    def dfs(v):
        seen.add(v)
        visit.append(v)
        for u in nb[v]:
            if u not in seen:
                parent[u] = v
                dfs(u)
    dfs(start)
    vpos = {v: k for k, v in enumerate(visit)}
    opens: dict[int, list] = {}
    closes: dict[int, list] = {}
    for a in order:
        for b in nb[a]:
            if vpos[a] < vpos[b] and parent.get(b) != a and parent.get(a) != b:
                opens.setdefault(a, []).append((a, b))
                closes.setdefault(b, []).append((a, b))

    def bsym(a, b):
        return {2: "=", 3: "#", 4: "$"}.get(structure.order(a, b), "")

    nums: dict[tuple, int] = {}
    free = list(range(1, 100))
    out: list[str] = []

    def write(v):
        s = _atom_text(mol, v, structure, nb[v], len(hmap[v]))
        for e in closes.get(v, []):
            n = nums.pop(e)
            s += bsym(*e) + _num(n)
            free.append(n)
            free.sort()
        for e in opens.get(v, []):
            n = free.pop(0)
            nums[e] = n
            s += _num(n)
        out.append(s)
        children = [u for u in nb[v] if parent.get(u) == v]
        for k, u in enumerate(children):
            branch = k < len(children) - 1
            if branch:
                out.append("(")
            out.append(bsym(v, u))
            write(u)
            if branch:
                out.append(")")

    write(start)
    return "".join(out)


def to_smiles(mol: Molecule, structure=None) -> str:
    """Kekule SMILES using the given (or best) Lewis structure."""
    from .lewis import best_structure

    if mol.n == 0:
        return ""
    if structure is None:
        structure = best_structure(mol)
    parts = []
    for comp in mol.components():
        if all(mol.atoms[i].element == "H" for i in comp) or len(comp) == 1:
            if len(comp) == 1:
                i = comp[0]
                el = mol.atoms[i].element
                q = structure.charges[i]
                c = "" if not q else ("+" if q > 0 else "-")
                parts.append(f"[{el}{c}]")
                continue
            if len(comp) == 2:
                parts.append("[H][H]")
                continue
        parts.append(_component_smiles(mol, comp, structure))
    return ".".join(parts)


SPIN_PREFIX = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}


_CANON_SMILES: dict = {}


def canonical_smiles(mol: Molecule) -> str:
    """Cached per structure and spin; see _canonical_smiles."""
    k = (mol.constitution_key(), mol.multiplicity)
    hit = _CANON_SMILES.get(k)
    if hit is None:
        hit = _canonical_smiles(mol)
        _CANON_SMILES[k] = hit
    return hit


def _canonical_smiles(mol: Molecule) -> str:
    """Deterministic SMILES: smallest string over the optimal resonance forms.

    RMG reads bracket atoms as radicals ([CH2] is a triplet, [O] a triplet
    atom).  When the molecule's spin differs from what the string implies, a
    prefix is added: singlet[CH2], singlet[O].
    """
    from .lewis import solve_lewis

    structs = solve_lewis(mol, unpaired=_target_unpaired(mol))
    if not structs:
        structs = solve_lewis(mol)
    if not structs:
        return mol.formula_string()
    # prefer the resonance form with the most benzene-like rings (benzyl as
    # [CH2]-phenyl, not a ring radical), then the shortest/smallest string
    from .isomers import aromatic_rings
    smi, s = min(((to_smiles(mol, x), x) for x in structs[:12]),
                 key=lambda t: (-aromatic_rings(mol, t[1]), len(t[0]), t[0]))
    target = mol.multiplicity or (s.unpaired + 1)
    try:
        implied = parse_smiles(smi).multiplicity
    except SmilesError:
        implied = target
    if implied != target and target in SPIN_PREFIX:
        smi = SPIN_PREFIX[target] + smi
    return smi


def _target_unpaired(mol: Molecule):
    if mol.multiplicity is None:
        return None
    return mol.multiplicity - 1
