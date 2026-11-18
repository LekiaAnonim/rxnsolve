"""Constitutional isomer enumeration and chemical plausibility priors.

Every connected graph has a spanning tree, so we

1. grow all non-isomorphic labelled trees on the heavy atoms (one atom at a
   time, deduplicated by canonical key),
2. add 0..R ring-closure edges (R <= degrees of unsaturation),
3. distribute the hydrogens over the free sigma valences,
4. keep the constitutions that have a valid Lewis structure.

The Lewis solution of each candidate also feeds :func:`plausibility`, a
log-prior that penalises what chemists rarely see in combustion mechanisms
(carbenes, charge separation, small rings, peroxide chains, enols...).
"""

from __future__ import annotations

import math
import time
from collections import Counter

from .elements import ELEMENTS
from .lewis import has_lewis, smallest_rings, solve_lewis
from .molecule import Molecule, canonical_order, degree_of_unsaturation

ENUM_MAX_SIGMA = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "Si": 4, "B": 3,
                  "F": 1, "Cl": 1, "Br": 1, "I": 1, "H": 1}


def _max_sigma(el):
    return ENUM_MAX_SIGMA.get(el, ELEMENTS[el]["max_sigma"])


def _skel_key(els, edges):
    nodes = list(range(len(els)))
    nb = {v: [] for v in nodes}
    for a, b in edges:
        nb[a].append((b, 0))
        nb[b].append((a, 0))
    cert, _ = canonical_order(nodes, {v: (els[v],) for v in nodes}, nb)
    return cert


class EnumerationLimit(Exception):
    pass


def enumerate_isomers(counts: dict[str, int], max_results: int = 4000,
                      max_heavy: int = 9, time_budget: float = 8.0,
                      max_rings: int | None = None) -> list[Molecule] | None:
    """All constitutional isomers with a valid Lewis structure.

    Returns None when the formula is too large (limits exceeded), so callers
    can fall back to reaction-driven candidates.
    """
    counts = {k: v for k, v in counts.items() if v}
    n_h = counts.get("H", 0)
    heavy = sorted([el for el, k in counts.items() if el != "H" for _ in range(k)],
                   key=lambda e: (-_max_sigma(e), e))
    t0 = time.time()
    if not heavy:
        return _hydrogen_only(n_h)
    if len(heavy) > max_heavy:
        return None
    for el in heavy:
        if el not in ELEMENTS:
            return None
    dbe = degree_of_unsaturation(counts)
    if len(heavy) >= 7 and dbe >= 3.5:
        # aromatics and other highly unsaturated C7+ species: across every
        # formula in RMG-models, 50 of 52 like this overflow the limits after
        # seconds of work - don't start; name and reaction candidates cover them
        return None
    ring_cap = max(0, int(math.floor(dbe + 1e-9)))
    if max_rings is not None:
        ring_cap = min(ring_cap, max_rings)
    if len(heavy) > 6:
        ring_cap = min(ring_cap, 2)
    if len(heavy) > 8:
        ring_cap = min(ring_cap, 1)

    # ---- 1. trees
    remaining0 = Counter(heavy)
    level = {}
    for el in sorted(remaining0):
        els = (el,)
        level[_skel_key(els, ())] = (els, ())
    for _ in range(len(heavy) - 1):
        nxt = {}
        for els, edges in level.values():
            rem = remaining0 - Counter(els)
            deg = Counter()
            for a, b in edges:
                deg[a] += 1
                deg[b] += 1
            for el in sorted(rem):
                if _max_sigma(el) < 1:
                    continue
                for v in range(len(els)):
                    if deg[v] >= _max_sigma(els[v]):
                        continue
                    new_els = els + (el,)
                    new_edges = edges + ((v, len(els)),)
                    k = _skel_key(new_els, new_edges)
                    if k not in nxt:
                        nxt[k] = (new_els, new_edges)
            if time.time() - t0 > time_budget:
                return None
        level = nxt
    skeletons = dict(level)

    # ---- 2. ring closures
    frontier = dict(level)
    for _ in range(ring_cap):
        nxt = {}
        for els, edges in frontier.values():
            es = set(edges)
            deg = Counter()
            for a, b in edges:
                deg[a] += 1
                deg[b] += 1
            for a in range(len(els)):
                if deg[a] >= _max_sigma(els[a]):
                    continue
                for b in range(a + 1, len(els)):
                    if (a, b) in es or (b, a) in es or deg[b] >= _max_sigma(els[b]):
                        continue
                    new_edges = edges + ((a, b),)
                    k = _skel_key(els, new_edges)
                    if k not in nxt and k not in skeletons:
                        nxt[k] = (els, new_edges)
            if time.time() - t0 > time_budget:
                return None
        skeletons.update(nxt)
        frontier = nxt
        if len(skeletons) > 20 * max_results:
            return None

    # ---- 3. hydrogens, 4. Lewis check
    out: dict[str, Molecule] = {}
    for els, edges in skeletons.values():
        deg = Counter()
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        caps = [_max_sigma(els[v]) - deg[v] for v in range(len(els))]
        if sum(caps) < n_h:
            continue
        for dist in _distribute(n_h, caps):
            mol = Molecule()
            for el in els:
                mol.add_atom(el)
            for a, b in edges:
                mol.add_bond(a, b)
            for v, k in enumerate(dist):
                for _ in range(k):
                    h = mol.add_atom("H")
                    mol.add_bond(v, h)
            key = mol.constitution_key()
            if key in out:
                continue
            if not has_lewis(mol):
                continue
            out[key] = mol
            if len(out) > max_results:
                return None
        if time.time() - t0 > time_budget:
            return None
    return list(out.values())


def _distribute(n, caps, i=0):
    if i == len(caps):
        if n == 0:
            yield ()
        return
    rest = sum(caps[i + 1:])
    lo = max(0, n - rest)
    for k in range(min(n, caps[i]), lo - 1, -1):
        for tail in _distribute(n - k, caps, i + 1):
            yield (k,) + tail


def _hydrogen_only(n_h):
    m = Molecule()
    if n_h == 1:
        m.add_atom("H")
        return [m]
    if n_h == 2:
        a = m.add_atom("H")
        b = m.add_atom("H")
        m.add_bond(a, b)
        return [m]
    return []


# ====================================================================
# Plausibility prior
# ====================================================================

_PLAUSIBILITY: dict = {}


def plausibility(mol: Molecule, min_deficiency: int | None = None) -> tuple[float, list[str]]:
    """Log prior (0 = ordinary) and the reasons for each penalty (cached per structure)."""
    k = (mol.constitution_key(), min_deficiency)
    hit = _PLAUSIBILITY.get(k)
    if hit is None:
        hit = _plausibility(mol, min_deficiency)
        _PLAUSIBILITY[k] = hit
    return hit[0], list(hit[1])


def _plausibility(mol: Molecule, min_deficiency: int | None = None) -> tuple[float, list[str]]:
    structs = solve_lewis(mol, max_structures=4)
    if not structs:
        return -50.0, ["no Lewis structure"]
    s = structs[0]
    unpair, deficiency, charges, radicals, _ = s.cost
    reasons = []
    logp = 0.0
    ve_total = sum(ELEMENTS[a.element]["ve"] for a in mol.atoms) - mol.charge
    odd = ve_total % 2
    base_def = odd if min_deficiency is None else min_deficiency
    extra_def = max(0, deficiency - base_def)
    if extra_def:
        logp -= 4.0 * (extra_def / 2.0)
        reasons.append(f"electron-deficient centre (carbene/nitrene), {extra_def} e- short")
    if charges:
        logp -= 1.5 * charges
        reasons.append(f"{charges} formal charges")
        # carbanion ylides (C- not in a C#X triple bond, as in CO or isocyanides)
        for i, a in enumerate(mol.atoms):
            if a.element == "C" and s.charges[i] < 0 and \
                    not any(s.order(i, j) == 3 for j in mol.adj[i]):
                logp -= 3.0
                reasons.append("carbanion ylide")
                break
    if radicals - odd >= 2:
        logp -= 3.0
        reasons.append("biradical")
    ring = smallest_rings(mol)
    rings3 = sum(1 for r in ring if r == 3)
    rings4 = sum(1 for r in ring if r == 4)
    if rings3:
        logp -= 2.5
        reasons.append("3-membered ring")
    if rings4:
        logp -= 1.5
        reasons.append("4-membered ring")
    # sp centres in small rings
    for i, a in enumerate(mol.atoms):
        if ring[i] and ring[i] < 8:
            pis = sum(s.order(i, j) - 1 for j in mol.adj[i])
            trip = any(s.order(i, j) == 3 for j in mol.adj[i])
            if pis >= 2 or trip:
                logp -= 3.0
                reasons.append("cumulene/alkyne in small ring")
                break
    # heteroatom chains
    oo = 0
    for i, j, _ in mol.bonds():
        ei, ej = mol.atoms[i].element, mol.atoms[j].element
        if ei == "O" and ej == "O":
            oo += 1
    if oo >= 2:
        logp -= 1.0 * (oo - 1)
        reasons.append(f"{oo} O-O bonds")
    for i, a in enumerate(mol.atoms):
        if a.element == "O" and sum(1 for j in mol.adj[i] if mol.atoms[j].element == "O") >= 2:
            logp -= 4.0
            reasons.append("O-O-O chain")
            break
    # enol / ynol (C=C-OH) - less common than the carbonyl tautomer
    for i, a in enumerate(mol.atoms):
        if a.element != "O":
            continue
        hs = sum(1 for j in mol.adj[i] if mol.atoms[j].element == "H")
        cs = [j for j in mol.adj[i] if mol.atoms[j].element == "C"]
        if hs == 1 and cs:
            c = cs[0]
            if any(s.order(c, k) >= 2 and mol.atoms[k].element == "C" for k in mol.adj[c]):
                logp -= 1.0
                reasons.append("enol")
                break
    # cumulated double bonds (allenes, cumulenes) are less common than conjugation
    cumul = sum(1 for i in range(mol.n)
                if mol.atoms[i].element == "C" and sum(1 for j in mol.adj[i] if s.order(i, j) == 2) >= 2)
    if cumul:
        logp -= 0.5 * cumul
        reasons.append("cumulated double bond")
    arom = aromatic_rings(mol, s, ring)
    if arom:
        logp += 2.0 * arom
        reasons.append(f"{arom} aromatic ring(s) (bonus)")
    return logp, reasons


def aromatic_rings(mol: Molecule, s, ring=None) -> int:
    """Count 6-membered rings that carry three alternating ring double bonds."""
    count = 0
    seen = set()
    for start in range(mol.n):
        if mol.atoms[start].element != "C":
            continue
        # walk 6-cycles through start using DFS limited to depth 6
        stack = [(start, [start])]
        while stack:
            v, path = stack.pop()
            if len(path) == 6:
                if start in mol.adj[v]:
                    cyc = frozenset(path)
                    if cyc not in seen:
                        seen.add(cyc)
                        ring_doubles = 0
                        cyc_list = path + [start]
                        for a, b in zip(cyc_list, cyc_list[1:]):
                            if s.order(a, b) == 2:
                                ring_doubles += 1
                        if ring_doubles == 3:
                            count += 1
                continue
            for u in mol.adj[v]:
                if u not in path and mol.atoms[u].element != "H" and u > start:
                    stack.append((u, path + [u]))
    return count
