"""Lewis structure solver.

Input : sigma connectivity (explicit H), net charge, optionally the number of
        unpaired electrons (multiplicity - 1).
Output: every optimal Lewis structure (= the resonance set), each giving
        bond orders, lone pairs, unpaired (radical) electrons and formal
        charges for every atom.

Electron bookkeeping for an atom with bond-order sum b and formal charge q:

    nonbonding electrons   n = ve - q - b          (must be >= 0)
    electrons around atom  e = 2b + n              (must be <= shell capacity)
    lone pairs / radicals  n = 2*LP + r

Structures are ranked lexicographically by

    1. unpaired-electron "unpairing"  - radicals made by splitting a lone pair
       (only matters when a spin state is requested; it puts the two radicals
       of triplet O2 on different atoms instead of on one)
    2. octet deficiency (missing electrons around H / second-row atoms)
    3. number of formal charges
    4. number of radical electrons
    5. electronegativity term - negative charge prefers electronegative atoms

Pi bonds are searched by branch-and-bound over the edges whose two atoms
still have free valence; formal charges and radical placement are then fixed
by a small dynamic programme so that the net charge and spin are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .elements import ELEMENTS, max_bond_sum
from .molecule import Molecule

PAULING = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58,
    "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Br": 2.96, "I": 2.66,
}


@dataclass
class LewisStructure:
    orders: dict[tuple[int, int], int]
    charges: list[int]
    lone_pairs: list[int]
    radicals: list[int]
    cost: tuple = field(default=())

    def order(self, i: int, j: int) -> int:
        return self.orders.get((i, j) if i < j else (j, i), 0)

    def bond_sum(self, i: int, mol: Molecule) -> int:
        return sum(self.order(i, j) for j in mol.adj[i])

    @property
    def unpaired(self) -> int:
        return sum(self.radicals)

    @property
    def multiplicity(self) -> int:
        return self.unpaired + 1

    def signature(self):
        return (tuple(sorted(self.orders.items())), tuple(self.charges),
                tuple(self.radicals), tuple(self.lone_pairs))

    def pi_bonds(self) -> int:
        return sum(o - 1 for o in self.orders.values() if o > 1)

    def electron_census(self, mol: Molecule) -> dict:
        """Totals of the electron bookkeeping for reporting."""
        bonding_pairs = sum(self.orders.values())
        lp = sum(self.lone_pairs)
        rad = sum(self.radicals)
        ve = sum(ELEMENTS[a.element]["ve"] for a in mol.atoms) - mol.charge
        return {
            "valence_electrons": ve,
            "bonding_pairs": bonding_pairs,
            "sigma_bonds": len(self.orders),
            "pi_bonds": self.pi_bonds(),
            "lone_pairs": lp,
            "unpaired_electrons": rad,
            "formal_charges": sum(1 for q in self.charges if q),
            "check": 2 * bonding_pairs + 2 * lp + rad == ve,
        }

    def atom_table(self, mol: Molecule) -> list[dict]:
        rows = []
        for i, a in enumerate(mol.atoms):
            b = self.bond_sum(i, mol)
            e = 2 * b + 2 * self.lone_pairs[i] + self.radicals[i]
            rows.append({
                "index": i, "element": a.element, "bonds": b,
                "lone_pairs": self.lone_pairs[i], "radicals": self.radicals[i],
                "formal_charge": self.charges[i], "electrons": e,
                "octet": _shell_target(a.element),
            })
        return rows


def _shell_target(el: str) -> int:
    oc = ELEMENTS[el]["octet"]
    return min(8, oc)


class _Opt:
    __slots__ = ("q", "n", "r", "lp", "cost")

    def __init__(self, q, n, r, cost):
        self.q, self.n, self.r = q, n, r
        self.lp = (n - r) // 2
        self.cost = cost


def _allowed_charges(mol: Molecule, i: int) -> tuple[int, ...]:
    el = mol.atoms[i].element
    ch = ELEMENTS[el]["charges"]
    if mol.charge and el == "H" and all(a.element == "H" for a in mol.atoms):
        return (0, 1, -1)
    return ch


def _atom_options(el: str, b: int, charges, spin_free: bool):
    """All (charge, radical) choices for an atom whose bond sum is b."""
    ve = ELEMENTS[el]["ve"]
    oc = ELEMENTS[el]["octet"]
    target = _shell_target(el)
    chi = PAULING.get(el, 2.5)
    out = []
    for q in charges:
        n = ve - q - b
        if n < 0 or 2 * b + n > oc:
            continue
        d = max(0, target - (2 * b + n))
        base = n % 2
        r_values = range(base, n + 1, 2) if spin_free else (base,)
        for r in r_values:
            unpair = (r - base) // 2
            out.append(_Opt(q, n, r, (unpair, d, abs(q), r, int(round(q * chi * 1000)))))
    return out


def _add(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2], c1[3] + c2[3], c1[4] + c2[4])


ZERO = (0, 0, 0, 0, 0)
INF = (10**9, 0, 0, 0, 0)


def solve_lewis(mol: Molecule, charge: int | None = None, unpaired: int | None = None,
                max_structures: int = 32) -> list[LewisStructure]:
    """Return all optimal Lewis structures (resonance forms) of ``mol``."""
    n = mol.n
    if n == 0:
        return []
    if charge is None:
        charge = mol.charge
    atoms = mol.atoms
    deg = [len(mol.adj[i]) for i in range(n)]
    allowed = [_allowed_charges(mol, i) for i in range(n)]
    bmax = []
    for i in range(n):
        el = atoms[i].element
        bm = max(max_bond_sum(el, q) for q in allowed[i])
        if deg[i] > bm:
            return []  # sigma skeleton impossible for this element
        bmax.append(bm)
    cap = [bmax[i] - deg[i] for i in range(n)]
    # a halogen with one bond partner keeps a single bond: CH2Br is [CH2]Br and
    # BrO is [O]Br, not radicals parked on an expanded-octet Br=C / Br=O.
    # Hypervalent halogens (ClO2, ClO3: two or more partners) may still expand.
    for i in range(n):
        if atoms[i].element in ("F", "Cl", "Br", "I") and deg[i] <= 1:
            cap[i] = 0

    # pi-capable edges, ordered by BFS so atoms finish early (better bounds)
    pi_adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j, _ in mol.bonds():
        if cap[i] > 0 and cap[j] > 0 and atoms[i].element != "H" and atoms[j].element != "H":
            pi_adj[i].append(j)
            pi_adj[j].append(i)
    edges: list[tuple[int, int]] = []
    seen_e = set()
    visited = set()
    for s in range(n):
        if s in visited or not pi_adj[s]:
            continue
        queue = [s]
        visited.add(s)
        while queue:
            v = queue.pop(0)
            for u in sorted(pi_adj[v]):
                e = (v, u) if v < u else (u, v)
                if e not in seen_e:
                    seen_e.add(e)
                    edges.append(e)
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
    last_edge = {}
    for k, (i, j) in enumerate(edges):
        last_edge[i] = k
        last_edge[j] = k

    spin_free = unpaired is not None
    opt_cache: dict[tuple[int, int], list[_Opt]] = {}

    def options(i, b):
        key = (i, b)
        if key not in opt_cache:
            opt_cache[key] = _atom_options(atoms[i].element, b, allowed[i], spin_free)
        return opt_cache[key]

    def min_cost(i, b):
        opts = options(i, b)
        return min((o.cost for o in opts), default=INF)

    def range_min(i):
        best = INF
        for b in range(deg[i], bmax[i] + 1):
            c = min_cost(i, b)
            if c < best:
                best = c
        return best

    # atoms never touched by pi edges are finished from the start
    fixed_lb = ZERO
    open_lb = {}
    for i in range(n):
        if i not in last_edge:
            fixed_lb = _add(fixed_lb, min_cost(i, deg[i]))
        else:
            open_lb[i] = range_min(i)
    if fixed_lb[0] >= INF[0]:
        return []

    pi = [0] * len(edges)
    bsum = list(deg)
    best_cost = [None]
    results: list[tuple[tuple, list[int], list[_Opt]]] = []

    def evaluate_full():
        # atoms with a single choice are added directly; the DP only runs
        # over atoms that can carry a charge or split a lone pair
        chosen = [None] * n
        fixed = ZERO
        fq = fr = 0
        multi = []
        for i in range(n):
            opts = options(i, bsum[i])
            if not opts:
                return None
            if len(opts) == 1:
                o = opts[0]
                chosen[i] = o
                fixed = _add(fixed, o.cost)
                fq += o.q
                fr += o.r
            else:
                multi.append(i)
        layers = []
        states = {(fq, fr): fixed}
        for i in multi:
            opts = options(i, bsum[i])
            new = {}
            back = {}
            for (sq, sr), c in states.items():
                for o in opts:
                    key = (sq + o.q, sr + o.r)
                    nc = _add(c, o.cost)
                    cur = new.get(key)
                    if cur is None or nc < cur:
                        new[key] = nc
                        back[key] = ((sq, sr), o)
            layers.append(back)
            states = new
        finals = [(c, k) for k, c in states.items()
                  if k[0] == charge and (unpaired is None or k[1] == unpaired)]
        if not finals:
            return None
        c, k = min(finals)
        for li in range(len(multi) - 1, -1, -1):
            prev, o = layers[li][k]
            chosen[multi[li]] = o
            k = prev
        return c, chosen

    def leaf():
        res = evaluate_full()
        if res is None:
            return
        c, chosen = res
        if best_cost[0] is None or c < best_cost[0]:
            best_cost[0] = c
            results.clear()
        if c == best_cost[0] and len(results) < max_structures * 4:
            results.append((c, list(pi), chosen))

    def lower_bound(done_lb, k):
        lb = done_lb
        for i, c in open_lb.items():
            if last_edge[i] >= k:
                lb = _add(lb, c)
        return lb

    def dfs(k, done_lb):
        if best_cost[0] is not None:
            lb = lower_bound(done_lb, k)
            if lb[:4] > best_cost[0][:4]:
                return
        if k == len(edges):
            leaf()
            return
        i, j = edges[k]
        top = min(2, bmax[i] - bsum[i], bmax[j] - bsum[j])
        for p in range(top, -1, -1):
            pi[k] = p
            bsum[i] += p
            bsum[j] += p
            dl = done_lb
            ok = True
            for a in (i, j):
                if last_edge[a] == k:
                    c = min_cost(a, bsum[a])
                    if c[0] >= INF[0]:
                        ok = False
                    dl = _add(dl, c)
            if ok:
                dfs(k + 1, dl)
            bsum[i] -= p
            bsum[j] -= p
        pi[k] = 0

    dfs(0, fixed_lb)

    structures = []
    seen = set()
    if len(results) > 1:
        ring = smallest_rings(mol)
        pens = [_geometry_penalty(mol, ring, edges, pis) for _, pis, _ in results]
        low = min(pens)
        results = [r for r, p in zip(results, pens) if p == low]
    for c, pis, chosen in results:
        orders = {}
        for (i, j, _) in mol.bonds():
            orders[(i, j)] = 1
        for (e, p) in zip(edges, pis):
            if p:
                orders[e] = 1 + p
        s = LewisStructure(
            orders=orders,
            charges=[o.q for o in chosen],
            lone_pairs=[o.lp for o in chosen],
            radicals=[o.r for o in chosen],
            cost=c,
        )
        sig = s.signature()
        if sig in seen:
            continue
        seen.add(sig)
        structures.append(s)
        if len(structures) >= max_structures:
            break
    return structures


def smallest_rings(mol: Molecule) -> list[int]:
    """Size of the smallest ring through each atom (0 = acyclic atom)."""
    out = [0] * mol.n
    for s in range(mol.n):
        if len(mol.adj[s]) < 2:
            continue
        best = 0
        nbs = list(mol.adj[s])
        # shortest cycle through s: BFS from s, find two branches meeting
        branch = {s: None}
        dist = {s: 0}
        queue = []
        for u in nbs:
            branch[u] = u
            dist[u] = 1
            queue.append(u)
        k = 0
        while k < len(queue):
            v = queue[k]
            k += 1
            if best and dist[v] * 2 > best:
                break
            for u in mol.adj[v]:
                if u == s:
                    continue
                if u not in dist:
                    dist[u] = dist[v] + 1
                    branch[u] = branch[v]
                    queue.append(u)
                elif branch[u] != branch[v]:
                    size = dist[u] + dist[v] + 1
                    if not best or size < best:
                        best = size
        out[s] = best
    return out


def _geometry_penalty(mol, ring, edges, pis) -> int:
    """Count sp centres forced into small rings (allene or alkyne in a ring < 8)."""
    pi_at = [0] * mol.n
    triple = [False] * mol.n
    for (i, j), p in zip(edges, pis):
        pi_at[i] += p
        pi_at[j] += p
        if p == 2:
            triple[i] = triple[j] = True
    pen = 0
    for i in range(mol.n):
        if ring[i] and ring[i] < 8 and (pi_at[i] >= 2 or triple[i]):
            pen += 1
    return pen


def best_structure(mol: Molecule) -> LewisStructure:
    target = None if mol.multiplicity is None else mol.multiplicity - 1
    structs = solve_lewis(mol, unpaired=target) if target is not None else []
    if not structs:
        structs = solve_lewis(mol)
    if not structs:
        raise ValueError(f"no Lewis structure for {mol.formula_string()} {mol.label or ''}")
    return structs[0]


def min_unpaired(mol: Molecule) -> int | None:
    """Unpaired electrons in the best low-spin Lewis structure (None = impossible)."""
    structs = solve_lewis(mol, max_structures=1)
    if not structs:
        return None
    return structs[0].unpaired


def lewis_text(mol: Molecule, s: LewisStructure) -> str:
    """Plain-text electron census, one line per heavy atom."""
    lines = []
    for row in s.atom_table(mol):
        if row["element"] == "H":
            continue
        q = row["formal_charge"]
        lines.append(
            f"{row['element']}{row['index']:<3} bonds={row['bonds']} lone_pairs={row['lone_pairs']} "
            f"radicals={row['radicals']} formal_charge={q:+d} electrons={row['electrons']}/{row['octet']}"
        )
    c = s.electron_census(mol)
    lines.append(
        f"valence e-={c['valence_electrons']}  bonding pairs={c['bonding_pairs']} "
        f"(sigma {c['sigma_bonds']}, pi {c['pi_bonds']})  lone pairs={c['lone_pairs']}  "
        f"unpaired={c['unpaired_electrons']}  formal charges={c['formal_charges']}"
    )
    return "\n".join(lines)
