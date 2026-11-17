"""Molecular graph with explicit hydrogens and canonical labelling.

A species is identified by its *constitution*: which atoms are sigma-bonded to
which.  Bond orders, lone pairs, radicals and formal charges are not stored as
identity - they are worked out from the constitution by :mod:`rxnsolve.lewis`
(resonance structures share one constitution).  Spin state (singlet / triplet)
is carried separately in ``Molecule.multiplicity``.

Canonical labelling uses colour refinement plus individualisation-refinement
with automorphism pruning, run on the "skeleton" (every atom except terminal
hydrogens; a terminal H is an H bonded to exactly one non-H atom).  Terminal
hydrogens become a count on their skeleton atom, which removes the large
H-permutation symmetry from the search.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .elements import ELEMENTS, hill_formula, normalize_symbol


class Atom:
    __slots__ = ("element", "charge", "radicals")

    def __init__(self, element: str, charge: int = 0, radicals: int = 0):
        self.element = element
        self.charge = charge
        self.radicals = radicals

    def copy(self) -> "Atom":
        return Atom(self.element, self.charge, self.radicals)

    def __repr__(self) -> str:
        return f"Atom({self.element},{self.charge:+d},r{self.radicals})"


class Molecule:
    """Undirected graph of atoms. ``adj[i]`` maps neighbour -> bond order.

    The bond order stored here is only what the input said (SMILES); the
    solvers never rely on it - they use the sigma connectivity.
    """

    def __init__(self) -> None:
        self.atoms: list[Atom] = []
        self.adj: list[dict[int, int]] = []
        self.multiplicity: int | None = None
        self.charge: int = 0
        self.label: str | None = None
        self._key: str | None = None

    # ------------------------------------------------------------ building
    def add_atom(self, element: str, charge: int = 0, radicals: int = 0) -> int:
        self.atoms.append(Atom(normalize_symbol(element), charge, radicals))
        self.adj.append({})
        self._key = None
        return len(self.atoms) - 1

    def add_bond(self, i: int, j: int, order: int = 1) -> None:
        if i == j:
            raise ValueError("self bond")
        self.adj[i][j] = order
        self.adj[j][i] = order
        self._key = None

    def remove_bond(self, i: int, j: int) -> None:
        del self.adj[i][j]
        del self.adj[j][i]
        self._key = None

    def has_bond(self, i: int, j: int) -> bool:
        return j in self.adj[i]

    def copy(self) -> "Molecule":
        m = Molecule()
        m.atoms = [a.copy() for a in self.atoms]
        m.adj = [dict(d) for d in self.adj]
        m.multiplicity = self.multiplicity
        m.charge = self.charge
        m.label = self.label
        m._key = self._key
        return m

    # ------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def n(self) -> int:
        return len(self.atoms)

    def element(self, i: int) -> str:
        return self.atoms[i].element

    def bonds(self) -> Iterable[tuple[int, int, int]]:
        for i, nb in enumerate(self.adj):
            for j, o in nb.items():
                if i < j:
                    yield i, j, o

    def n_bonds(self) -> int:
        return sum(len(nb) for nb in self.adj) // 2

    def degree(self, i: int) -> int:
        return len(self.adj[i])

    def h_neighbors(self, i: int) -> list[int]:
        return [j for j in self.adj[i] if self.atoms[j].element == "H"]

    def formula(self) -> Counter:
        return Counter(a.element for a in self.atoms)

    def formula_string(self) -> str:
        return hill_formula(self.formula())

    def heavy_count(self) -> int:
        return sum(1 for a in self.atoms if a.element != "H")

    def total_charge(self) -> int:
        return self.charge

    def mass(self) -> float:
        return sum(ELEMENTS[a.element]["mass"] for a in self.atoms)

    def components(self) -> list[list[int]]:
        seen = [False] * self.n
        comps = []
        for s in range(self.n):
            if seen[s]:
                continue
            stack = [s]
            seen[s] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in self.adj[v]:
                    if not seen[u]:
                        seen[u] = True
                        stack.append(u)
            comps.append(sorted(comp))
        return comps

    def is_connected(self) -> bool:
        return self.n <= 1 or len(self.components()) == 1

    def subgraph(self, idx: list[int]) -> tuple["Molecule", dict[int, int]]:
        """Induced subgraph; returns (molecule, old_index -> new_index)."""
        m = Molecule()
        remap = {}
        for old in idx:
            a = self.atoms[old]
            remap[old] = m.add_atom(a.element, a.charge, a.radicals)
        for old in idx:
            for nb, o in self.adj[old].items():
                if nb in remap and remap[old] < remap[nb]:
                    m.add_bond(remap[old], remap[nb], o)
        m.charge = sum(self.atoms[i].charge for i in idx)
        return m, remap

    def split(self) -> list[tuple["Molecule", dict[int, int]]]:
        return [self.subgraph(c) for c in self.components()]

    @staticmethod
    def union(mols: list["Molecule"]) -> tuple["Molecule", list[int]]:
        """Disjoint union; returns (molecule, atom offset of each input)."""
        m = Molecule()
        offsets = []
        for part in mols:
            off = m.n
            offsets.append(off)
            for a in part.atoms:
                m.add_atom(a.element, a.charge, a.radicals)
            for i, j, o in part.bonds():
                m.add_bond(i + off, j + off, o)
            m.charge += part.charge
        return m, offsets

    # ------------------------------------------------------------ identity
    def constitution_key(self) -> str:
        """Canonical string for the sigma constitution (+ net charge)."""
        if self._key is None:
            self._key = constitution_key(self)
        return self._key

    def key_with_spin(self) -> str:
        k = self.constitution_key()
        return k if not self.multiplicity else f"{k}~m{self.multiplicity}"

    def same_species(self, other: "Molecule") -> bool:
        return self.constitution_key() == other.constitution_key()

    def __repr__(self) -> str:
        return f"<Molecule {self.formula_string()} {self.label or ''}>"


# ====================================================================
# Canonical labelling
# ====================================================================

def skeleton(mol: Molecule, comp: list[int] | None = None):
    """Split a (connected) atom set into skeleton atoms and terminal H counts.

    Returns (nodes, hmap) where hmap[node] is the sorted list of terminal
    hydrogens hanging on that node.
    """
    atoms = mol.atoms
    if comp is None:
        comp = list(range(mol.n))
    nodes = []
    hmap: dict[int, list[int]] = {}
    for i in comp:
        a = atoms[i]
        if a.element == "H" and len(mol.adj[i]) == 1:
            (nb,) = mol.adj[i]
            if atoms[nb].element != "H":
                continue  # terminal hydrogen, folded into its neighbour
        nodes.append(i)
    nodeset = set(nodes)
    for v in nodes:
        hmap[v] = sorted(u for u in mol.adj[v] if u not in nodeset)
    return nodes, hmap


def _refine(nodes, colors, nbrs):
    """Colour refinement until the partition is equitable.

    ``colors`` maps node -> int; cell order is preserved (new colours are
    ranks of (old colour, neighbour multiset)).
    """
    ncol = len(set(colors.values()))
    while True:
        sig = {}
        for v in nodes:
            sig[v] = (colors[v], tuple(sorted((e, colors[u]) for u, e in nbrs[v])))
        uniq = sorted(set(sig.values()))
        if len(uniq) == ncol:
            rank = {s: i for i, s in enumerate(uniq)}
            return {v: rank[sig[v]] for v in nodes}
        rank = {s: i for i, s in enumerate(uniq)}
        colors = {v: rank[sig[v]] for v in nodes}
        ncol = len(uniq)


def _individualize(nodes, colors, v):
    cv = colors[v]
    keyed = {u: (colors[u], 0 if (u == v or colors[u] != cv) else 1) for u in nodes}
    uniq = sorted(set(keyed.values()))
    rank = {s: i for i, s in enumerate(uniq)}
    return {u: rank[keyed[u]] for u in nodes}


class _UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


MAX_LEAVES = 20000


def canonical_order(nodes: list[int], labels: dict, nbrs: dict):
    """Canonical ordering of a labelled graph.

    labels[v] : sortable node label
    nbrs[v]   : list of (neighbour, edge_label)
    Returns (certificate, order) where order[i] is the node at canonical
    position i.  Two graphs are isomorphic iff their certificates are equal.
    """
    if not nodes:
        return ((), ()), []
    lab_rank = {l: i for i, l in enumerate(sorted(set(labels[v] for v in nodes)))}
    colors = _refine(nodes, {v: lab_rank[labels[v]] for v in nodes}, nbrs)

    best = {"cert": None, "order": None}
    autos: list[dict] = []
    leaves = [0]

    def certificate(cols):
        order = sorted(nodes, key=cols.__getitem__)
        pos = {v: i for i, v in enumerate(order)}
        edges = []
        for v in order:
            pv = pos[v]
            for u, e in nbrs[v]:
                pu = pos[u]
                if pv < pu:
                    edges.append((pv, pu, e))
        edges.sort()
        return (tuple(labels[v] for v in order), tuple(edges)), order

    def rec(cols, prefix):
        if leaves[0] > MAX_LEAVES:
            return
        cells: dict[int, list[int]] = {}
        for v in nodes:
            cells.setdefault(cols[v], []).append(v)
        target = None
        for c in sorted(cells):
            if len(cells[c]) > 1:
                target = sorted(cells[c])
                break
        if target is None:
            leaves[0] += 1
            cert, order = certificate(cols)
            if best["cert"] is None or cert < best["cert"]:
                best["cert"], best["order"] = cert, order
            elif cert == best["cert"]:
                autos.append({a: b for a, b in zip(best["order"], order)})
            return
        explored: list[int] = []
        for v in target:
            if explored:
                fixing = [g for g in autos if all(g[p] == p for p in prefix)]
                if fixing:
                    uf = _UF(target)
                    for g in fixing:
                        for x in target:
                            y = g[x]
                            if y in uf.p:
                                uf.union(x, y)
                    rv = uf.find(v)
                    if any(uf.find(w) == rv for w in explored):
                        continue
            child = _refine(nodes, _individualize(nodes, cols, v), nbrs)
            rec(child, prefix + [v])
            explored.append(v)

    rec(colors, [])
    return best["cert"], best["order"]


def _cert_to_str(cert) -> str:
    labs, edges = cert
    lab_s = ".".join("".join(str(x) for x in l) for l in labs)
    edge_s = ",".join(f"{a}-{b}" if not e else f"{a}-{b}:{e}" for a, b, e in edges)
    return f"{lab_s}|{edge_s}"


def component_canon(mol: Molecule, comp: list[int], with_orders: bool = False):
    """Canonical (key, skeleton order, hmap) for one connected component."""
    nodes, hmap = skeleton(mol, comp)
    nodeset = set(nodes)
    atoms = mol.atoms
    if with_orders:
        labels = {v: (atoms[v].element, len(hmap[v]), atoms[v].charge, atoms[v].radicals)
                  for v in nodes}
        nbrs = {v: [(u, o) for u, o in mol.adj[v].items() if u in nodeset] for v in nodes}
    else:
        labels = {v: (atoms[v].element, len(hmap[v])) for v in nodes}
        nbrs = {v: [(u, 0) for u in mol.adj[v] if u in nodeset] for v in nodes}
    cert, order = canonical_order(nodes, labels, nbrs)
    return _cert_to_str(cert), order, hmap


def constitution_key(mol: Molecule) -> str:
    keys = sorted(component_canon(mol, c)[0] for c in mol.components())
    k = "+".join(keys)
    if mol.charge:
        k += f"^{mol.charge:+d}"
    return k


def isomorphism(a: Molecule, b: Molecule) -> dict[int, int] | None:
    """Atom mapping a -> b preserving constitution, or None."""
    if a.n != b.n:
        return None
    ca = [(component_canon(a, c), c) for c in a.components()]
    cb = [(component_canon(b, c), c) for c in b.components()]
    ca.sort(key=lambda t: t[0][0])
    cb.sort(key=lambda t: t[0][0])
    if [t[0][0] for t in ca] != [t[0][0] for t in cb]:
        return None
    mapping: dict[int, int] = {}
    for ((_, oa, ha), _), ((_, ob, hb), _) in zip(ca, cb):
        for va, vb in zip(oa, ob):
            mapping[va] = vb
            for h1, h2 in zip(ha[va], hb[vb]):
                mapping[h1] = h2
    return mapping


def degree_of_unsaturation(counts: dict[str, int]) -> float:
    """Rings + pi bonds for a neutral closed-shell formula (half-integer = radical)."""
    tetra = counts.get("C", 0) + counts.get("Si", 0)
    tri = counts.get("N", 0) + counts.get("B", 0) + counts.get("P", 0)
    mono = sum(counts.get(x, 0) for x in ("H", "F", "Cl", "Br", "I", "Li", "Na", "K"))
    return (2 * tetra + 2 + tri - mono) / 2.0
