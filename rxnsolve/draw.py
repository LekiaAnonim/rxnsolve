"""2D layout and SVG drawing of Lewis structures and arrow-pushing mechanisms.

Layout: classical MDS on "zig-zag" graph distances (bond length 1, 120 degree
chains), refined by stress majorisation; hydrogens, lone pairs and radical
dots are then placed in the widest free angular gaps around each atom.
Colours are CSS classes so the page theme controls them.
"""

from __future__ import annotations

import math
from html import escape

from .lewis import LewisStructure
from .molecule import Molecule

BOND_PX = 42.0


# ====================================================================
# Layout
# ====================================================================

def _graph_dist(nodes, adj):
    idx = {v: k for k, v in enumerate(nodes)}
    n = len(nodes)
    D = [[0] * n for _ in range(n)]
    for s in range(n):
        dist = {nodes[s]: 0}
        q = [nodes[s]]
        k = 0
        while k < len(q):
            v = q[k]
            k += 1
            for u in adj[v]:
                if u in idx and u not in dist:
                    dist[u] = dist[v] + 1
                    q.append(u)
        for v, d in dist.items():
            D[s][idx[v]] = d
        for t in range(n):
            if t != s and nodes[t] not in dist:
                D[s][t] = n  # disconnected (should not happen per component)
    return D


def _zigzag(L):
    if L <= 1:
        return float(L)
    if L % 2 == 0:
        return L * math.sqrt(3) / 2
    return math.sqrt((L * math.sqrt(3) / 2) ** 2 + 0.25)


def _classical_mds(T):
    n = len(T)
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(0.0, 0.0), (T[0][1], 0.0)]
    sq = [[T[i][j] ** 2 for j in range(n)] for i in range(n)]
    rm = [sum(r) / n for r in sq]
    tm = sum(rm) / n
    B = [[-0.5 * (sq[i][j] - rm[i] - rm[j] + tm) for j in range(n)] for i in range(n)]
    vecs = []
    for k in range(2):
        v = [math.sin(1.7 * (i + 1) + k) for i in range(n)]
        lam = 0.0
        for _ in range(200):
            w = [sum(B[i][j] * v[j] for j in range(n)) for i in range(n)]
            for (u, lu) in vecs:  # deflate
                d = sum(wi * ui for wi, ui in zip(w, u))
                w = [wi - d * ui for wi, ui in zip(w, u)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / norm for x in w]
            lam = norm
        vecs.append((v, lam))
    return [(vecs[0][0][i] * math.sqrt(max(vecs[0][1], 1e-9)),
             vecs[1][0][i] * math.sqrt(max(vecs[1][1], 1e-9))) for i in range(n)]


def _stress(pos, T, iters=150):
    n = len(pos)
    if n < 3:
        return pos
    W = [[0.0 if i == j else 1.0 / (T[i][j] ** 2) for j in range(n)] for i in range(n)]
    X = [list(p) for p in pos]
    for _ in range(iters):
        new = []
        for i in range(n):
            sx = sy = sw = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = X[i][0] - X[j][0]
                dy = X[i][1] - X[j][1]
                d = math.hypot(dx, dy) or 1e-6
                w = W[i][j]
                sx += w * (X[j][0] + T[i][j] * dx / d)
                sy += w * (X[j][1] + T[i][j] * dy / d)
                sw += w
            new.append([sx / sw, sy / sw])
        X = new
    return [tuple(p) for p in X]


def layout(mol: Molecule, show_h: bool = True) -> dict[int, tuple[float, float]]:
    """Coordinates (bond length = 1) for every drawn atom.

    Small molecules are laid out with their hydrogens (no H clashes); larger
    ones place hydrogens afterwards in the free angular gaps.
    """
    if show_h and max((len(c) for c in mol.components()), default=0) <= 22:
        heavy = list(range(mol.n))
    else:
        heavy = [i for i in range(mol.n) if mol.atoms[i].element != "H"
                 or all(mol.atoms[j].element == "H" for j in mol.adj[i])]
    pos: dict[int, tuple[float, float]] = {}
    x_off = 0.0
    comps = []
    hs = set(heavy)
    seen = set()
    for s in heavy:
        if s in seen:
            continue
        comp = []
        st = [s]
        seen.add(s)
        while st:
            v = st.pop()
            comp.append(v)
            for u in mol.adj[v]:
                if u in hs and u not in seen:
                    seen.add(u)
                    st.append(u)
        comps.append(sorted(comp))
    for comp in comps:
        D = _graph_dist(comp, mol.adj)
        T = [[_zigzag(D[i][j]) for j in range(len(comp))] for i in range(len(comp))]
        start = [(x + 0.15 * math.sin(12.9898 * (k + 1)), y + 0.15 * math.cos(78.233 * (k + 1)))
                 for k, (x, y) in enumerate(_classical_mds(T))]
        p = _stress(start, T, iters=250)
        # orient: widest extent horizontal
        if len(p) > 1:
            cx = sum(x for x, _ in p) / len(p)
            cy = sum(y for _, y in p) / len(p)
            sxx = sum((x - cx) ** 2 for x, _ in p)
            syy = sum((y - cy) ** 2 for _, y in p)
            sxy = sum((x - cx) * (y - cy) for x, y in p)
            ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
            ca, sa = math.cos(-ang), math.sin(-ang)
            p = [((x - cx) * ca - (y - cy) * sa, (x - cx) * sa + (y - cy) * ca) for x, y in p]
        minx = min(x for x, _ in p)
        for v, (x, y) in zip(comp, p):
            pos[v] = (x - minx + x_off, y)
        x_off = max(x for x, _ in pos.values()) + 1.8
    if show_h:
        for v in list(pos):
            hlist = [u for u in mol.adj[v] if u not in pos]
            if not hlist:
                continue
            angles = _free_angles(pos, v, [u for u in mol.adj[v] if u in pos], len(hlist))
            for u, a in zip(hlist, angles):
                pos[u] = (pos[v][0] + 0.85 * math.cos(a), pos[v][1] + 0.85 * math.sin(a))
    return pos


def _free_angles(pos, v, neighbors, k):
    """k directions spread through the free space around atom v."""
    if k <= 0:
        return []
    x0, y0 = pos[v]
    occ = sorted(math.atan2(pos[u][1] - y0, pos[u][0] - x0) for u in neighbors)
    if not occ:
        start = math.pi / 2 if k > 1 else 0.0
        return [start + 2 * math.pi * i / k for i in range(k)][:k] if k != 2 else [math.pi * 0.83, math.pi * 0.17 + math.pi]
    gaps = []
    for i, a in enumerate(occ):
        b = occ[(i + 1) % len(occ)] + (2 * math.pi if i == len(occ) - 1 else 0)
        gaps.append([a, b - a, 0])
    for _ in range(k):
        g = max(gaps, key=lambda g: g[1] / (g[2] + 2))
        g[2] += 1
    out = []
    for a, width, cnt in gaps:
        for j in range(cnt):
            out.append(a + width * (j + 1) / (cnt + 1))
    return out


# ====================================================================
# SVG
# ====================================================================

def _fmt(x):
    return f"{x:.1f}"


def molecule_svg_parts(mol: Molecule, s: LewisStructure, pos: dict, origin=(0.0, 0.0),
                       scale=BOND_PX, show_electrons=True, highlight=None, ids=None):
    """SVG elements (strings) for one molecule; returns (parts, pixel positions)."""
    highlight = highlight or {}
    ox, oy = origin
    px = {i: (ox + x * scale, oy - y * scale) for i, (x, y) in pos.items()}
    parts = []
    drawn = set(px)
    # bonds
    for i, j, _ in mol.bonds():
        if i not in drawn or j not in drawn:
            continue
        o = s.order(i, j)
        (x1, y1), (x2, y2) = px[i], px[j]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1
        ux, uy = dx / L, dy / L
        # keep bond lines clear of the atom labels
        sh1 = 7 if mol.atoms[i].element == "H" else 9
        sh2 = 7 if mol.atoms[j].element == "H" else 9
        ax, ay = x1 + ux * sh1, y1 + uy * sh1
        bx, by = x2 - ux * sh2, y2 - uy * sh2
        nx, ny = -uy, ux
        cls = "bond" + (" hl" if (min(i, j), max(i, j)) in highlight.get("bonds", ()) else "")
        offs = {1: [0], 2: [-2.6, 2.6], 3: [-4, 0, 4]}.get(o, [0])
        for d in offs:
            parts.append(f'<line class="{cls}" x1="{_fmt(ax + nx * d)}" y1="{_fmt(ay + ny * d)}" '
                         f'x2="{_fmt(bx + nx * d)}" y2="{_fmt(by + ny * d)}"/>')
    # atoms
    for i in drawn:
        x, y = px[i]
        el = mol.atoms[i].element
        cls = "atom el-" + el + (" hl" if i in highlight.get("atoms", ()) else "")
        parts.append(f'<text class="{cls}" x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="middle" '
                     f'dominant-baseline="central">{escape(el)}</text>')
        if ids is not None and i in ids:
            parts.append(f'<text class="mapno" x="{_fmt(x + 8)}" y="{_fmt(y + 11)}">{ids[i]}</text>')
        q = s.charges[i]
        if q:
            sym = "+" if q > 0 else "−"
            parts.append(f'<circle class="chg" cx="{_fmt(x + 10)}" cy="{_fmt(y - 10)}" r="5.2"/>'
                         f'<text class="chgt" x="{_fmt(x + 10)}" y="{_fmt(y - 10)}" text-anchor="middle" '
                         f'dominant-baseline="central">{sym}</text>')
    if show_electrons:
        for i in drawn:
            lp, r = s.lone_pairs[i], s.radicals[i]
            if not lp and not r:
                continue
            nbrs = [u for u in mol.adj[i] if u in drawn]
            angles = _free_angles(pos, i, nbrs, lp + r)
            x, y = px[i]
            for k, a in enumerate(angles):
                cx = x + 13 * math.cos(a)
                cy = y - 13 * math.sin(a)
                if k < lp:
                    tx, ty = -math.sin(a), -math.cos(a)
                    for d in (-2.6, 2.6):
                        parts.append(f'<circle class="lp" cx="{_fmt(cx + tx * d)}" cy="{_fmt(cy + ty * d)}" r="1.6"/>')
                else:
                    parts.append(f'<circle class="rad" cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="2.3"/>')
    return parts, px


def _bbox(px, pad=24):
    pts = list(px.values()) if isinstance(px, dict) else list(px)
    xs = [p[0] for p in pts] or [0]
    ys = [p[1] for p in pts] or [0]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def lewis_svg(mol: Molecule, s: LewisStructure, show_h: bool | None = None, title: str = "") -> str:
    if show_h is None:
        show_h = mol.n <= 34
    pos = layout(mol, show_h=show_h)
    parts, px = molecule_svg_parts(mol, s, pos)
    x0, y0, x1, y1 = _bbox(px)
    w, h = x1 - x0, y1 - y0
    t = f"<title>{escape(title)}</title>" if title else ""
    return (f'<svg class="lewis" xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(x0)} {_fmt(y0)} '
            f'{_fmt(w)} {_fmt(h)}" width="{_fmt(w)}" height="{_fmt(h)}" role="img">{t}'
            + "".join(parts) + "</svg>")


def reaction_svg(sol) -> str:
    """Reactants with curly arrows, a reaction arrow, then products (atom-mapped)."""
    if sol.r_struct is None or sol.p_struct is None:
        return ""
    R, P = sol.R, sol.P
    show_h = R.n <= 30
    pos_r = layout(R, show_h=True)
    # products: lay out each product molecule, then use the atom map for numbering
    pos_p = layout(P, show_h=True)
    changed_r = {"bonds": set(sol.broken) | {(min(i, j), max(i, j)) for i, j, *_ in sol.order_changes}}
    inv = {v: k for k, v in sol.atom_map.items()}
    changed_p = {"bonds": {tuple(sorted((sol.atom_map[i], sol.atom_map[j]))) for i, j in sol.formed}
                 | {tuple(sorted((sol.atom_map[i], sol.atom_map[j]))) for i, j, *_ in sol.order_changes}}
    # map numbers on atoms that change
    involved = set()
    for i, j in sol.broken + sol.formed:
        involved |= {i, j}
    for i, j, *_ in sol.order_changes:
        involved |= {i, j}
    involved |= set(sol.atom_changes)
    num = {a: k + 1 for k, a in enumerate(sorted(involved))}
    parts_r, px_r = molecule_svg_parts(R, sol.r_struct, pos_r, (0, 0), highlight=changed_r, ids=num)
    x0, y0, x1, y1 = _bbox(px_r, pad=30)
    arrow_x = x1 + 10
    arrow_len = 56
    shift = arrow_x + arrow_len + 30 - min(p[0] for p in layout_px(pos_p).values())
    ids_p = {sol.atom_map[a]: n for a, n in num.items()}
    parts_p, px_p = molecule_svg_parts(P, sol.p_struct, pos_p, (shift, 0), highlight=changed_p, ids=ids_p)
    X0, Y0, X1, Y1 = _bbox(list(px_r.values()) + list(px_p.values()), pad=34)
    mid_y = (Y0 + Y1) / 2
    parts = parts_r + parts_p
    parts.append(f'<line class="rxarrow" x1="{_fmt(arrow_x)}" y1="{_fmt(mid_y)}" '
                 f'x2="{_fmt(arrow_x + arrow_len)}" y2="{_fmt(mid_y)}" marker-end="url(#rxhead)"/>')
    # curly arrows on the reactant side
    for k, a in enumerate(sol.arrows):
        p1 = _anchor(a.source, px_r)
        p2 = _anchor(a.target, px_r)
        if p1 is None or p2 is None:
            continue
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 4:
            p2 = (p2[0] + 6, p2[1] - 6)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1
        bend = 18 + 6 * (k % 3)
        cx, cy = mx - dy / L * bend, my + dx / L * bend
        marker = "fish" if a.kind == "fishhook" else "full"
        parts.append(f'<path class="curly {marker}" d="M{_fmt(p1[0])},{_fmt(p1[1])} '
                     f'Q{_fmt(cx)},{_fmt(cy)} {_fmt(p2[0])},{_fmt(p2[1])}" '
                     f'marker-end="url(#{marker}head)"/>')
    w, h = X1 - X0, Y1 - Y0
    defs = ('<defs>'
            '<marker id="rxhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" class="rxheadp"/></marker>'
            '<marker id="fullhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" class="curlyhead"/></marker>'
            '<marker id="fishhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,5 z" class="curlyhead"/></marker></defs>')
    return (f'<svg class="rxn" xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(X0)} {_fmt(Y0)} '
            f'{_fmt(w)} {_fmt(h)}" width="{_fmt(w)}" height="{_fmt(h)}" role="img">{defs}'
            + "".join(parts) + "</svg>")


def layout_px(pos, scale=BOND_PX):
    return {i: (x * scale, -y * scale) for i, (x, y) in pos.items()}


def _anchor(t, px):
    if t[0] == "atom":
        return px.get(t[1])
    a, b = px.get(t[1]), px.get(t[2])
    if a is None or b is None:
        return a or b
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


SVG_CSS = """
svg.lewis, svg.rxn { overflow: visible; max-width: 100%; height: auto; }
svg .bond { stroke: var(--ink); stroke-width: 1.6; stroke-linecap: round; }
svg .bond.hl { stroke: var(--accent); stroke-width: 2.4; }
svg .atom { font: 600 14px var(--mono); fill: var(--ink); }
svg .atom.el-H { font-weight: 500; fill: var(--muted); }
svg .atom.el-O { fill: var(--o); } svg .atom.el-N { fill: var(--n); }
svg .lp { fill: var(--ink); } svg .rad { fill: var(--accent); }
svg .chg { fill: none; stroke: var(--ink); stroke-width: 1; }
svg .chgt { font: 600 9px var(--mono); fill: var(--ink); }
svg .mapno { font: 9px var(--mono); fill: var(--accent); }
svg .rxarrow { stroke: var(--ink); stroke-width: 1.6; }
svg .rxheadp { fill: var(--ink); }
svg .curly { fill: none; stroke: var(--accent); stroke-width: 1.5; }
svg .curlyhead { fill: var(--accent); }
"""
