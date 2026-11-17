"""Self-contained HTML report: species, Lewis structures, evidence, mechanisms."""

from __future__ import annotations

import html
import json
import time
from collections import Counter

from .draw import SVG_CSS, lewis_svg, reaction_svg
from .lewis import lewis_text
from .mcd import lewis_set

MAX_DRAWN_REACTIONS = 300

CSS = """
:root{--bg:#fbfaf7;--panel:#ffffff;--ink:#1f2328;--muted:#6b6f76;--line:#e4e2dc;
--accent:#c2410c;--ok:#15803d;--warn:#b45309;--bad:#b91c1c;--o:#b91c1c;--n:#1d4ed8;
--chip:#f1efe9;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--panel:#1d2024;
--ink:#e8e6e1;--muted:#9aa0a6;--line:#30343a;--accent:#fb923c;--ok:#4ade80;--warn:#fbbf24;
--bad:#f87171;--o:#f87171;--n:#93c5fd;--chip:#262a30}}
:root[data-theme="dark"]{--bg:#15171a;--panel:#1d2024;--ink:#e8e6e1;--muted:#9aa0a6;--line:#30343a;
--accent:#fb923c;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--o:#f87171;--n:#93c5fd;--chip:#262a30}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans)}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
.sub{color:var(--muted)}.mono{font-family:var(--mono)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:18px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:12px}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
input[type=search],select{background:var(--panel);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:7px 10px;font:inherit;min-width:220px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;background:var(--chip)}
tr.row{cursor:pointer}tr.row:hover td{background:var(--chip)}
.badge{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.known{color:var(--n)}.identified{color:var(--ok)}.unresolved{color:var(--warn)}.no-formula{color:var(--bad)}
.bar{height:6px;background:var(--chip);border-radius:3px;width:90px;display:inline-block;vertical-align:middle}
.bar i{display:block;height:6px;border-radius:3px;background:var(--ok)}
.detail td{background:var(--bg)}.detail{display:none}.detail.open{display:table-row}
.grid{display:grid;grid-template-columns:minmax(220px,1fr) 2fr;gap:18px}
@media (max-width:760px){.grid{grid-template-columns:1fr}table{font-size:12px}th,td{padding:6px}}
.cands td,.cands th{font-size:12px;padding:4px 8px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font:12px/1.45 var(--mono)}
.rxn{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:10px 14px}
.rxn summary{cursor:pointer;font-family:var(--mono)}
.rxn .cls{font-family:var(--sans);color:var(--accent);margin-left:8px;font-size:12px}
.scroll{overflow-x:auto}.legend{color:var(--muted);font-size:12px}
.finding{border-left:3px solid var(--bad);background:var(--panel);padding:10px 14px;margin:8px 0;border-radius:6px}
""" + SVG_CSS

JS = """
const q=document.getElementById('q'),fs=document.getElementById('fs');
function filt(){const t=(q.value||'').toLowerCase(),s=fs.value;
document.querySelectorAll('tr.row').forEach(r=>{const ok=(!t||r.dataset.k.includes(t))&&(!s||r.dataset.s===s);
r.style.display=ok?'':'none';const d=r.nextElementSibling;if(!ok)d.classList.remove('open');});}
q.addEventListener('input',filt);fs.addEventListener('change',filt);
document.querySelectorAll('tr.row').forEach(r=>r.addEventListener('click',()=>r.nextElementSibling.classList.toggle('open')));
const rq=document.getElementById('rq'),rc=document.getElementById('rc');
function rfilt(){const t=(rq.value||'').toLowerCase(),c=rc.value;
document.querySelectorAll('details.rxn').forEach(d=>{const ok=(!t||d.dataset.k.includes(t))&&(!c||d.dataset.c===c);d.style.display=ok?'':'none';});}
rq.addEventListener('input',rfilt);rc.addEventListener('change',rfilt);
"""


def _e(x) -> str:
    return html.escape(str(x))


def species_detail(ident, st) -> str:
    c = st.chosen or (st.ranked()[0] if st.candidates else None)
    out = ['<div class="grid"><div>']
    if c is not None:
        structs = lewis_set(c.mol)[:3]
        for s in structs:
            out.append(lewis_svg(c.mol, s, title=f"{st.name} Lewis structure"))
        if len(lewis_set(c.mol)) > 1:
            out.append(f'<div class="legend">{len(lewis_set(c.mol))} resonance structures</div>')
        if structs:
            out.append("<pre>" + _e(lewis_text(c.mol, structs[0])) + "</pre>")
    out.append("</div><div>")
    if st.notes:
        out.append("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in st.notes) + "</ul>")
    ranked = st.ranked()[:8]
    if ranked:
        out.append('<table class="cands"><tr><th>candidate</th><th>P</th><th>sources</th>'
                   '<th>prior</th><th>thermo</th><th>reactions</th><th>notes</th></tr>')
        for cand in ranked:
            notes = "; ".join(cand.reasons[:3])
            if cand.thermo_note:
                notes = cand.thermo_note + ("; " + notes if notes else "")
            out.append(f"<tr><td class='mono'>{_e(cand.smiles)}</td><td>{cand.posterior:.2f}</td>"
                       f"<td>{_e(', '.join(sorted(cand.sources)))}</td><td>{cand.log_prior + cand.log_name:+.1f}</td>"
                       f"<td>{cand.log_thermo:+.1f}</td><td>{cand.log_like:+.1f}</td><td>{_e(notes)}</td></tr>")
        out.append("</table>")
    if c is not None and c.distances:
        rx = {r.index: r for r in ident.mech.reactions}
        items = sorted(c.distances.items(), key=lambda kv: (kv[1] is None, kv[1] or 0))[:14]
        out.append("<div class='legend'>Witness reactions (sigma bonds broken + formed for the chosen structure):</div><ul>")
        for i, d in items:
            out.append(f"<li class='mono'>{_e(rx[i].equation)} &nbsp;→&nbsp; {d if d is not None else '&gt;4'}</li>")
        out.append("</ul>")
    out.append("</div></div>")
    return "".join(out)


def write_report(ident, path: str, title: str = "Species identification", inputs: str = "") -> str:
    species = list(ident.species.values())
    counts = Counter(s.status for s in species)
    sols = ident.solutions
    classes = Counter(getattr(s, "reaction_class", "?") for s in sols.values())
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>"]
    parts.append(f"<h1>{_e(title)}</h1><div class='sub'>rxnsolve · {time.strftime('%Y-%m-%d %H:%M')}"
                 f"{' · ' + _e(inputs) if inputs else ''}</div>")
    parts.append("<div class='stats'>")
    for label, val in [("species", len(species)), ("given", counts.get("known", 0)),
                       ("identified", counts.get("identified", 0)),
                       ("still open", counts.get("unresolved", 0) + counts.get("no-formula", 0)),
                       ("reactions", len(ident.mech.reactions)), ("reactions solved", len(sols))]:
        parts.append(f"<div class='stat'><b>{val}</b><span>{label}</span></div>")
    parts.append("</div>")
    parts.append("<p class='sub'>Each structure is scored by its formula (hard constraint), chemical "
                 "plausibility of its Lewis structure, the species name, thermochemistry (library + group "
                 "additivity vs. the NASA polynomials) and every reaction it takes part in: the fewer sigma "
                 "bonds a reaction must break and form with that structure, the more the reaction supports it.</p>")

    findings = getattr(ident, "audit_findings", None)
    if findings:
        parts.append(f"<h2>Audit: {len(findings)} given structures look wrong</h2>")
        for f in findings:
            rx = "".join(f"<li class='mono'>{_e(eq)}: given {g if g is not None else '&gt;4'} vs "
                         f"suggested {b if b is not None else '&gt;4'} edits</li>" for eq, g, b in f["reactions"])
            parts.append(f"<div class='finding'><b>{_e(f['species'])}</b> ({_e(f['formula'])}) given "
                         f"<span class='mono'>{_e(f['given'])}</span>, evidence prefers "
                         f"<span class='mono'>{_e(f['suggested'])}</span> (log-odds {f['log_gap']:+.1f}).<br>"
                         f"<span class='legend'>{_e(f['given_thermo'])} | {_e(f['suggested_thermo'])}</span>"
                         f"<ul>{rx}</ul></div>")

    parts.append("<h2>Species</h2><div class='toolbar'><input id='q' type='search' placeholder='Filter species, formula, SMILES'>"
                 "<select id='fs'><option value=''>all statuses</option><option>known</option>"
                 "<option>identified</option><option>unresolved</option><option>no-formula</option></select></div>")
    parts.append("<div class='scroll'><table><tr><th>species</th><th>formula</th><th>status</th>"
                 "<th>structure (SMILES)</th><th>confidence</th><th>how</th></tr>")
    order = {"unresolved": 0, "no-formula": 1, "identified": 2, "known": 3}
    for st in sorted(species, key=lambda s: (order.get(s.status, 9), -s.confidence)):
        c = st.chosen
        smi = c.smiles if c else (st.ranked()[0].smiles + " ?" if st.candidates else "")
        conf = st.confidence if st.resolved else (st.ranked()[0].posterior if st.candidates else 0)
        key = f"{st.name} {st.formula_str} {smi}".lower()
        parts.append(f"<tr class='row' data-k='{_e(key)}' data-s='{st.status}'><td class='mono'>{_e(st.name)}</td>"
                     f"<td class='mono'>{_e(st.formula_str)}</td><td><span class='badge {st.status}'>{st.status}</span></td>"
                     f"<td class='mono'>{_e(smi)}</td><td><span class='bar'><i style='width:{conf * 100:.0f}%'></i></span> "
                     f"{conf:.2f}</td><td>{_e(st.method)}</td></tr>")
        parts.append(f"<tr class='detail'><td colspan='6'>{species_detail(ident, st)}</td></tr>")
    parts.append("</table></div>")

    if sols:
        parts.append("<h2>Solved reactions</h2><p class='legend'>Atom mapping with the fewest sigma-bond edits; "
                     "highlighted bonds change; numbers follow atoms across the arrow; curved arrows are "
                     "electron moves (half-head = one electron, full head = a pair).</p>")
        parts.append("<div class='toolbar'><input id='rq' type='search' placeholder='Filter reactions'>"
                     "<select id='rc'><option value=''>all reaction classes</option>"
                     + "".join(f"<option>{_e(k)}</option>" for k, _ in classes.most_common()) + "</select></div>")
        drawn = 0
        rx = {r.index: r for r in ident.mech.reactions}
        for idx, sol in sorted(sols.items()):
            r = rx[idx]
            lines = "\n".join(sol.summary_lines())
            svg = ""
            if drawn < MAX_DRAWN_REACTIONS and sol.R is not None and sol.R.n <= 34 and sol.distance is not None:
                try:
                    svg = reaction_svg(sol)
                    drawn += 1
                except Exception:  # noqa: BLE001 - a drawing glitch must not break the report
                    svg = ""
            parts.append(f"<details class='rxn' data-k='{_e(r.equation.lower())}' data-c='{_e(sol.reaction_class)}'>"
                         f"<summary>{idx}. {_e(r.equation)}<span class='cls'>{_e(sol.reaction_class)}"
                         f"{'' if sol.distance is None else f' · {sol.distance} edits'}</span></summary>"
                         f"<div class='scroll'>{svg}</div><pre>{_e(lines)}</pre></details>")
    parts.append(f"<script>{JS}</script></main></body></html>")
    text = "".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def to_json(ident) -> dict:
    out = {"species": [], "log": ident.log}
    for st in ident.species.values():
        out["species"].append({
            "name": st.name, "formula": st.formula_str, "status": st.status,
            "smiles": st.chosen.smiles if st.chosen else None,
            "confidence": round(st.confidence, 4), "method": st.method, "notes": st.notes,
            "candidates": [{"smiles": c.smiles, "posterior": round(c.posterior, 4),
                            "sources": sorted(c.sources), "thermo": c.thermo_note}
                           for c in st.ranked()[:5]],
        })
    out["reactions"] = [{"index": i, "class": s.reaction_class, "sigma_edits": s.distance,
                         "summary": s.summary_lines()} for i, s in sorted(ident.solutions.items())]
    if getattr(ident, "audit_findings", None):
        out["audit"] = ident.audit_findings
    return out


def write_json(ident, path: str) -> str:
    with open(path, "w") as fh:
        json.dump(to_json(ident), fh, indent=1)
    return path
