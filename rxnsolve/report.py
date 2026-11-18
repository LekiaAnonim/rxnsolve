"""Static report site: how every species was identified.

    out/index.html             overview and species table
    out/species/<id>.html      one page per species - the evidence chain
    out/reactions.html         every solved reaction (atom map, arrows, spin)
    out/style.css              shared look (white, quiet, one accent)
    out/report.html            redirect to index.html (older links)

Each species page reads top to bottom as the argument for its structure:
formula -> candidates -> Lewis structure -> name -> thermochemistry ->
deciding reactions (solved, with arrows) -> decision.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from collections import Counter

from .draw import SVG_CSS, lewis_svg, reaction_svg
from .mcd import SPIN_NAME, lewis_set, spin_text

MAX_REACTION_DIAGRAMS = 300     # on reactions.html
DECISIVE_DIAGRAMS = 2           # solved diagrams per species page
D_NONE = 5                      # display value for "more than 4 edits"
NAME_REASONS = ("known name", "condensed formula", "LLNL naming pattern")

CSS = """
:root{--bg:#ffffff;--ink:#111827;--muted:#6b7280;--faint:#9ca3af;--line:#e5e7eb;--soft:#f8fafc;
--accent:#1d4ed8;--change:#ea580c;--ok:#15803d;--okbg:#ecfdf3;--warn:#b45309;--warnbg:#fffbeb;
--bad:#b91c1c;--badbg:#fef2f2;--given:#1d4ed8;--givenbg:#eff6ff;--o:#c0262d;--n:#1d4ed8;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header.top{border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:5}
header.top .in{max-width:1120px;margin:0 auto;padding:12px 20px;display:flex;gap:24px;align-items:center;flex-wrap:wrap}
header.top .brand{font-weight:650;letter-spacing:-.01em}
header.top nav{display:flex;gap:18px;font-size:14px;margin-left:auto}
header.top nav a{color:var(--muted)}header.top nav a.on{color:var(--ink);font-weight:600}
main{max-width:1120px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px;letter-spacing:-.015em}
h2{font-size:18px;margin:40px 0 12px;letter-spacing:-.01em}
h3{font-size:15px;margin:0 0 6px}
.sub{color:var(--muted);font-size:14px}.mono{font-family:var(--mono);font-size:13.5px}
.muted{color:var(--muted)}.small{font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0 8px}
.stat{border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat b{display:block;font-size:24px;font-weight:650;letter-spacing:-.02em}
.stat span{color:var(--muted);font-size:13px}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 12px}
input[type=search],input[type=text],select{border:1px solid var(--line);border-radius:8px;padding:8px 11px;font:inherit;font-size:14px;background:#fff;color:var(--ink);min-width:220px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
tbody tr:hover td{background:var(--soft)}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap}
.pill.identified{color:var(--ok);background:var(--okbg)}.pill.known{color:var(--given);background:var(--givenbg)}
.pill.unresolved{color:var(--warn);background:var(--warnbg)}.pill.no-formula,.pill.flag{color:var(--bad);background:var(--badbg)}
.pill.allowed{color:var(--ok);background:var(--okbg)}.pill.forbidden{color:var(--bad);background:var(--badbg)}
.bar{display:inline-block;width:84px;height:6px;border-radius:3px;background:var(--line);vertical-align:middle;margin-right:8px}
.bar i{display:block;height:6px;border-radius:3px;background:var(--ok)}
.grid2{display:grid;grid-template-columns:minmax(260px,5fr) 7fr;gap:32px;align-items:start}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 14px}
.structure{border:1px solid var(--line);border-radius:10px;padding:16px;text-align:center;margin-bottom:14px}
.structure svg{margin:4px auto;display:block}
.steps{list-style:none;padding:0;margin:0;counter-reset:s}
.steps>li{position:relative;padding:0 0 20px 34px;border-left:1px solid var(--line);margin-left:12px}
.steps>li:last-child{border-left-color:transparent}
.steps>li::before{counter-increment:s;content:counter(s);position:absolute;left:-12px;top:0;width:24px;height:24px;border-radius:50%;
background:#fff;border:1px solid var(--ink);font-size:12px;font-weight:600;text-align:center;line-height:22px}
.steps h3{margin:0 0 4px;font-size:15px;line-height:24px}.steps p{margin:2px 0}
.steps table{margin-top:8px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;font-size:14px;margin:0}.kv dt{color:var(--muted)}.kv dd{margin:0}
.probs{margin-top:8px}
.probs div{display:grid;grid-template-columns:minmax(120px,240px) 1fr 44px;gap:10px;align-items:center;margin:5px 0;font-size:13px}
.probs .track{height:8px;background:var(--line);border-radius:4px}.probs .track i{display:block;height:8px;border-radius:4px;background:var(--faint)}
.probs .win .track i{background:var(--ok)}.probs .win span:first-child{font-weight:600}
.rx{border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:14px 16px}
.rx h3{font-weight:600;font-size:14.5px}
.rx .meta{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 8px;font-size:13px;color:var(--muted);align-items:center}
.rx .facts{font-size:13.5px;margin:8px 0 0;padding-left:18px}.rx .facts li{margin:2px 0}
.scroll{overflow-x:auto}
.legend{color:var(--muted);font-size:13px}
.flagbox{border-left:3px solid var(--bad);background:var(--badbg);border-radius:6px;padding:12px 14px;margin:10px 0;font-size:14px}
.crumb{font-size:13px;color:var(--muted);margin-bottom:10px}
footer{color:var(--faint);font-size:12px;margin-top:60px;border-top:1px solid var(--line);padding-top:12px}
""" + SVG_CSS

INDEX_JS = """
const q=document.getElementById('q'),fs=document.getElementById('fs');
function f(){const t=(q.value||'').toLowerCase(),s=fs.value;
document.querySelectorAll('#sp tbody tr').forEach(r=>{r.style.display=((!t||r.dataset.k.includes(t))&&(!s||r.dataset.s===s))?'':'none'})}
q.addEventListener('input',f);fs.addEventListener('change',f);
"""

RX_JS = """
const q=document.getElementById('q'),c=document.getElementById('c'),sp=document.getElementById('sp');
function f(){const t=(q.value||'').toLowerCase(),k=c.value,s=sp.value;
document.querySelectorAll('.rx').forEach(d=>{d.style.display=((!t||d.dataset.k.includes(t))&&(!k||d.dataset.c===k)&&(!s||d.dataset.s===s))?'':'none'})}
[q,c,sp].forEach(e=>e.addEventListener(e.tagName==='INPUT'?'input':'change',f));
const pre=new URLSearchParams(location.search).get('species');if(pre){q.value=pre.toLowerCase();f();}
"""

def _e(x) -> str:
    return html.escape(str(x))


def _slugs(species) -> dict[str, str]:
    out = {}
    for k, st in enumerate(species):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", st.name).strip("_") or "species"
        out[st.name] = f"{k:03d}_{safe}"
    return out


def _page(title: str, body: str, nav: str, depth: int, model: str, js: str = "") -> str:
    up = "../" * depth
    links = [("overview", f"{up}index.html"), ("reactions", f"{up}reactions.html")]
    navh = "".join(f"<a href='{h}' class='{'on' if k == nav else ''}'>{k.title()}</a>" for k, h in links)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><link rel='stylesheet' href='{up}style.css'></head><body>"
            f"<header class='top'><div class='in'><span class='brand'>rxnsolve</span>"
            f"<span class='sub'>{_e(model)}</span><nav>{navh}</nav></div></header>"
            f"<main>{body}<footer>Generated by rxnsolve · {time.strftime('%Y-%m-%d %H:%M')}</footer></main>"
            + (f"<script>{js}</script>" if js else "") + "</body></html>")


def _status_label(st) -> str:
    return {"known": "given", "unresolved": "left open", "no-formula": "no thermo data"}.get(st.status, st.status)


def _species_link(name, slugs, depth):
    if name in slugs:
        return f"<a class='mono' href='{'../' * depth}species/{slugs[name]}.html'>{_e(name)}</a>"
    return f"<span class='mono'>{_e(name)}</span>"


def _equation_html(r, slugs, depth):
    def side(names):
        counts = Counter(names)
        return " + ".join((f"{counts[n]} " if counts[n] > 1 else "") + _species_link(n, slugs, depth)
                          for n in dict.fromkeys(names))
    arrow = " ⇌ " if r.reversible else " → "
    third = f" <span class='muted small'>(+{_e(r.third_body)})</span>" if r.third_body else ""
    return side(r.reactants) + arrow + side(r.products) + third


def _runner_up(st):
    c = st.chosen
    for x in st.ranked():
        if c is None or x.key != c.key:
            return x
    return None


def decisive_reactions(st):
    """Reactions with distances for the chosen structure, those that favour it most first."""
    c = st.chosen
    r2 = _runner_up(st)
    rows = []
    if c is None:
        return rows, r2
    for ri, d in c.distances.items():
        d2 = r2.distances.get(ri, "n/a") if r2 is not None else "n/a"
        a = D_NONE if d is None else d
        b = None if d2 == "n/a" else (D_NONE if d2 is None else d2)
        rows.append((None if b is None else b - a, ri, d, d2))
    rows.sort(key=lambda t: (-(t[0] if t[0] is not None else -99), t[2] if t[2] is not None else 9, t[1]))
    return rows, r2


def _edits(d):
    if d == "n/a":
        return "<span class='muted'>–</span>"
    return "&gt;4" if d is None else str(d)


def _reaction_card(sol, r, slugs, depth, draw=True):
    spin = ""
    if sol.spin_allowed is True:
        spin = "<span class='pill allowed'>spin-allowed</span>"
    elif sol.spin_allowed is False:
        spin = "<span class='pill forbidden'>spin-forbidden</span>"
    edits = "" if sol.distance is None else f"{sol.distance} bond edit{'s' if sol.distance != 1 else ''}"
    svg = ""
    if draw and sol.distance is not None and sol.R is not None and sol.R.n <= 36:
        try:
            svg = f"<div class='scroll'>{reaction_svg(sol)}</div>"
        except Exception:  # noqa: BLE001 - a drawing glitch must not break the report
            svg = ""
    facts = []
    if sol.distance is not None:
        lab = sol.label
        for i, j in sol.broken:
            facts.append(f"bond broken: <span class='mono'>{_e(lab(i))} – {_e(lab(j))}</span>")
        for i, j in sol.formed:
            facts.append(f"bond formed: <span class='mono'>{_e(lab(i))} – {_e(lab(j))}</span>")
        for i, j, o1, o2 in sol.order_changes:
            facts.append(f"bond order: <span class='mono'>{_e(lab(i))} – {_e(lab(j))}</span> {o1} → {o2}")
        for mv in sol.moved:
            facts.append(f"atoms moved: {_e(mv['atoms'])} from {_e(mv['from'])} to {_e(mv['to'])}")
        for i, ch in sorted(sol.atom_changes.items()):
            parts = [f"{k.replace('_', ' ')} {v:+d}" for k, v in ch.items() if v]
            if parts:
                facts.append(f"electrons on <span class='mono'>{_e(lab(i))}</span>: {', '.join(parts)}")
    elif sol.balanced:
        facts.append("no atom mapping within 6 bond edits (lumped or non-elementary step)")
    else:
        facts.append("reaction is not atom-balanced")
    if sol.spin_allowed is not None:
        facts.append("spin: " + _e(spin_text(sol.spin_r, sol.spin_p, sol.spin_allowed)))
    return (f"<h3>{r.index}. {_equation_html(r, slugs, depth)}</h3>"
            f"<div class='meta'><span>{_e(sol.reaction_class)}</span>"
            f"{'<span>· ' + edits + '</span>' if edits else ''}{spin}</div>{svg}"
            "<ul class='facts'>" + "".join(f"<li>{x}</li>" for x in facts) + "</ul>")


# ====================================================================
# Species page
# ====================================================================

def species_page(ident, st, slugs) -> str:
    rx = {r.index: r for r in ident.mech.reactions}
    c = st.chosen if st.chosen is not None else (st.ranked()[0] if st.candidates else None)
    flagged = [f for f in (getattr(ident, "audit_findings", None) or []) if f["species"] == st.name]
    shows_p = st.resolved and (st.status != "known" or st.audited)
    head = (f"<div class='crumb'><a href='../index.html'>Overview</a> / species</div>"
            f"<h1 class='mono' style='font-size:24px'>{_e(st.name)}</h1>"
            f"<div class='sub'>{_e(st.formula_str)} · <span class='pill {('flag' if flagged else st.status)}'>"
            f"{'given, flagged by audit' if flagged else _status_label(st)}</span>"
            + (f" · structure <span class='mono'>{_e(c.smiles)}</span>" if c is not None else "")
            + (f" · probability {st.confidence:.2f}" if shows_p else "") + "</div>")

    # ---- left column: structure and electrons
    left = []
    if c is not None:
        structs = lewis_set(c.mol)
        left.append("<div class='structure'>")
        for s in structs[:2]:
            left.append(lewis_svg(c.mol, s, title=f"Lewis structure of {st.name}"))
        what = "Lewis structure" if len(structs) <= 1 else f"{len(structs)} resonance structures, 2 shown"
        left.append(f"<div class='legend'>{what}. Black dot pairs are lone pairs; orange dots are "
                    "unpaired electrons.</div></div>")
        if structs:
            cen = structs[0].electron_census(c.mol)
            mult = c.mol.multiplicity or (structs[0].unpaired + 1)
            left.append("<div class='card'><h3>Electrons</h3><dl class='kv'>"
                        f"<dt>valence electrons</dt><dd>{cen['valence_electrons']}</dd>"
                        f"<dt>bonding pairs</dt><dd>{cen['bonding_pairs']} ({cen['sigma_bonds']} σ, {cen['pi_bonds']} π)</dd>"
                        f"<dt>lone pairs</dt><dd>{cen['lone_pairs']}</dd>"
                        f"<dt>unpaired electrons</dt><dd>{cen['unpaired_electrons']}</dd>"
                        f"<dt>spin state</dt><dd>{SPIN_NAME.get(mult, mult)} (2S+1 = {mult})</dd>"
                        f"<dt>formal charges</dt><dd>{cen['formal_charges'] or 'none'}</dd></dl></div>")
    if st.notes:
        left.append("<div class='card'><h3>Notes</h3><ul class='small' style='margin:0;padding-left:18px'>"
                    + "".join(f"<li>{_e(n)}</li>" for n in st.notes) + "</ul></div>")

    # ---- right column: the evidence chain
    steps = []
    iso_note = next((n for n in st.notes if "isomer" in n), "")
    steps.append(("Formula", f"<p><span class='mono'>{_e(st.formula_str)}</span> from the thermo card."
                             f"{(' ' + _e(iso_note) + '.') if iso_note else ''}</p>"))
    if st.status == "known" and not st.audited:
        given = ident.known.get(st.name, c.smiles if c else "")
        steps.append(("Given structure",
                      f"<p>Taken from the SMILES file as <span class='mono'>{_e(given)}</span> and not "
                      "re-scored. Run an audit to check it against the reactions and thermochemistry.</p>"))
    else:
        srcs = Counter(src for cand in st.candidates.values() for src in cand.sources)
        parts = ", ".join(f"{v} from {k}" for k, v in srcs.most_common() if k != "user")
        steps.append(("Candidates", f"<p>{len(st.candidates)} structures kept for comparison"
                                    f"{(' (' + _e(parts) + ')') if parts else ''}.</p>"))
        if c is not None:
            prior_why = [x for x in c.reasons if not x.startswith("completes reaction") and x not in NAME_REASONS]
            steps.append(("Lewis structure",
                          f"<p>Plausibility {c.log_prior:+.1f} relative to the best candidate"
                          + (f": {_e('; '.join(prior_why))}" if prior_why else " (an ordinary structure)")
                          + ".</p>"))
            name_why = [x for x in c.reasons if x in NAME_REASONS]
            if not ident.use_names:
                name_txt = "<p>Not used (blind mode).</p>"
            elif c.log_name:
                name_txt = (f"<p>The name points to this structure ({_e(', '.join(name_why) or 'name hint')}, "
                            f"{c.log_name:+.1f}).</p>")
            else:
                name_txt = "<p>The name gives no hint for this structure.</p>"
            steps.append(("Name", name_txt))
            t = ident.mech.thermo.get(st.name)
            if t is not None and t.h298 is not None:
                r2 = _runner_up(st)
                if ident.thermo is None:
                    tt = f"<p>Not used: {_e(getattr(ident, 'thermo_info', ''))}.</p>"
                else:
                    tt = (f"<p>Mechanism H298 = {t.h298:.1f} kcal/mol, S298 = {t.s298:.1f} cal/mol/K.</p>"
                          f"<p>{_e(c.thermo_note or 'no thermo estimate')} ({c.log_thermo:+.1f}).</p>")
                if r2 is not None and r2.thermo_note:
                    tt += (f"<p class='muted small'>Runner-up <span class='mono'>{_e(r2.smiles)}</span>: "
                           f"{_e(r2.thermo_note)} ({r2.log_thermo:+.1f}).</p>")
                steps.append(("Thermochemistry", tt))
            else:
                steps.append(("Thermochemistry", "<p>No thermo data for this species.</p>"))
            rows, r2 = decisive_reactions(st)
            if rows:
                sup = sum(1 for g, *_ in rows if g is not None and g > 0)
                against = sum(1 for g, *_ in rows if g is not None and g < 0)
                tbl = ["<table class='small'><thead><tr><th>reaction</th><th>this structure</th>"
                       f"<th>runner-up{(' <span class=mono>' + _e(r2.smiles) + '</span>') if r2 else ''}</th>"
                       "</tr></thead><tbody>"]
                for gap, ri, d, d2 in rows[:8]:
                    tbl.append(f"<tr><td>{_equation_html(rx[ri], slugs, 1)}</td><td>{_edits(d)}</td>"
                               f"<td>{_edits(d2)}</td></tr>")
                tbl.append("</tbody></table><p class='legend'>Bond edits = bonds broken + bonds formed "
                           "to run the reaction with that structure.</p>")
                steps.append(("Reactions",
                              f"<p>{len(rows)} reaction{'s' if len(rows) != 1 else ''} had every other species known. "
                              f"This structure needs "
                              f"fewer bond edits than the runner-up in {sup} of them and more in {against}.</p>"
                              + "".join(tbl)))
            else:
                steps.append(("Reactions", "<p>No reaction had all its other species identified when this "
                                           "species was decided.</p>"))
            steps.append(("Entropy and heat capacity", _scp_html(ident, st, c)))
            steps.append(("Kinetics", _kinetics_html(st, c, rx, slugs)))
        ranked = st.ranked()[:5]
        if ranked:
            bars = ["<div class='probs'>"]
            for cand in ranked:
                win = c is not None and cand.key == c.key
                bars.append(f"<div class='{'win' if win else ''}'><span class='mono'>{_e(cand.smiles)}</span>"
                            f"<span class='track'><i style='width:{max(cand.posterior * 100, 0.5):.1f}%'></i></span>"
                            f"<span>{cand.posterior:.2f}</span></div>")
            bars.append("</div>")
            if st.status == "identified":
                verdict = (f"<p>Identified in round {st.round} with probability {st.confidence:.2f} "
                           f"({_e(st.method)}).</p>")
            elif st.status == "known":
                verdict = (f"<p>Given structure; audit probability {st.confidence:.2f}"
                           + (". <b>An alternative is more than 20 times more likely.</b>" if flagged else ".")
                           + "</p>")
            else:
                verdict = "<p>Left open: no structure cleared the threshold with a clear margin.</p>"
            steps.append(("Decision", verdict + "".join(bars)))
    chain = ["<ol class='steps'>"] + [f"<li><h3>{_e(t)}</h3>{b}</li>" for t, b in steps] + ["</ol>"]
    flag_html = "".join(
        f"<div class='flagbox'>Audit: given <span class='mono'>{_e(f['given'])}</span>, the evidence prefers "
        f"<span class='mono'>{_e(f['suggested'])}</span> (log-odds {f['log_gap']:+.1f}).</div>" for f in flagged)

    # ---- deciding reactions, solved
    solved = []
    rows = decisive_reactions(st)[0] if c is not None else []
    picks = [ri for gap, ri, d, d2 in rows if d is not None and ri in ident.solutions][:DECISIVE_DIAGRAMS]
    if not picks:
        picks = [r.index for r in ident.mech.reactions
                 if st.name in r.species() and r.index in ident.solutions
                 and ident.solutions[r.index].distance is not None][:DECISIVE_DIAGRAMS]
    if picks:
        solved.append("<h2>Deciding reactions, solved</h2><p class='legend'>The atom mapping that needs the "
                      "fewest bond edits. Orange bonds change; the numbers follow atoms across the arrow; curved "
                      "arrows move electrons (half head = one electron, full head = a pair).</p>")
        for ri in picks:
            solved.append(f"<div class='rx'>{_reaction_card(ident.solutions[ri], rx[ri], slugs, 1)}</div>")
    n_all = sum(1 for r in ident.mech.reactions if st.name in r.species())
    more = (f"<p class='small'><a href='../reactions.html?species={_e(st.name)}'>"
            + (f"All {n_all} reactions" if n_all != 1 else "The one reaction")
            + f" containing {_e(st.name)}</a></p>") if n_all else ""
    return (head + flag_html + "<div class='grid2' style='margin-top:22px'><div>" + "".join(left) + "</div><div>"
            "<h2 style='margin-top:0'>Why this structure</h2>" + "".join(chain) + "</div></div>"
            + "".join(solved) + more)


def _scp_html(ident, st, c) -> str:
    """Predicted S298 and Cp(T) of this structure against the species' own thermo card."""
    if getattr(ident.thermo, "scp", None) is None:
        return "<p>Not used: no entropy and heat-capacity model available.</p>"
    if not c.scp_note:
        return "<p>No thermo card for this species.</p>"
    r2 = _runner_up(st)
    out = [f"<p>A ring locks bonds that rotate freely in a chain, so rings have lower entropy and a "
           f"different heat-capacity curve. Predicted from this structure: {_e(c.scp_note)} "
           f"({c.log_scp:+.1f}).</p>"]
    if r2 is not None and r2.scp_note:
        out.append(f"<p class='muted small'>Runner-up <span class='mono'>{_e(r2.smiles)}</span>: "
                   f"{_e(r2.scp_note)} ({r2.log_scp:+.1f}).</p>")
    return "".join(out)


def _kinetics_html(st, c, rx, slugs) -> str:
    """Competing abstraction channels: do A and Ea agree with this structure?"""
    rivals = [x for x in st.ranked()[:8] if x.key != c.key and x.log_kin < 0]
    if not c.kin_checks and not rivals:
        return ("<p>No competing H-abstraction channel with an identified sibling product, so the rate "
                "parameters say nothing about this species.</p>")
    out = ["<p>Competing abstraction channels from the same parent: a larger A should go with more "
           "equivalent hydrogens, and a lower Ea with the weaker C–H bond (the more stable radical).</p>"]
    if c.kin_checks:
        out.append("<table class='small'><thead><tr><th>channel vs sibling</th><th>signal</th><th>agrees</th>"
                   "<th>detail</th></tr></thead><tbody>")
        for i1, i2, sib, signal, ok, detail in c.kin_checks[:8]:
            out.append(f"<tr><td>{_equation_html(rx[i1], slugs, 1)}<br><span class='muted'>vs</span> "
                       f"{_equation_html(rx[i2], slugs, 1)}</td><td>{_e(signal)}</td>"
                       f"<td>{'yes' if ok else '<b>no</b>'}</td><td class='mono'>{_e(detail)}</td></tr>")
        out.append("</tbody></table>")
    if rivals:
        out.append("<p>" + "; ".join(
            f"<span class='mono'>{_e(x.smiles)}</span> contradicts "
            f"{sum(1 for k in x.kin_checks if not k[4])} signal(s) ({x.log_kin:+.1f})" for x in rivals[:4]) + ".</p>")
    out.append(f"<p class='legend'>This structure: {c.log_kin:+.1f}. Each contradicted signal costs "
               "ln(0.1/0.9) = −2.2, at most three per species; the signals held in 90% of 5,001 tested pairs.</p>")
    return "".join(out)


# ====================================================================
# Site
# ====================================================================

def write_site(ident, out_dir: str, title: str = "Species identification", inputs: str = "",
               seconds: float | None = None) -> str:
    os.makedirs(os.path.join(out_dir, "species"), exist_ok=True)
    species = list(ident.species.values())
    slugs = _slugs(species)
    model = title.split(": ", 1)[-1]
    with open(os.path.join(out_dir, "style.css"), "w", encoding="utf-8") as fh:
        fh.write(CSS)
    counts = Counter(s.status for s in species)
    sols = ident.solutions
    findings = getattr(ident, "audit_findings", None) or []
    forbidden = sum(1 for s in sols.values() if s.spin_allowed is False)

    # ---- overview
    b = [f"<h1>{_e(title)}</h1><div class='sub'>{_e(inputs)}"
         + (f" · {seconds:.0f} s" if seconds else "") + "</div>"
         + f"<div class='sub small' style='margin-top:4px'>Thermochemistry: {_e(getattr(ident, 'thermo_info', ''))}</div>"]
    stats = [("species", len(species)), ("given", counts.get("known", 0)),
             ("identified", counts.get("identified", 0)),
             ("left open", counts.get("unresolved", 0) + counts.get("no-formula", 0)),
             ("reactions solved", f"{len(sols)} / {len(ident.mech.reactions)}"),
             ("spin-forbidden steps", forbidden)]
    if any(s.audited for s in species):
        stats.insert(2, ("given, flagged by audit", len(findings)))
    b.append("<div class='stats'>" + "".join(f"<div class='stat'><b>{v}</b><span>{_e(k)}</span></div>"
                                             for k, v in stats) + "</div>")
    warn = [w for w in ident.mech.warnings if "thermo" in w or "SPECIES" in w]
    if warn:
        b.append("<div class='flagbox' style='border-left-color:var(--warn);background:var(--warnbg)'>"
                 + "<br>".join(_e(w) for w in warn) + "</div>")
    if findings:
        b.append(f"<h2>Audit flags ({len(findings)})</h2>")
        for f in findings:
            b.append(f"<div class='flagbox'>{_species_link(f['species'], slugs, 0)} ({_e(f['formula'])}): given "
                     f"<span class='mono'>{_e(f['given'])}</span>, the evidence prefers "
                     f"<span class='mono'>{_e(f['suggested'])}</span> (log-odds {f['log_gap']:+.1f}). "
                     f"<span class='muted'>{_e(f['given_thermo'])}</span></div>")
    b.append("<h2>Species</h2><div class='toolbar'><input id='q' type='search' "
             "placeholder='Filter by name, formula or SMILES'><select id='fs'><option value=''>All statuses</option>"
             "<option value='identified'>identified</option><option value='known'>given</option>"
             "<option value='unresolved'>left open</option><option value='no-formula'>no thermo data</option>"
             "</select></div>")
    b.append("<div class='scroll'><table id='sp'><thead><tr><th>species</th><th>formula</th><th>status</th>"
             "<th>structure</th><th>probability</th><th>deciding evidence</th></tr></thead><tbody>")
    order = {"unresolved": 0, "no-formula": 1, "identified": 2, "known": 3}
    flagged = {f["species"] for f in findings}
    for st in sorted(species, key=lambda s: (s.name not in flagged, order.get(s.status, 9), -s.confidence)):
        c = st.chosen
        smi = c.smiles if c else (st.ranked()[0].smiles + " ?" if st.candidates else "–")
        if st.status == "known" and not st.audited:
            prob = "<span class='muted'>given</span>"
        elif st.status == "no-formula":
            prob = "<span class='muted'>–</span>"
        else:
            p = st.confidence if st.resolved else (st.ranked()[0].posterior if st.candidates else 0.0)
            prob = f"<span class='bar'><i style='width:{p * 100:.0f}%'></i></span>{p:.2f}"
        pill_cls = "flag" if st.name in flagged else st.status
        pill_txt = "flagged" if st.name in flagged else _status_label(st)
        key = f"{st.name} {st.formula_str} {smi}".lower()
        b.append(f"<tr data-k='{_e(key)}' data-s='{st.status}'><td>{_species_link(st.name, slugs, 0)}</td>"
                 f"<td class='mono'>{_e(st.formula_str)}</td><td><span class='pill {pill_cls}'>{pill_txt}</span></td>"
                 f"<td class='mono'>{_e(smi)}</td><td>{prob}</td><td class='small muted'>{_e(st.method)}</td></tr>")
    b.append("</tbody></table></div>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(_page(title, "".join(b), "overview", 0, model, INDEX_JS))
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='0; url=index.html'>"
                 "<a href='index.html'>Open the report</a>")

    # ---- species pages
    for st in species:
        with open(os.path.join(out_dir, "species", slugs[st.name] + ".html"), "w", encoding="utf-8") as fh:
            fh.write(_page(f"{st.name} · {model}", species_page(ident, st, slugs), "", 1, model))

    # ---- reactions page
    rx = {r.index: r for r in ident.mech.reactions}
    classes = Counter(s.reaction_class for s in sols.values())
    rb = ["<h1>Reactions</h1><div class='sub'>"
          f"{len(sols)} of {len(ident.mech.reactions)} reactions solved (all their species identified or given). "
          "Orange bonds change; numbers follow atoms across the arrow; curved arrows move electrons.</div>",
          "<div class='toolbar' style='margin-top:16px'><input id='q' type='search' "
          "placeholder='Filter by species or equation'><select id='c'><option value=''>All reaction classes</option>"
          + "".join(f"<option>{_e(k)}</option>" for k, _ in classes.most_common()) + "</select>"
          "<select id='sp'><option value=''>Any spin</option><option value='allowed'>spin-allowed</option>"
          "<option value='forbidden'>spin-forbidden</option></select></div>"]
    drawn = 0
    for idx, sol in sorted(sols.items()):
        r = rx[idx]
        draw = drawn < MAX_REACTION_DIAGRAMS
        card = _reaction_card(sol, r, slugs, 0, draw=draw)
        if draw and "<svg" in card:
            drawn += 1
        spin = "allowed" if sol.spin_allowed else ("forbidden" if sol.spin_allowed is False else "")
        key = (r.equation + " " + " ".join(r.species())).lower()
        rb.append(f"<div class='rx' id='r{idx}' data-k='{_e(key)}' data-c='{_e(sol.reaction_class)}' "
                  f"data-s='{spin}'>{card}</div>")
    if drawn >= MAX_REACTION_DIAGRAMS:
        rb.append(f"<p class='legend'>Diagrams are drawn for the first {MAX_REACTION_DIAGRAMS} reactions; "
                  "the rest list their bond changes.</p>")
    with open(os.path.join(out_dir, "reactions.html"), "w", encoding="utf-8") as fh:
        fh.write(_page(f"Reactions · {model}", "".join(rb), "reactions", 0, model, RX_JS))
    return os.path.join(out_dir, "index.html")


def write_report(ident, path: str, title: str = "Species identification", inputs: str = "") -> str:
    """Compatibility wrapper: writes the site into the folder of ``path``."""
    return write_site(ident, os.path.dirname(os.path.abspath(path)) or ".", title, inputs)


def to_json(ident) -> dict:
    out = {"species": [], "log": ident.log}
    for st in ident.species.values():
        out["species"].append({
            "name": st.name, "formula": st.formula_str, "status": st.status,
            "smiles": st.chosen.smiles if st.chosen else None,
            "confidence": round(st.confidence, 4), "method": st.method, "notes": st.notes,
            "audited": st.audited,
            "candidates": [{"smiles": c.smiles, "posterior": round(c.posterior, 4),
                            "sources": sorted(c.sources), "thermo": c.thermo_note}
                           for c in st.ranked()[:5]],
        })
    out["reactions"] = [{"index": i, "class": s.reaction_class, "sigma_edits": s.distance,
                         "spin_allowed": s.spin_allowed,
                         "summary": s.summary_lines()} for i, s in sorted(ident.solutions.items())]
    if getattr(ident, "audit_findings", None):
        out["audit"] = ident.audit_findings
    return out


def write_json(ident, path: str) -> str:
    with open(path, "w") as fh:
        json.dump(to_json(ident), fh, indent=1)
    return path
