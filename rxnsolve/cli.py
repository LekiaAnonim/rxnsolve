"""Command line interface.

    python -m rxnsolve identify chem.inp -t therm.dat [-k SMILES.txt] -o out/
    python -m rxnsolve audit    chem.inp -t therm.dat -k SMILES.txt -o out/
    python -m rxnsolve solve    "C + [OH] >> [CH3] + O" [--svg rxn.svg]
    python -m rxnsolve lewis    "C=C[CH2]" [--svg allyl.svg]
    python -m rxnsolve isomers  C3H6O
    python -m rxnsolve build-thermo --models ../RMG-models
    python -m rxnsolve serve --models ../RMG-models     # web app on http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter


def _formula(text: str) -> Counter:
    from .elements import normalize_symbol
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", text):
        c[normalize_symbol(el)] += int(n) if n else 1
    return c


def cmd_identify(a, audit=False):
    from .pipeline import run_pipeline

    res = run_pipeline(a.chem, a.thermo, a.known, out_dir=a.out, mode="audit" if audit else "identify",
                       use_names=not a.no_names, use_thermo=not a.no_thermo,
                       use_spin_labels=not a.no_spin_labels,
                       audit_only=getattr(a, "audit_only", False), solve_reactions=not a.no_reactions,
                       threshold=a.threshold, time_limit=a.time_limit, verbose=not a.quiet)
    ident = res["ident"]
    if audit:
        print(f"audit: {res['flagged']} of {len(ident.known)} given structures disagree with the "
              f"evidence; {len(ident.audit_partial)} only partly checked (time budget)")
    out = os.path.abspath(a.out)
    print(f"{res['counts']} in {res['seconds']:.1f} s\nresults saved in {out}/\n"
          f"  index.html           report (open in a browser)\n"
          f"  SMILES.txt           structures, RMG importer format\n"
          f"  identification.json  everything, machine-readable")


def _side(text):
    from .smiles import parse_smiles
    mols = []
    for tok in [t.strip() for t in text.split(" + ") if t.strip()]:
        m = re.match(r"^(\d+)\s+(.*)$", tok)
        n, smi = (int(m.group(1)), m.group(2)) if m else (1, tok)
        for _ in range(n):
            mol = parse_smiles(smi)
            mol.label = smi
            mols.append(mol)
    return mols


def cmd_solve(a):
    from .draw import SVG_CSS, reaction_svg
    from .mcd import solve_reaction

    if ">>" not in a.reaction:
        sys.exit('write the reaction as "A + B >> C + D" (SMILES, separated by " + ")')
    lhs, rhs = a.reaction.split(">>")
    R, P = _side(lhs), _side(rhs)
    sol = solve_reaction(R, P, max_total=a.max_edits)
    print(a.reaction.strip())
    for line in sol.summary_lines():
        print("  " + line)
    if a.svg:
        _write_svg(a.svg, reaction_svg(sol), SVG_CSS)


def cmd_lewis(a):
    from .draw import SVG_CSS, lewis_svg
    from .lewis import lewis_text, solve_lewis
    from .smiles import parse_smiles, to_smiles

    mol = parse_smiles(a.smiles)
    target = None if mol.multiplicity is None else mol.multiplicity - 1
    structs = solve_lewis(mol, unpaired=target) or solve_lewis(mol)
    print(f"{a.smiles}: {mol.formula_string()}, {len(structs)} resonance structure(s)")
    for k, s in enumerate(structs, 1):
        print(f"\n[{k}] {to_smiles(mol, s)}")
        print(lewis_text(mol, s))
    if a.svg and structs:
        _write_svg(a.svg, lewis_svg(mol, structs[0]), SVG_CSS)


def _write_svg(path, svg, css):
    svg = svg.replace(">", "><style>:root{--ink:#111827;--muted:#6b7280;--change:#ea580c;--o:#c0262d;--n:#1d4ed8;"
                           "--mono:Menlo,monospace}" + css + "</style>", 1)
    with open(path, "w") as fh:
        fh.write(svg)
    print(f"wrote {path}")


def cmd_isomers(a):
    from .isomers import enumerate_isomers, plausibility
    from .smiles import canonical_smiles

    f = _formula(a.formula)
    iso = enumerate_isomers(f, max_heavy=a.max_heavy)
    if iso is None:
        sys.exit(f"{a.formula}: too many isomers to enumerate with the current limits")
    ranked = sorted(((plausibility(m), canonical_smiles(m)) for m in iso), key=lambda t: -t[0][0])
    print(f"{a.formula}: {len(iso)} constitutional isomers with a valid Lewis structure")
    for (lp, why), smi in ranked[: a.top]:
        print(f"  {lp:+5.1f}  {smi:28s} {'; '.join(why)}")


def cmd_build_thermo(a):
    from .thermo import GA_FILE, LIBRARY_FILE, build_library, save_group_additivity, save_library

    lib = build_library(a.models, verbose=not a.quiet)
    save_library(lib)
    save_group_additivity(lib, verbose=True)
    print(f"{len(lib['records'])} records, {len(lib['features'])} structures -> {LIBRARY_FILE}")
    print(f"group values -> {GA_FILE}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxnsolve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("identify", "audit"):
        p = sub.add_parser(name, help=("identify species" if name == "identify"
                                       else "check given structures against the evidence"))
        p.add_argument("chem", help="CHEMKIN mechanism (SPECIES + REACTIONS)")
        p.add_argument("-t", "--thermo", help="thermo file (if not inside chem)")
        p.add_argument("-k", "--known", help="SMILES.txt of species already known (NAME SMILES)")
        p.add_argument("-o", "--out", default="rxnsolve_out", help="output folder")
        p.add_argument("--no-names", action="store_true",
                       help="ignore species names as structure hints")
        p.add_argument("--no-spin-labels", action="store_true",
                       help="also ignore spin labels in names: (S), CH2*, SING, (1D), (T)")
        p.add_argument("--no-thermo", action="store_true", help="ignore thermochemistry")
        p.add_argument("--no-reactions", action="store_true", help="skip solving every reaction")
        p.add_argument("--threshold", type=float, default=0.9)
        if name == "audit":
            p.add_argument("--audit-only", action="store_true",
                           help="check the given structures only; do not identify the rest")
        p.add_argument("--time-limit", type=float, default=1800)
        p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("solve", help="solve one reaction given as SMILES")
    p.add_argument("reaction")
    p.add_argument("--svg")
    p.add_argument("--max-edits", type=int, default=6)
    p = sub.add_parser("lewis", help="Lewis structures of a SMILES")
    p.add_argument("smiles")
    p.add_argument("--svg")
    p = sub.add_parser("isomers", help="enumerate constitutional isomers of a formula")
    p.add_argument("formula")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--max-heavy", type=int, default=9)
    p = sub.add_parser("serve", help="web app: connect RMG-models, pick a mechanism, identify it")
    p.add_argument("--models", default=os.environ.get("RXNSOLVE_MODELS"),
                   help="folder of mechanisms (RMG-models layout); can also be set in the page")
    p.add_argument("--runs", default="rxnsolve_runs", help="where run reports are written")
    p.add_argument("--port", type=int, default=8770)
    p = sub.add_parser("build-thermo", help="build the thermo library from identified models")
    p.add_argument("--models", required=True)
    p.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "identify":
        cmd_identify(a)
    elif a.cmd == "audit":
        cmd_identify(a, audit=True)
    elif a.cmd == "solve":
        cmd_solve(a)
    elif a.cmd == "lewis":
        cmd_lewis(a)
    elif a.cmd == "isomers":
        cmd_isomers(a)
    elif a.cmd == "serve":
        from .webapp import serve
        serve(a.models, a.runs, port=a.port)
    elif a.cmd == "build-thermo":
        cmd_build_thermo(a)


if __name__ == "__main__":
    main()
