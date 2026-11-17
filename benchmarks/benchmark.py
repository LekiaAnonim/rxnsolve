"""Benchmark rxnsolve against species already identified with the RMG importer.

Each RMG-models folder has a SMILES.txt written by importChemkin.py; we hide
it, run the identifier, and compare constitutions.

    python benchmarks/benchmark.py GRI-mech-3.0 --mode names
    python benchmarks/benchmark.py USC_Mech_ii --mode blind --seed 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rxnsolve.chemkin import parse_chemkin, read_known_smiles  # noqa: E402
from rxnsolve.identify import Identifier  # noqa: E402
from rxnsolve.smiles import parse_smiles  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "RMG-models")


def find_inputs(folder: str):
    """Read --species/--reactions/--thermo/--known from the folder's import.sh."""
    path = os.path.join(MODELS, folder)
    script = open(os.path.join(path, "import.sh")).read()
    def arg(flag):
        m = re.search(rf"--{flag}\s+(\S+)", script)
        return os.path.join(path, m.group(1)) if m else None
    return arg("reactions"), arg("thermo"), os.path.join(path, "SMILES.txt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["names", "blind"], default="names",
                    help="names: use species names; blind: ignore names entirely")
    ap.add_argument("--seed", type=int, default=0,
                    help="number of species to give as known (random, reproducible)")
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--time-limit", type=float, default=1800)
    ap.add_argument("--no-thermo", action="store_true", help="ignore thermochemistry evidence")
    args = ap.parse_args(argv)

    chem, thermo, smiles = find_inputs(args.model)
    mech = parse_chemkin(chem, thermo)
    truth_raw = read_known_smiles(smiles)
    truth = {}
    for k, v in truth_raw.items():
        if k in mech.species:
            try:
                truth[k] = parse_smiles(v)
            except Exception:
                pass
    rng = random.Random(42)
    names = sorted(truth)
    seeds = rng.sample(names, min(args.seed, len(names))) if args.seed else []
    known = {k: truth_raw[k] for k in seeds}

    t0 = time.time()
    ident = Identifier(mech, known, use_names=(args.mode == "names"), verbose=args.verbose,
                       time_limit=args.time_limit,
                       thermo=None if args.no_thermo else "auto",
                       exclude_models={args.model})
    ident.run()
    dt = time.time() - t0

    rows = []
    correct = wrong = unresolved = 0
    in_top3 = 0
    for name in mech.species:
        st = ident.species[name]
        t = truth.get(name)
        if t is None or name in known:
            continue
        tkey = t.constitution_key()
        ranked = st.ranked()
        rank = next((i for i, c in enumerate(ranked) if c.key == tkey), None)
        if rank is not None and rank < 3:
            in_top3 += 1
        if st.status == "identified":
            ok = st.chosen.key == tkey
            correct += ok
            wrong += not ok
            verdict = "OK" if ok else "WRONG"
        else:
            unresolved += 1
            verdict = "open"
        rows.append(dict(name=name, formula=st.formula_str, verdict=verdict,
                         got=st.chosen.smiles if st.chosen else None,
                         truth=truth_raw[name], conf=round(st.confidence, 3),
                         method=st.method, rank_of_truth=rank, n_cand=len(st.candidates)))
    total = correct + wrong + unresolved
    print(f"\n=== {args.model}  mode={args.mode}  seeds={len(seeds)}  ({dt:.1f} s)")
    print(f"species scored: {total}   correct: {correct}   wrong: {wrong}   unresolved: {unresolved}"
          f"   truth in top-3: {in_top3}")
    if total:
        print(f"accuracy (correct / scored): {correct / total:.1%}   "
              f"precision (correct / identified): {correct / max(1, correct + wrong):.1%}")
    for r in rows:
        if r["verdict"] != "OK":
            print(f"  {r['verdict']:5s} {r['name']:14s} {r['formula']:9s} got={r['got']!s:24s} "
                  f"truth={r['truth']:22s} conf={r['conf']:.2f} rank={r['rank_of_truth']} "
                  f"n={r['n_cand']} {r['method']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(model=args.model, mode=args.mode, seeds=seeds, seconds=dt,
                           correct=correct, wrong=wrong, unresolved=unresolved, rows=rows), fh, indent=1)
    return correct, wrong, unresolved


if __name__ == "__main__":
    main()
