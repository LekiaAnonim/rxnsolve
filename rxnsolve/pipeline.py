"""One identification / audit run, from files to the report site.

Shared by the command line (``python -m rxnsolve identify|audit``) and the
web app (``python -m rxnsolve serve``).
"""

from __future__ import annotations

import os
import time
from collections import Counter

from .chemkin import parse_chemkin, read_known_smiles
from .identify import Identifier
from .report import write_json, write_site


def run_pipeline(chem: str, thermo: str | None = None, known: str | None = None,
                 out_dir: str = "rxnsolve_out", mode: str = "identify", use_names: bool = True,
                 use_thermo: bool = True, exclude_models: set[str] | None = None,
                 use_spin_labels: bool = True, audit_only: bool = False, solve_reactions: bool = True,
                 threshold: float = 0.9, time_limit: float = 1800.0, verbose: bool = False,
                 title: str | None = None, on_ident=None) -> dict:
    """Run identify (or audit + identify) and write SMILES.txt, JSON and the report site.

    ``on_ident(ident)`` is called once the Identifier exists, so a caller can
    watch ``ident.log`` and ``ident.phase`` while it runs.
    """
    t0 = time.time()
    mech = parse_chemkin(chem, thermo)
    with_formula = sum(1 for s in mech.species if mech.formula(s))
    if not mech.species:
        raise ValueError("no species found - " + "; ".join(mech.warnings or ["is this a CHEMKIN file?"]))
    if with_formula == 0:
        raise ValueError(f"none of the {len(mech.species)} species has a readable thermo card, so no "
                         f"formula is known. Thermo file: {thermo or 'inside the mechanism'}")
    known_map = read_known_smiles(known) if known else {}
    if mode == "audit" and not known_map:
        raise ValueError("audit needs a SMILES file with the given structures (--known)")
    ident = Identifier(mech, known_map, use_names=use_names, verbose=verbose,
                       thermo="auto" if use_thermo else None, exclude_models=exclude_models,
                       use_spin_labels=use_spin_labels, threshold=threshold, time_limit=time_limit)
    ident.phase = "starting"
    if on_ident is not None:
        on_ident(ident)
    if with_formula < len(mech.species):
        ident.say(f"warning: {len(mech.species) - with_formula} of {len(mech.species)} species have no "
                  "thermo data and cannot be identified")
    if mode == "audit":
        ident.setup()
        ident.phase = "audit"
        ident.audit(verbose=verbose)
        if not audit_only:
            ident.phase = "identify"
            ident.run()
    else:
        ident.phase = "identify"
        ident.run()
    if solve_reactions:
        ident.phase = "reactions"
        ident.say("solving every reaction whose species are all known")
        ident.solve_all_reactions()
    ident.phase = "report"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "SMILES.txt"), "w") as fh:
        fh.write(ident.smiles_txt())
    write_json(ident, os.path.join(out_dir, "identification.json"))
    where = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(chem))), os.path.basename(chem))
    if title is None:
        if mode == "audit":
            title = ("Audit: " if audit_only else "Audit + identification: ") + where
        else:
            title = "Species identification: " + where
    inputs = " + ".join(os.path.basename(x) for x in (chem, thermo, known) if x)
    opts = []
    if not use_names:
        opts.append("names not used")
    if not use_thermo:
        opts.append("no thermochemistry")
    if exclude_models:
        opts.append("own thermo left out of the library")
    if opts:
        inputs += " · " + ", ".join(opts)
    seconds = time.time() - t0
    index = write_site(ident, out_dir, title=title, inputs=inputs, seconds=seconds)
    ident.phase = "done"
    counts = Counter(s.status for s in ident.species.values())
    ident.say(f"report written: {index}")
    return {"index": index, "out_dir": out_dir, "counts": dict(counts), "seconds": round(seconds, 1),
            "flagged": len(getattr(ident, "audit_findings", None) or []),
            "reactions_solved": len(ident.solutions), "reactions": len(mech.reactions), "ident": ident}
