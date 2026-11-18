# rxnsolve — reaction-constrained species identification for CHEMKIN files

A second species-identification system for Prometheus, independent of RMG-Py.
The RMG importer (`importChemkin.py`) asks *"does an RMG reaction family
reproduce this CHEMKIN reaction?"*. rxnsolve instead solves the chemistry
directly: for each candidate structure it works out the Lewis structure
(bond orders, lone pairs, radicals, formal charges), the atom-to-atom mapping
of every reaction, the bonds broken and formed, the atoms that move, and the
electron flow. It then asks which structure lets the whole mechanism run with
the fewest bond changes, consistent with the thermochemistry.

Pure Python 3.9+, standard library only. No RDKit, no RMG, no cluster.

## What it does

| Piece | Module | Output |
|---|---|---|
| Lewis structures | `lewis.py` | bond orders, lone pairs, unpaired electrons, formal charges, all resonance forms, electron census |
| Reaction solver | `mcd.py` | minimum-chemical-distance atom map, σ bonds broken/formed, π-bond order changes, atoms transferred between molecules, per-atom Δ(lone pairs, radicals, charge), fishhook/full arrows, reaction class |
| Isomer enumeration | `isomers.py` | every constitutional isomer of a formula with a valid Lewis structure + a chemical-plausibility prior |
| Name hints | `names.py` | aliases (GRI/USC/Aramco/LLNL), condensed formulas (`CH3CH(OH)CH3`), LLNL patterns (`C7H14OOH2-4`, `NC7KET24`), N-/I-/S-/T- prefixes |
| Thermo evidence | `thermo.py` | thermo library + Benson-style group additivity learned from `RMG-models` |
| Identification | `identify.py` | iterative constraint propagation over the whole mechanism; audit of existing `SMILES.txt` |
| Drawing / report | `draw.py`, `report.py` | SVG Lewis structures (lone-pair and radical dots, charges) and atom-mapped mechanisms with curly arrows, in one HTML file |

## Electron spin (singlet, doublet, triplet, ...)

Identity is the connectivity, so CH₂ and CH₂(S) share one structure. Spin is a
separate label, the multiplicity 2S+1, set in this order:

1. the `--known` SMILES. An explicit prefix (`singlet[CH2]`, `triplet[O]`) always
   wins. A bare bracket SMILES follows RMG's convention, where every unfilled
   valence is a radical (`[CH2]` is a triplet). For the ground-state species in
   item 3 the table overrides it instead, so `[C]` is read as triplet carbon
   rather than a quintet, and `[CH]` as a doublet rather than a quartet.
2. spin labels in the name: `(S)`, `CH2S`, `S-CH2`, `CH2*`, `SING`, `(1D)` → singlet;
   `(T)`, `TRIP` → triplet. `--no-spin-labels` turns this off.
3. Hund's-rule ground states where Lewis theory pairs electrons wrongly: O₂, CH₂, O,
   C, S, Si, NH, S₂, SO, NCN, CCO are triplets; N is a quartet; CH is a doublet
4. otherwise the Lewis solver pairs every electron it can: an odd electron count
   gives a doublet, an even count gives a singlet

The multiplicity is the target number of unpaired electrons for the Lewis solver.
It spreads radicals over atoms before it splits a lone pair, so triplet O₂ is
·O–O· and singlet CH₂ keeps a lone pair. Output SMILES carry a prefix
(`singlet[CH2]`, `singlet[O]`) whenever RMG's bracket convention would otherwise
imply a different spin. The reaction solver checks spin conservation (Wigner's
rule: some total spin must be reachable from both sides) and marks steps such
as `triplet CH2 + H2 -> CH4` as spin-forbidden.

## How a species is identified

For a species *S* and candidate structure *c*:

```
log P(c | mechanism) = prior(c) + name(c) + thermo(c) + Σ_reactions −1.5 · w_r · (d_r(c) − min d_r)
```

* **formula:** hard constraint from the NASA thermo card.
* **prior(c):** Lewis-structure plausibility. It penalizes carbenes, charge separation, ylides, small rings, O–O–O chains and enols, and gives aromatic rings a bonus.
* **name(c):** a soft hint only.
* **thermo(c):** two sources:
  * the library vote: same-formula records whose H₂₉₈/S₂₉₈ match.
  * otherwise, group additivity vs. the mechanism's NASA H₂₉₈.

  This term is robust to bad thermo cards: a mismatch saturates at −3, and when no candidate fits, the species' thermo counts a quarter as much.
* **d_r(c):** the minimum number of σ bonds broken + formed to run reaction *r* with *S = c*, with every other species of *r* already resolved. Elementary steps need 1–3 edits.
* **w_r:** 0.25 for lumped steps (3+ species on one side). It is halved again when no candidate explains the step in under 3 edits.

Candidates come from three places:

* the formula (complete enumeration for small species)
* the name
* **reaction-driven generation:** structures assembled from the fragments left when the known species are carved out of the other side of a reaction. This is the main source for large species.

Species that pass the confidence threshold (0.90, relaxed to 0.5 if stuck) are fixed. That turns more reactions into witnesses for the rest. Two species never receive the same structure.

## Benchmarks against the RMG importer's identifications

Each `RMG-models/*/SMILES.txt` (from `importChemkin.py`) is hidden and used as
the answer key. The mechanism under test is excluded from the thermo library.
*blind* = species names are not used as structure hints (formula + thermo +
reactions only). Spin labels such as `(S)` are still read unless
`--no-spin-labels` is given. They don't affect the scores, which compare
connectivity.

Run `python benchmarks/benchmark.py <model> --mode names|blind`. Current numbers are
in `benchmarks/RESULTS.md`.

## Usage

```bash
# identify every species (optionally seed with species you already know)
python -m rxnsolve identify chem.inp -t therm.dat -k SMILES.txt -o out/
#   -> out/report.html  out/SMILES.txt (RMG importer format)  out/identification.json

# check an existing identification (e.g. an RMG-models SMILES.txt)
python -m rxnsolve audit chem.inp -t therm.dat -k SMILES.txt -o audit/
#   each given structure is re-scored against alternatives with every other
#   species held fixed; flagged when an alternative is > e^3 (20x) more likely.
#   Then the species the SMILES file does not cover are identified, seeded by
#   the given ones (--audit-only skips that step).

# solve one reaction: bonds broken/formed, atoms moved, arrows, class
python -m rxnsolve solve "CCO[O] >> C=C + [O]O" --svg ho2_elim.svg

# Lewis structures / resonance / electron census
python -m rxnsolve lewis "[O-][O+]=O"

# all constitutional isomers of a formula, ranked by plausibility
python -m rxnsolve isomers C3H6O

# rebuild the thermo library from identified models
python -m rxnsolve build-thermo --models ../RMG-models
```

Python API:

```python
from rxnsolve.identify import identify
ident = identify("chem.inp", "therm.dat", known="SMILES.txt", use_names=True)
ident.solve_all_reactions()
print(ident.smiles_txt())
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Known limits

* Stereochemistry (cis/trans, R/S) is not represented. Positional isomers that no
  reaction or thermo card can separate (e.g. o- vs p-cresol) stay ambiguous; the
  report lists the ranked alternatives.
* σ-edit counting favours very reactive structures (carbenes, ylides) in lumped
  reactions. Priors and the thermo term correct most of this, but lumped
  mechanisms give weaker evidence.
* The thermo library is only as good as the identifications it is built from.
  `audit` flags suspect entries. For example, `PCI2017/111-Jin` lists `pC3H4`
  (propyne) as `[CH]C=C`.
* Pure Python: a 250-species / 1,500-reaction mechanism takes about 1–3 minutes.
* `audit` checks each species on at most 24 witness reactions (elementary and
  small first) and 40 alternatives, within 20 s. Species that run out of time
  are listed as partly checked. It takes about 1 minute for a 110-species model.
* Identification gives each open species 30 s per round. Bond-edit distances
  are cached, so the next round continues where the last one stopped. A species
  that was cut off is only decided once at least 10 of its reactions have been
  scored.
* Output prefixes `doublet`/`quartet` (e.g. `doublet[CH]`) are understood by
  rxnsolve. RMG's importer may only read `singlet`/`triplet`.
