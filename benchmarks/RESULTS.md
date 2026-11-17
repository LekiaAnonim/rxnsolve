# Benchmark results (2026-11-17)

Answer key: the `SMILES.txt` written by the RMG importer for each model in
`RMG-models`. The model under test is excluded from the thermo library.
*names* = species names used as soft hints. *blind* = names ignored; only the
formula, thermochemistry and reactions are used. No species are given as known.

| Mechanism | Species | Mode | Correct | Wrong | Left open | Precision | Truth in top 3 | Time |
|---|---|---|---|---|---|---|---|---|
| GRI-Mech 3.0 (+RMG extras) | 59 | names | 59 | 0 | 0 | 100% | 59 | 5 s |
| GRI-Mech 3.0 (+RMG extras) | 59 | blind | 59 | 0 | 0 | 100% | 59 | 5 s |
| USC Mech II | 110 | names | 106 | 2 | 2 | 98.1% | 110 | 71 s |
| USC Mech II | 110 | blind | 104 | 4 | 2 | 96.3% | 110 | 137 s |
| AramcoMech 1.3 (C4) | 251 | names | 245 | 3 | 3 | 98.8% | 249 | 90 s |
| AramcoMech 1.3 (C4) | 251 | blind | 245 | 3 | 3 | 98.8% | 249 | 61 s |

Precision = correct / (correct + wrong). Times are wall-clock with up to four
runs in parallel on one laptop. Numbers include the fix for Fortran blank-sign
exponents in thermo cards (`0.869E 01`), which had hidden the thermo of 3 USC
species and 1 Aramco species.

## How the evidence adds up (USC Mech II, names mode)

| Version | Correct | Wrong |
|---|---|---|
| reactions + names + plausibility only | 86 | 17 |
| + thermo library and group additivity | 104 | 3 |
| + library votes per formula, lumped-reaction weighting, ylide prior, ring names | 105 | 2 |
| + thermo parser reads blank-sign exponents | 106 | 2 |

Blind USC went from 59 to 104 correct once thermochemistry was added: rings and
aromatics (benzene, phenol, cyclopentadiene, oxiranes) need it as an anchor.

## What is still wrong or open, and why

* **Positional isomers that no evidence separates.** USC `OC6H4CH3`/`HOC6H4CH3`
  (o- vs p-cresol) and Aramco `C4H8OH-2O2` have the same reactions and nearly
  equal thermo. The open ones are left open, with the truth ranked 1–2.
* **`H2C4O`** (`C=C=C=C=O` vs `C#COC#C`): 3 witness reactions, close thermo.
  The truth is ranked 2nd.
* **`CH3CHOCH2` in blind mode.** Methyloxirane and methyl vinyl ether are only
  ~3 kcal/mol apart, and the 3-ring prior tips the choice to the ether. With
  names (`CH3CHOCH2` reads as a ring) it is correct.
* **`aC3H4` in blind mode.** Vinylcarbene beats allene because USC's
  `C2H2 + CH3 = aC3H4 + H`-type steps are chemically activated (multi-well)
  reactions. Counting σ edits favours the carbene (2 edits vs 4).
* **Possible answer-key issues.** In AramcoMech, `C4H5-2` is CH₂=C=C•–CH₃ in
  some mechanisms but is mapped to `C=[C]C=C` in the RMG-models key. `CC4H8O`
  (2- vs 3-methyloxetane) needs a closer look.

## Audit of existing identifications

`python -m rxnsolve audit` re-scores every given structure against
alternatives while the rest of the mechanism stays fixed.

| Model | Species checked | Flagged | Time |
|---|---|---|---|
| GRI-Mech 3.0 | 59 | 0 | <1 s |
| USC Mech II | 110 | 2: `C4H7` (+3.1, borderline), `C6H3` (+11.1) | 72 s |
| PCI2017/111-Jin | 109 | 1: `pC3H4` given `[CH]C=C`, evidence prefers propyne `C#CC` (+20.0) | 66 s |

The Jin flag is the misidentification listed below. The USC flags are the
isomer pairs that identification also disputes: worth a look, not proof.

## Answer-key problems found along the way

Building the thermo library exposed suspect entries in `RMG-models`:

* `PCI2017/111-Jin`: `pC3H4` (propyne) is recorded as `[CH]C=C` (vinylcarbene)
  and carries propyne's thermo.
* `PCI2013/411-Darcy`: `CC3H4` (cyclopropene) is recorded as `[CH]C=C`.

98 structures in the library have thermo values spread by more than
15 kcal/mol across models, which is worth an audit pass
(`python -m rxnsolve audit`).

Reproduce:

```bash
python benchmarks/benchmark.py GRI-mech-3.0   --mode blind
python benchmarks/benchmark.py USC_Mech_ii    --mode names
python benchmarks/benchmark.py AramcoMech_1.3 --mode blind
```
