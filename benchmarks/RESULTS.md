# Benchmark results (2026-11-18)

Answer key: the `SMILES.txt` written by the RMG importer for each model in
`RMG-models`. The model under test is excluded from the thermo library.
*names* = species names used as soft hints. *blind* = names not used as structure
hints; only the formula, thermochemistry and reactions are used (spin labels such
as `(S)` are still read; scoring compares connectivity, so they don't change the
counts). No species are given as known.

| Mechanism | Species | Mode | Correct | Wrong | Left open | Precision | Truth in top 3 | Time |
|---|---|---|---|---|---|---|---|---|
| GRI-Mech 3.0 (+RMG extras) | 59 | names | 59 | 0 | 0 | 100% | 59 | 30 s |
| GRI-Mech 3.0 (+RMG extras) | 59 | blind | 59 | 0 | 0 | 100% | 59 | 30 s |
| USC Mech II | 110 | names | 106 | 2 | 2 | 98.1% | 110 | 49 s |
| USC Mech II | 110 | blind | 105 | 3 | 2 | 97.2% | 110 | 67 s |
| AramcoMech 1.3 (C4) | 251 | names | 245 | 4 | 2 | 98.4% | 249 | 59 s |
| AramcoMech 1.3 (C4) | 251 | blind | 245 | 4 | 2 | 98.4% | 248 | 64 s |

Precision = correct / (correct + wrong). Times are wall-clock with six runs in
parallel on one laptop; each run includes about 20 s to fit the thermo models
without the mechanism under test. Numbers include the fix for Fortran blank-sign
exponents in thermo cards (`0.869E 01`), which had hidden the thermo of 3 USC
species and 1 Aramco species.

## How much depends on thermo copied between models

Most mechanisms copy thermo from shared sources (Burcat, GRI, earlier models).
For most benchmark species, the same NASA polynomial already sits in another
RMG-models folder under a known structure:

| Mechanism | Thermo match in another model, right structure | Match, wrong structure | No match |
|---|---|---|---|
| GRI-Mech 3.0 | 83% | 0% | 17% |
| USC Mech II | 97% | 3% | 0% |
| AramcoMech 1.3 | 94% | 6% | 0% |

The ablation below switches the thermo-library lookup off (no record votes, no
library H298). Group additivity for H298, S298 and Cp stays on, fitted without
the model under test. This is the situation for a species whose thermo exists in
no other model.

| Mechanism | Mode | With lookup: correct / wrong / open | Without lookup, H298 only | Without lookup, + entropy and Cp |
|---|---|---|---|---|
| GRI-Mech 3.0 | blind | 59 / 0 / 0 | 58 / 0 / 1 | 59 / 0 / 0 |
| USC Mech II | names | 106 / 2 / 2 | 96 / 8 / 6 | 98 / 6 / 6 |
| USC Mech II | blind | 105 / 3 / 2 | 89 / 8 / 13 | 88 / 5 / 17 |
| AramcoMech 1.3 | names | 245 / 4 / 2 | 227 / 14 / 10 | 236 / 8 / 7 |
| AramcoMech 1.3 | blind | 245 / 4 / 2 | 206 / 8 / 37 | 224 / 12 / 15 |

The lookup adds about 10 to 20 species per mechanism. Without it, blind
identification gets 80 to 89% correct with 95% precision, and most of the
remaining loss is "left open", not wrong answers. Group additivity is learned
from other models' species, so it carries general thermochemistry from them,
but it never matches a species' own polynomial.

### Entropy and heat capacity

Each candidate's S298 and Cp (300 to 1500 K) are predicted by group additivity
plus R ln(2S+1) for spin and compared with the species' own thermo card.
Held-out errors: S298 median 1.1 cal/mol/K (mean 2.1), Cp median 0.4 to 0.65.
On thermo evidence alone, ranking the truth among every isomer of its formula:

| Model | H298 only, truth first | H + S + Cp, truth first | Ring species first |
|---|---|---|---|
| USC Mech II | 40% | 64% | 1 to 9 of 19 |
| AramcoMech 1.3 | 42% | 56% | 2 to 7 of 14 |
| GRI-Mech 3.0 | 68% | 81% | none |

The first version over-trusted two cases: groups seen in very few structures
(vinylidene, whose rarest group appears once) and the 1 to 2 cal/mol/K
differences between positional isomers. Sigma now widens up to 3x for groups
seen in fewer than 10 structures, deviations within one sigma count as no
evidence, and a card whose H298 fits no candidate has its S and Cp down-weighted
too (this fixed three GRI species with a bad card, CH3OOH at -54.9 kcal/mol).

### Kinetics

Competing H-abstraction channels from one parent were tested on 5,001 pairs in
7 identified mechanisms: the larger A goes with more equivalent hydrogens in
90% of pairs, and the lower Ea with the weaker C-H in 90%. Used as evidence, it
changed no result on GRI, USC or Aramco: the species still wrong or open there
are not products of competing channels (4 of 21 in USC, 2 of 45 in Aramco), and
their siblings were already identified. It decides only one species (C3H4-A,
correctly). Its value on large low-temperature mechanisms is not measured yet.

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
  It is picked wrongly in USC and Aramco. In Aramco it used to be left open,
  until the larger thermo library from the parser fixes tipped it. The truth is
  ranked 1st–2nd.
* **`CH3CHOCH2` in blind mode.** Methyloxirane and methyl vinyl ether are only
  ~3 kcal/mol apart, and the 3-ring prior tips the choice to the ether. With
  names (`CH3CHOCH2` reads as a ring) it is correct.
* **`aC3H4` in blind mode.** Vinylcarbene beats allene because USC's
  `C2H2 + CH3 = aC3H4 + H`-type steps are chemically activated (multi-well)
  reactions. Counting σ edits favours the carbene (2 edits vs 4).
* **Possible answer-key issues.** In AramcoMech, `C4H5-2` is CH₂=C=C•–CH₃ in
  some mechanisms but is mapped to `C=[C]C=C` in the RMG-models key. `CC4H8O`
  (2- vs 3-methyloxetane) needs a closer look.

## Other models run through the web app

| Model | Species | Agree with SMILES.txt | Differ | Time |
|---|---|---|---|---|
| 2-BTP (Br/F flame-inhibitor chemistry), names, own thermo excluded | 188 | 183 | 4 (H2C4O, C6H3, the cresol/methylphenoxy pair) | 65 s |

For the methylphenoxy and cresol species, 2-BTP's key says *meta* while USC
Mech II's key says *ortho*. The RMG importer's own answers for these
positional isomers are not consistent between models.

## Large species

Two models with near-complete answer keys and many species above 10 heavy
atoms, names mode, before the entropy and Cp evidence:

| Model | Large species | Correct | Wrong | Open |
|---|---|---|---|---|
| Chernov | 23 | 1 | 7 | 15 |
| Narayanaswamy | 29 | 4 | 0 | 25 |

Large aromatics are the weak spot: there are no complete isomer lists above 9
heavy atoms, and thermo and reactions separate their isomers poorly. A faster
setting (one bond deletion per side above 14 heavy atoms, 20 candidates instead
of 50) added 5 wrong large species on Chernov and was not adopted.

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
