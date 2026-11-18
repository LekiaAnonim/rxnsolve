# Changelog

## Unreleased

### Added
- Entropy and heat-capacity evidence. Each candidate's S298 and Cp (300 to
  1500 K) are predicted by group additivity plus R ln(2S+1) and compared with
  the species' own thermo card. Fitted values ship in
  `data/entropy_cp_additivity.json`.
- Kinetics evidence from competing H-abstraction channels: a larger A should go
  with more equivalent hydrogens, a lower Ea with the weaker C-H bond.
- Reaction rate parameters (A, n, Ea) are kept when parsing.
- Species pages show "Entropy and heat capacity" and "Kinetics" steps.
- The thermo library stores Cp at 300, 500, 800, 1000 and 1500 K.
- Benchmarks report large species (more than 10 heavy atoms) separately.

### Changed
- Entropy and Cp uncertainty widens up to 3x for groups seen in fewer than 10
  structures, and deviations within one sigma count as no evidence.
- A thermo card whose H298 fits no candidate is down-weighted as a whole,
  including its S298 and Cp.
- The deciding-evidence line names structures that kinetics ruled out.
- Faster identification without changing results: structure caches for Lewis
  feasibility, plausibility, group features and SMILES; fragment memoization in
  the bond-deletion search; an exact lower bound on bond edits; no isomer
  enumeration for formulas with 7 or more heavy atoms and DBE of at least 3.5,
  which never finish.

### Fixed
- Uniqueness compared spin as written, so a radical given in SMILES.txt and the
  same radical found by the engine counted as different species. It now
  compares the stated spin or the Lewis ground state.

## 2026-11-17

- First version: Lewis structures, minimum-bond-edit reaction solver, isomer
  enumeration, name hints, thermo library and group additivity, audit mode,
  spin handling, web app, report site, and benchmarks against RMG-models.
