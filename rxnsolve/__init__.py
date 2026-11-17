"""rxnsolve - reaction-constrained species identification for CHEMKIN mechanisms.

Pure Python, no RDKit or RMG needed.  Main parts:

* molecule  - explicit-H molecular graph and canonical labelling
* smiles    - SMILES reader / writer
* lewis     - Lewis structures: bond orders, lone pairs, radicals, formal charges
* mcd       - reaction solver: minimum bond-edit atom mapping, electron flow
* isomers   - constitutional isomer enumeration from a formula
* names     - structure hints from CHEMKIN species names
* chemkin   - CHEMKIN mechanism / thermo parser
* identify  - the identification engine
"""

__version__ = "0.1.0"

from .molecule import Molecule  # noqa: F401
from .smiles import parse_smiles, to_smiles, canonical_smiles  # noqa: F401
from .lewis import solve_lewis, best_structure  # noqa: F401
