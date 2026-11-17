"""Element data used by the Lewis, enumeration and reaction solvers.

Every quantity here is chemistry, not configuration:

* ``ve``        - valence electrons of the neutral atom
* ``octet``     - electron capacity of the valence shell (2 for H/He, 8 for
                  second-row atoms, 12 for third-row atoms that may expand)
* ``max_sigma`` - highest number of sigma neighbours the enumerator will give
                  the atom in a neutral, ordinary molecule
* ``charges``   - formal charges the Lewis solver may place on the atom
"""

from __future__ import annotations

ELEMENTS: dict[str, dict] = {
    "H":  dict(Z=1,  mass=1.008,   ve=1, octet=2,  max_sigma=1, charges=(0,)),
    "He": dict(Z=2,  mass=4.0026,  ve=2, octet=2,  max_sigma=0, charges=(0,)),
    "Li": dict(Z=3,  mass=6.94,    ve=1, octet=2,  max_sigma=1, charges=(0, 1)),
    "Be": dict(Z=4,  mass=9.0122,  ve=2, octet=4,  max_sigma=2, charges=(0,)),
    "B":  dict(Z=5,  mass=10.81,   ve=3, octet=8,  max_sigma=3, charges=(0, -1)),
    "C":  dict(Z=6,  mass=12.011,  ve=4, octet=8,  max_sigma=4, charges=(0, -1, 1)),
    "N":  dict(Z=7,  mass=14.007,  ve=5, octet=8,  max_sigma=3, charges=(0, 1, -1)),
    "O":  dict(Z=8,  mass=15.999,  ve=6, octet=8,  max_sigma=2, charges=(0, 1, -1)),
    "F":  dict(Z=9,  mass=18.998,  ve=7, octet=8,  max_sigma=1, charges=(0,)),
    "Ne": dict(Z=10, mass=20.180,  ve=8, octet=8,  max_sigma=0, charges=(0,)),
    "Na": dict(Z=11, mass=22.990,  ve=1, octet=2,  max_sigma=1, charges=(0, 1)),
    "Mg": dict(Z=12, mass=24.305,  ve=2, octet=4,  max_sigma=2, charges=(0,)),
    "Al": dict(Z=13, mass=26.982,  ve=3, octet=8,  max_sigma=3, charges=(0,)),
    "Si": dict(Z=14, mass=28.085,  ve=4, octet=8,  max_sigma=4, charges=(0,)),
    "P":  dict(Z=15, mass=30.974,  ve=5, octet=12, max_sigma=5, charges=(0, 1, -1)),
    "S":  dict(Z=16, mass=32.06,   ve=6, octet=12, max_sigma=4, charges=(0, 1, -1)),
    "Cl": dict(Z=17, mass=35.45,   ve=7, octet=12, max_sigma=1, charges=(0,)),
    "Ar": dict(Z=18, mass=39.948,  ve=8, octet=8,  max_sigma=0, charges=(0,)),
    "K":  dict(Z=19, mass=39.098,  ve=1, octet=2,  max_sigma=1, charges=(0, 1)),
    "Ca": dict(Z=20, mass=40.078,  ve=2, octet=4,  max_sigma=2, charges=(0,)),
    "Br": dict(Z=35, mass=79.904,  ve=7, octet=12, max_sigma=1, charges=(0,)),
    "Kr": dict(Z=36, mass=83.798,  ve=8, octet=8,  max_sigma=0, charges=(0,)),
    "I":  dict(Z=53, mass=126.90,  ve=7, octet=12, max_sigma=1, charges=(0,)),
    "Xe": dict(Z=54, mass=131.29,  ve=8, octet=8,  max_sigma=0, charges=(0,)),
}

# SMILES "organic subset": may be written without brackets.
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
SMILES_VALENCES = {
    "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5),
    "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,),
}
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe"}

_UPPER = {sym.upper(): sym for sym in ELEMENTS}


def normalize_symbol(symbol: str) -> str:
    """Return the canonical capitalisation of an element symbol ('AR' -> 'Ar')."""
    s = symbol.strip()
    if s in ELEMENTS:
        return s
    up = s.upper()
    if up in _UPPER:
        return _UPPER[up]
    raise KeyError(f"unknown element symbol {symbol!r}")


def is_element(symbol: str) -> bool:
    try:
        normalize_symbol(symbol)
        return True
    except KeyError:
        return False


def valence_electrons(el: str) -> int:
    return ELEMENTS[el]["ve"]


def octet(el: str) -> int:
    return ELEMENTS[el]["octet"]


def standard_valence(el: str, charge: int = 0) -> int:
    """Usual number of bonds for an atom, using the isoelectronic rule.

    C-, N, O+ all behave like nitrogen (3 bonds); N+ behaves like carbon (4).
    """
    ve = ELEMENTS[el]["ve"] - charge
    if el in ("H", "Li", "Na", "K"):
        return max(0, 1 - abs(charge))
    if ve <= 0:
        return 0
    if ve <= 4:
        return ve
    return max(0, 8 - ve)


def max_bond_sum(el: str, charge: int) -> int:
    """Largest bond-order sum an atom can carry with this formal charge.

    From n = ve - q - b >= 0 and 2b + n <= octet.
    """
    ve = ELEMENTS[el]["ve"]
    oc = ELEMENTS[el]["octet"]
    return max(0, min(ve - charge, oc - ve + charge))


def hill_formula(counts: dict[str, int]) -> str:
    """Hill-system formula string: C, H, then alphabetical (no C: all alphabetical)."""
    counts = {k: v for k, v in counts.items() if v}
    parts = []

    def fmt(el):
        n = counts[el]
        return el if n == 1 else f"{el}{n}"

    if "C" in counts:
        parts.append(fmt("C"))
        if "H" in counts:
            parts.append(fmt("H"))
        rest = sorted(k for k in counts if k not in ("C", "H"))
    else:
        rest = sorted(counts)
    parts.extend(fmt(el) for el in rest)
    return "".join(parts)
