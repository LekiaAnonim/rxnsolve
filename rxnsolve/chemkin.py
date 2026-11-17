"""CHEMKIN mechanism and NASA-7 thermo parser.

Reads ELEMENTS, SPECIES, THERMO and REACTIONS blocks (any of them may live in
a separate file).  From each thermo card we take the elemental composition -
the hard constraint for identification - plus H(298) and S(298).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .elements import is_element, normalize_symbol

R_KCAL = 1.987204e-3


@dataclass
class ThermoEntry:
    name: str
    composition: Counter
    phase: str = "G"
    t_low: float = 300.0
    t_mid: float = 1000.0
    t_high: float = 5000.0
    high: list[float] = field(default_factory=list)
    low: list[float] = field(default_factory=list)
    comment: str = ""

    def _coeffs(self, T):
        return self.low if T < self.t_mid else self.high

    def h_rt(self, T):
        a = self._coeffs(T)
        return a[0] + a[1] * T / 2 + a[2] * T**2 / 3 + a[3] * T**3 / 4 + a[4] * T**4 / 5 + a[5] / T

    def s_r(self, T):
        a = self._coeffs(T)
        return (a[0] * math.log(T) + a[1] * T + a[2] * T**2 / 2 + a[3] * T**3 / 3
                + a[4] * T**4 / 4 + a[6])

    def cp_r(self, T):
        a = self._coeffs(T)
        return a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4

    @property
    def h298(self) -> float | None:
        """Enthalpy of formation at 298 K, kcal/mol."""
        if len(self.low) < 7:
            return None
        return self.h_rt(298.15) * R_KCAL * 298.15

    @property
    def s298(self) -> float | None:
        """Entropy at 298 K, cal/mol/K."""
        if len(self.low) < 7:
            return None
        return self.s_r(298.15) * R_KCAL * 1000


@dataclass
class Reaction:
    index: int
    equation: str
    reactants: list[str]          # stoichiometry expanded: 2OH -> ["OH", "OH"]
    products: list[str]
    reversible: bool = True
    third_body: str | None = None  # "M", or a species name for (+AR)
    falloff: bool = False
    duplicate: bool = False
    line: int = 0

    def species(self) -> set[str]:
        return set(self.reactants) | set(self.products)


@dataclass
class Mechanism:
    elements: list[str] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    thermo: dict[str, ThermoEntry] = field(default_factory=dict)
    reactions: list[Reaction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def formula(self, name: str) -> Counter | None:
        t = self.thermo.get(name) or self.thermo.get(name.upper())
        return None if t is None else t.composition

    def reactions_of(self, name: str) -> list[Reaction]:
        return [r for r in self.reactions if name in r.reactants or name in r.products]


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
_RXN_LINE = re.compile(rf"^(.*?=.*?)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*(?:!.*)?$")
_FALLOFF = re.compile(r"\(\s*\+\s*([^()\s]+)\s*\)")
_SECTION = re.compile(r"^\s*(ELEMENTS|ELEM|SPECIES|SPEC|THERMO|REACTIONS|REAC|TRANSPORT|END)\b", re.I)


def _strip_comment(line: str) -> str:
    i = line.find("!")
    return line if i < 0 else line[:i]


def _read(path: str) -> list[str]:
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        return fh.read().splitlines()


def parse_chemkin(chem_path: str, thermo_path: str | None = None) -> Mechanism:
    mech = Mechanism()
    lines = _read(chem_path)
    section = None
    rxn_lines: list[tuple[int, str]] = []
    thermo_lines: list[str] = []
    for ln, raw in enumerate(lines, 1):
        line = _strip_comment(raw)
        m = _SECTION.match(line)
        if m:
            word = m.group(1).upper()
            rest = line[m.end():]
            if word == "END":
                section = None
                continue
            section = {"ELEM": "ELEMENTS", "SPEC": "SPECIES", "REAC": "REACTIONS"}.get(word, word)
            line = rest
            if section == "THERMO":
                continue
            if section == "REACTIONS":
                continue
        if section == "ELEMENTS":
            for tok in line.split():
                if "/" in tok:
                    tok = tok.split("/")[0]
                if tok and is_element(tok):
                    mech.elements.append(normalize_symbol(tok))
        elif section == "SPECIES":
            for tok in line.split():
                if tok.upper() == "END":
                    continue
                if tok not in mech.species:
                    mech.species.append(tok)
        elif section == "THERMO":
            thermo_lines.append(raw)
        elif section == "REACTIONS":
            rxn_lines.append((ln, raw))
    if thermo_lines:
        _parse_thermo_lines(thermo_lines, mech)
    if thermo_path:
        _parse_thermo_lines(_read(thermo_path), mech)
    _parse_reactions(rxn_lines, mech)
    missing = [s for s in mech.species if s not in mech.thermo]
    if missing:
        mech.warnings.append(f"{len(missing)} species have no thermo data: {missing[:10]}")
    return mech


def _float(s: str) -> float:
    s = s.strip().replace("D", "E").replace("d", "e")
    # Fortran-style blank exponent sign: "0.869E 01" means 0.869E+01
    s = re.sub(r"([Ee])\s+(\d)", r"\1+\2", s)
    return float(s) if s else 0.0


def _parse_thermo_lines(lines: list[str], mech: Mechanism) -> None:
    clean = [l.rstrip("\n") for l in lines]
    i = 0
    default_mid = 1000.0
    n = len(clean)
    while i < n:
        line = clean[i]
        s = line.rstrip()
        if not s or s.lstrip().startswith("!"):
            i += 1
            continue
        up = s.strip().upper()
        if up.startswith("THERMO") or up == "END":
            i += 1
            continue
        nums = s.split()
        if len(nums) in (3, 4) and all(re.fullmatch(_NUM, x) for x in nums[:3]) and not s.endswith("1"):
            try:
                default_mid = float(nums[1])
            except ValueError:
                pass
            i += 1
            continue
        if s.endswith("1") and i + 3 < n and clean[i + 1].rstrip().endswith("2") \
                and clean[i + 2].rstrip().endswith("3") and clean[i + 3].rstrip().endswith("4"):
            entry = _parse_entry(clean[i:i + 4], default_mid)
            if entry and entry.name not in mech.thermo:
                mech.thermo[entry.name] = entry
            i += 4
            continue
        i += 1


def _parse_entry(block: list[str], default_mid: float) -> ThermoEntry | None:
    l1 = block[0].ljust(80)
    name = l1[:18].split()[0] if l1[:18].split() else l1.split()[0]
    comp = Counter()
    fields = [l1[24:29], l1[29:34], l1[34:39], l1[39:44], l1[73:78]]
    for f in fields:
        el = f[:2].strip()
        cnt = f[2:].strip()
        if not el or not cnt:
            continue
        try:
            k = int(round(float(cnt)))
        except ValueError:
            continue
        if k <= 0:
            continue
        try:
            comp[normalize_symbol(el)] += k
        except KeyError:
            if el.upper() == "E":
                comp["E"] += k
    if not comp:
        # free-format composition (some generators): look for El count pairs
        for el, cnt in re.findall(r"([A-Z][A-Za-z]?)\s+(\d+)", l1[18:45]):
            if is_element(el):
                comp[normalize_symbol(el)] += int(cnt)
    phase = l1[44:45].strip() or "G"
    try:
        t_low = _float(l1[45:55])
        t_high = _float(l1[55:65])
        mid_s = l1[65:73].strip()
        t_mid = _float(mid_s) if mid_s else default_mid
    except ValueError:
        t_low, t_high, t_mid = 300.0, 5000.0, default_mid
    coeffs = []
    for line in block[1:]:
        line = line.ljust(80)
        for k in range(5):
            chunk = line[15 * k:15 * (k + 1)]
            if chunk.strip():
                try:
                    coeffs.append(_float(chunk))
                except ValueError:
                    pass
    high, low = (coeffs[:7], coeffs[7:14]) if len(coeffs) >= 14 else ([], [])
    return ThermoEntry(name, comp, phase, t_low, t_mid or default_mid, t_high, high, low,
                       comment=l1[18:24].strip())


def _parse_reactions(rxn_lines, mech: Mechanism) -> None:
    species = set(mech.species) | set(mech.thermo)
    by_len = sorted(species, key=len, reverse=True)
    for ln, raw in rxn_lines:
        line = _strip_comment(raw).strip()
        if not line or "=" not in line:
            if line.upper().startswith("DUP") and mech.reactions:
                mech.reactions[-1].duplicate = True
            continue
        if "/" in line.split("=")[0] and not _RXN_LINE.match(line):
            continue
        m = _RXN_LINE.match(line)
        if not m:
            continue
        eq = m.group(1).strip()
        rxn = _parse_equation(eq, by_len, species, len(mech.reactions) + 1, ln, mech)
        if rxn is not None:
            mech.reactions.append(rxn)


def _parse_equation(eq, by_len, species, index, ln, mech):
    if "<=>" in eq:
        lhs, rhs = eq.split("<=>", 1)
        rev = True
    elif "=>" in eq:
        lhs, rhs = eq.split("=>", 1)
        rev = False
    else:
        lhs, rhs = eq.split("=", 1)
        rev = True
    third = None
    falloff = False
    sides = []
    for side in (lhs, rhs):
        fm = _FALLOFF.search(side)
        if fm:
            third = fm.group(1)
            falloff = True
            side = side[:fm.start()] + side[fm.end():]
        names = _split_side(side, by_len, species)
        if names is None:
            mech.warnings.append(f"line {ln}: cannot parse side '{side.strip()}' of {eq}")
            return None
        if "M" in names and "M" not in species:
            names = [x for x in names if x != "M"]
            third = third or "M"
        sides.append(names)
    return Reaction(index, eq, sides[0], sides[1], rev, third, falloff, False, ln)


def _split_side(side, by_len, species):
    s = side.strip()
    out = []
    pos = 0
    L = len(s)
    while pos < L:
        while pos < L and (s[pos].isspace() or s[pos] == "+"):
            pos += 1
        if pos >= L:
            break
        name = _match_species(s, pos, by_len)
        coeff = 1
        if name is None:
            cm = re.match(r"(\d+(?:\.\d*)?)\s*", s[pos:])
            if cm:
                name = _match_species(s, pos + cm.end(), by_len)
                if name is not None:
                    coeff = int(round(float(cm.group(1))))
                    pos += cm.end()
        if name is None:
            if s[pos] in "Mm" and (pos + 1 == L or s[pos + 1] in " +"):
                out.append("M")
                pos += 1
                continue
            return None
        out.extend([name] * coeff)
        pos += len(name)
    return out


def _match_species(s, pos, by_len):
    for name in by_len:
        if s.startswith(name, pos):
            end = pos + len(name)
            if end == len(s) or s[end].isspace() or s[end] == "+":
                return name
    # case-insensitive fallback
    up = s.upper()
    for name in by_len:
        if up.startswith(name.upper(), pos):
            end = pos + len(name)
            if end == len(s) or s[end].isspace() or s[end] == "+":
                return name
    return None


def read_known_smiles(path: str) -> dict[str, str]:
    """RMG importer 'SMILES.txt' style: NAME<whitespace>SMILES per line."""
    out = {}
    for raw in _read(path):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        out[parts[0]] = parts[1].strip()
    return out
