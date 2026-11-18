"""Unit tests: python3 -m unittest discover -s tests -v"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rxnsolve.chemkin import parse_chemkin  # noqa: E402
from rxnsolve.identify import Identifier  # noqa: E402
from rxnsolve.isomers import enumerate_isomers  # noqa: E402
from rxnsolve.lewis import solve_lewis  # noqa: E402
from rxnsolve.mcd import generate_completions, sigma_distance, solve_reaction  # noqa: E402
from rxnsolve.names import name_candidates  # noqa: E402
from rxnsolve.smiles import canonical_smiles, parse_smiles  # noqa: E402


def S(s):
    return parse_smiles(s)


class TestMolecule(unittest.TestCase):
    def test_same_constitution(self):
        self.assertEqual(S("OCC").constitution_key(), S("CCO").constitution_key())
        self.assertEqual(S("C=C[CH2]").constitution_key(), S("[CH2]C=C").constitution_key())
        self.assertEqual(S("c1ccccc1").constitution_key(), S("C1=CC=CC=C1").constitution_key())

    def test_different_constitution(self):
        self.assertNotEqual(S("CCO").constitution_key(), S("COC").constitution_key())
        self.assertNotEqual(S("[CH2]CC").constitution_key(), S("C[CH]C").constitution_key())

    def test_formula(self):
        self.assertEqual(S("CC(=O)O").formula_string(), "C2H4O2")
        self.assertEqual(S("[Ar]").formula_string(), "Ar")

    def test_smiles_round_trip(self):
        for s in ["CCO", "C=C[CH2]", "[C-]#[O+]", "O=C=O", "CC1CO1", "c1ccccc1O", "[O][O]"]:
            m = S(s)
            self.assertEqual(S(canonical_smiles(m)).constitution_key(), m.constitution_key(), s)


class TestLewis(unittest.TestCase):
    def census(self, s, unpaired=None):
        m = S(s)
        st = solve_lewis(m, unpaired=unpaired)
        self.assertTrue(st, s)
        for x in st:
            self.assertTrue(x.electron_census(m)["check"], s)
        return m, st

    def test_carbon_monoxide(self):
        m, st = self.census("[C-]#[O+]")
        self.assertEqual(len(st), 1)
        self.assertEqual(sorted(st[0].charges), [-1, 1])
        self.assertEqual(max(st[0].orders.values()), 3)

    def test_resonance_counts(self):
        self.assertEqual(len(self.census("[O-][O+]=O")[1]), 2)
        self.assertEqual(len(self.census("c1ccccc1")[1]), 2)
        self.assertEqual(len(self.census("C=C[CH2]")[1]), 2)
        self.assertEqual(len(self.census("c1ccc2ccccc2c1")[1]), 3)

    def test_radicals_and_spin(self):
        self.assertEqual(self.census("[CH3]")[1][0].unpaired, 1)
        self.assertEqual(self.census("[O][O]", unpaired=2)[1][0].unpaired, 2)
        # triplet O2: one unpaired electron on each oxygen, single bond
        m, st = self.census("[O][O]", unpaired=2)
        self.assertEqual(st[0].radicals, [1, 1])
        # singlet CH2 keeps a lone pair
        m, st = self.census("[CH2]", unpaired=0)
        self.assertEqual(st[0].lone_pairs[0], 1)

    def test_phenyl_no_ring_allene(self):
        m, st = self.census("[c]1ccccc1")
        self.assertEqual(len(st), 2)


class TestReactionSolver(unittest.TestCase):
    def solve(self, lhs, rhs):
        return solve_reaction([S(x) for x in lhs], [S(x) for x in rhs])

    def test_h_abstraction(self):
        sol = self.solve(["C", "[OH]"], ["[CH3]", "O"])
        self.assertEqual(sol.distance, 2)
        self.assertEqual(sol.reaction_class, "H-abstraction")
        self.assertEqual(sol.moved[0]["atoms"], "H")

    def test_recombination_and_scission(self):
        self.assertEqual(self.solve(["[CH3]", "[CH3]"], ["CC"]).reaction_class, "radical recombination")
        self.assertEqual(self.solve(["[CH2]CC"], ["C=C", "[CH3]"]).reaction_class, "beta-scission")
        self.assertEqual(self.solve(["[H]", "C=C"], ["[CH2]C"]).reaction_class,
                         "radical addition to multiple bond")

    def test_concerted_elimination(self):
        sol = self.solve(["CCO[O]"], ["C=C", "[O]O"])
        self.assertEqual(sol.distance, 3)
        self.assertEqual(sol.reaction_class, "concerted elimination")
        self.assertTrue(sol.arrows)

    def test_unbalanced(self):
        self.assertIsNone(sigma_distance([S("C")], [S("CC")]))

    def test_generate_completion(self):
        # CH4 + OH -> CH3 + ?  must propose water
        gens = generate_completions([S("C"), S("[OH]")], [S("[CH3]")], max_edits=2)
        self.assertIn(S("O").constitution_key(), gens)


class TestIsomers(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(enumerate_isomers({"C": 4, "H": 10})), 2)
        self.assertEqual(len(enumerate_isomers({"C": 7, "H": 16})), 9)
        self.assertEqual(len(enumerate_isomers({"C": 3, "H": 7})), 2)
        self.assertEqual(len(enumerate_isomers({"C": 4, "H": 10, "O": 1})), 7)


class TestNames(unittest.TestCase):
    def smi(self, name, formula):
        return {canonical_smiles(m) for m, _, _ in name_candidates(name, formula)}

    def test_alias_condensed_llnl(self):
        self.assertIn("C[CH]C", self.smi("IC3H7", {"C": 3, "H": 7}))
        self.assertIn("CCC=O", self.smi("CH3CH2CHO", {"C": 3, "H": 6, "O": 1}))
        self.assertIn("C[CH]CCCCC", self.smi("C7H15-2", {"C": 7, "H": 15}))


MECH = """ELEMENTS H O C AR END
SPECIES S1 S2 S3 S4 S5 S6 S7 S8 S9 AR END
THERMO ALL
300.000  1000.000  5000.000
{thermo}END
REACTIONS
S1+S2=S3+S4   1.0E13 0.0 0.0
S3+S2=S5+S1   1.0E13 0.0 0.0
S6+S4=S7+S8   1.0E13 0.0 0.0
S7+S7=S9      1.0E13 0.0 0.0
S6+S1=S7+S5   1.0E13 0.0 0.0
END
"""


def _card(name, comp):
    elems = "".join(f"{el:<2s}{n:>3d}" for el, n in comp)
    elems = elems.ljust(20)
    l1 = f"{name:<18s}      {elems}G   300.000  5000.000 1000.00      1"
    z = " 0.00000000E+00" * 5
    return (l1 + "\n" + z + "    2\n" + z + "    3\n" + " 0.00000000E+00" * 4 + "                   4\n")


class TestIdentifyBlind(unittest.TestCase):
    def test_small_mechanism_blind(self):
        # names carry no information: S1..S9
        comps = {"S1": [("H", 1)], "S2": [("O", 2)], "S3": [("O", 1), ("H", 1)], "S4": [("O", 1)],
                 "S5": [("H", 1), ("O", 1)], "S6": [("C", 1), ("H", 4)], "S7": [("C", 1), ("H", 3)],
                 "S8": [("O", 1), ("H", 1)], "S9": [("C", 2), ("H", 6)], "AR": [("AR", 1)]}
        thermo = "".join(_card(n, c) for n, c in comps.items())
        with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as fh:
            fh.write(MECH.format(thermo=thermo))
            path = fh.name
        mech = parse_chemkin(path)
        ident = Identifier(mech, {}, use_names=False, thermo=None).run()
        got = {n: st.chosen.smiles for n, st in ident.species.items() if st.chosen}
        self.assertEqual(got["S6"], "C")
        self.assertEqual(got["S7"], "[CH3]")
        self.assertEqual(got["S9"], "CC")
        os.unlink(path)


AUDIT_MECH = """ELEMENTS H O C END
SPECIES A B C D E F G END
THERMO ALL
300.000  1000.000  5000.000
{thermo}END
REACTIONS
D+B=E         1.0E13 0.0 0.0
E+A=D+C       1.0E13 0.0 0.0
F+G=E         1.0E13 0.0 0.0
END
"""


class TestAudit(unittest.TestCase):
    def test_flags_planted_error(self):
        # E is ethanol, but the "known" file wrongly says dimethyl ether
        comps = {"A": [("H", 1)], "B": [("O", 1), ("H", 1)], "C": [("H", 2), ("O", 1)],
                 "D": [("C", 2), ("H", 5)], "E": [("C", 2), ("H", 6), ("O", 1)],
                 "F": [("C", 1), ("H", 3)], "G": [("C", 1), ("H", 3), ("O", 1)]}
        thermo = "".join(_card(n, c) for n, c in comps.items())
        known = {"A": "[H]", "B": "[OH]", "C": "O", "D": "[CH2]C", "E": "COC",
                 "F": "[CH3]", "G": "[CH2]O"}
        with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as fh:
            fh.write(AUDIT_MECH.format(thermo=thermo))
            path = fh.name
        try:
            ident = Identifier(parse_chemkin(path), known, use_names=False, thermo=None)
            ident.setup()
            findings = ident.audit()
        finally:
            os.unlink(path)
        self.assertEqual([f["species"] for f in findings], ["E"])
        self.assertEqual(S(findings[0]["suggested"]).constitution_key(), S("CCO").constitution_key())
        self.assertEqual(findings[0]["given"], "COC")
        self.assertEqual(ident.audit_partial, [])


class TestSpin(unittest.TestCase):
    def test_ground_state_table_uses_hill_keys(self):
        from rxnsolve.elements import hill_formula
        from rxnsolve.identify import GROUND_STATE_MULTIPLICITY
        from rxnsolve.cli import _formula
        for key in GROUND_STATE_MULTIPLICITY:
            self.assertEqual(hill_formula(_formula(key)), key)
        self.assertEqual(GROUND_STATE_MULTIPLICITY["HN"], 3)   # NH
        self.assertEqual(GROUND_STATE_MULTIPLICITY["OS"], 3)   # SO
        self.assertEqual(GROUND_STATE_MULTIPLICITY["N"], 4)

    def test_spin_prefix_only_when_needed(self):
        cases = {"singlet[O]": "singlet[O]", "[O]": "[O]", "singlet[CH2]": "singlet[CH2]",
                 "triplet[CH2]": "[CH2]", "[O][O]": "[O][O]", "singlet[O][O]": "O=O",
                 "[N]": "[N]", "[CH3]": "[CH3]"}
        for given, written in cases.items():
            m = S(given)
            self.assertEqual(canonical_smiles(m), written, given)
            self.assertEqual(S(written).multiplicity, m.multiplicity, given)

    def test_spin_conservation(self):
        def allowed(lhs, rhs):
            return solve_reaction([S(x) for x in lhs], [S(x) for x in rhs]).spin_allowed
        self.assertFalse(allowed(["triplet[CH2]", "[H][H]"], ["C"]))
        self.assertTrue(allowed(["singlet[CH2]", "[H][H]"], ["C"]))
        self.assertTrue(allowed(["[O][O]", "[H]"], ["[O]O"]))
        self.assertTrue(allowed(["[CH3]", "[CH3]"], ["CC"]))
        self.assertFalse(allowed(["singlet[CH2]", "N#N"], ["triplet[CH2]", "N#N"]))

    def test_spin_labels_switch(self):
        comps = {"CH2": [("C", 1), ("H", 2)], "CH2(S)": [("C", 1), ("H", 2)],
                 "NH": [("N", 1), ("H", 1)]}
        mech_txt = ("ELEMENTS C H N END\nSPECIES CH2 CH2(S) NH END\nTHERMO ALL\n"
                    "300.000  1000.000  5000.000\n" + "".join(_card(n, c) for n, c in comps.items())
                    + "END\nREACTIONS\nEND\n")
        with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as fh:
            fh.write(mech_txt)
            path = fh.name
        try:
            mech = parse_chemkin(path)
            on = Identifier(mech, {}, use_names=False, thermo=None).run()
            off = Identifier(mech, {}, use_names=False, thermo=None, use_spin_labels=False).run()
        finally:
            os.unlink(path)
        self.assertEqual(on.species["CH2(S)"].chosen.smiles, "singlet[CH2]")
        self.assertEqual(on.species["CH2"].chosen.smiles, "[CH2]")
        self.assertEqual(on.species["NH"].chosen.mol.multiplicity, 3)   # Hund's rule table
        self.assertEqual(off.species["CH2(S)"].chosen.smiles, "[CH2]")  # label ignored


if __name__ == "__main__":
    unittest.main()
