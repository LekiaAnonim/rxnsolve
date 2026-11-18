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


class TestReportAndApp(unittest.TestCase):
    def _mech_files(self, folder):
        comps = {"S1": [("H", 1)], "S2": [("O", 2)], "S3": [("O", 1), ("H", 1)], "S4": [("O", 1)],
                 "S5": [("H", 1), ("O", 1)], "S6": [("C", 1), ("H", 4)], "S7": [("C", 1), ("H", 3)],
                 "S8": [("O", 1), ("H", 1)], "S9": [("C", 2), ("H", 6)], "AR": [("AR", 1)]}
        thermo = "".join(_card(n, c) for n, c in comps.items())
        chem = os.path.join(folder, "mechanism.txt")
        with open(chem, "w") as fh:
            fh.write(MECH.format(thermo=thermo))
        with open(os.path.join(folder, "import.sh"), "w") as fh:
            fh.write("python importChemkin.py --species mechanism.txt --reactions mechanism.txt "
                     "--thermo mechanism.txt --known SMILES.txt\n")
        with open(os.path.join(folder, "SMILES.txt"), "w") as fh:
            fh.write("S6\tC\nS7\t[CH3]\n")
        return chem

    def test_report_site(self):
        from rxnsolve.pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as d:
            chem = self._mech_files(d)
            out = os.path.join(d, "out")
            res = run_pipeline(chem, out_dir=out, use_names=False, use_thermo=False)
            for f in ("index.html", "reactions.html", "style.css", "SMILES.txt", "identification.json"):
                self.assertTrue(os.path.exists(os.path.join(out, f)), f)
            pages = os.listdir(os.path.join(out, "species"))
            self.assertEqual(len(pages), 10)
            with open(os.path.join(out, "species", next(p for p in pages if "S9" in p))) as fh:
                page = fh.read()
            for step in ("Formula", "Candidates", "Lewis structure", "Thermochemistry", "Reactions",
                         "Entropy and heat capacity", "Kinetics", "Decision"):
                self.assertIn(f"<h3>{step}</h3>", page)
            self.assertIn("<svg", page)
            self.assertEqual(res["counts"].get("identified"), 10)

    def test_app_discovers_models(self):
        from rxnsolve.webapp import discover
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "ModelA"))
            self._mech_files(os.path.join(d, "ModelA"))
            os.makedirs(os.path.join(d, "Empty"))
            models = discover(d)
            self.assertEqual(list(models), ["ModelA"])
            self.assertTrue(models["ModelA"]["known"].endswith("SMILES.txt"))
            self.assertTrue(models["ModelA"]["chem"].endswith("mechanism.txt"))


class TestThermoFallback(unittest.TestCase):
    """What happens to thermochemistry when the thermo library is missing."""

    def setUp(self):
        import rxnsolve.thermo as T
        self.T = T
        self._load_library = T.load_library
        self._ga_file = T.GA_FILE
        T._DEFAULT.clear()

    def tearDown(self):
        self.T.load_library = self._load_library
        self.T.GroupAdditivity.load.__func__.__defaults__ = (self._ga_file,)
        self.T._DEFAULT.clear()

    def test_group_additivity_without_library(self):
        T = self.T
        T.load_library = lambda path=None: None
        model = T.default_model()
        self.assertIsNotNone(model, "saved group values should keep group additivity working")
        self.assertEqual(model.n_records, 0)
        benzene, hexadiyne = S("c1ccccc1"), S("CC#CC#CC")
        ll_b, dh_b, note = model.log_likelihood(benzene, benzene.constitution_key(), 19.8, 64.4)
        self.assertIn("group additivity", note)
        self.assertLess(dh_b, 10.0)                    # benzene: real +19.8 kcal/mol, GA within 10
        ll_h, dh_h, _ = model.log_likelihood(hexadiyne, hexadiyne.constitution_key(), 19.8, 64.4)
        self.assertGreater(ll_b, ll_h)                 # still separates aromatic from open chain

    def test_identifier_reports_ga_only(self):
        T = self.T
        T.load_library = lambda path=None: None
        mech = parse_chemkin_text("H2O", [("H", 2), ("O", 1)])
        ident = Identifier(mech, {}, use_names=False)
        self.assertIn("group additivity only", ident.thermo_info)

    def test_nothing_available(self):
        T = self.T
        T.load_library = lambda path=None: None
        T.GroupAdditivity.load.__func__.__defaults__ = ("/nonexistent/group_additivity.json",)
        self.assertIsNone(T.default_model())
        mech = parse_chemkin_text("H2O", [("H", 2), ("O", 1)])
        ident = Identifier(mech, {}, use_names=False)
        self.assertIn("not found", ident.thermo_info)


KIN_MECH = """ELEMENTS H C END
SPECIES PENT H H2 SIB X END
THERMO ALL
300.000  1000.000  5000.000
{thermo}END
REACTIONS
PENT+H=H2+SIB   1.00E+06  2.50  4500.0
PENT+H=H2+X     5.00E+05  2.50  4400.0
END
"""


class TestKinetics(unittest.TestCase):
    def _run(self, kinetics=True):
        comps = {"PENT": [("C", 5), ("H", 12)], "H": [("H", 1)], "H2": [("H", 2)],
                 "SIB": [("C", 5), ("H", 11)], "X": [("C", 5), ("H", 11)]}
        thermo = "".join(_card(n, c) for n, c in comps.items())
        with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as fh:
            fh.write(KIN_MECH.format(thermo=thermo))
            path = fh.name
        try:
            mech = parse_chemkin(path)
        finally:
            os.unlink(path)
        if not kinetics:
            for r in mech.reactions:
                r.A = None
        known = {"PENT": "CCCCC", "H": "[H]", "H2": "[H][H]", "SIB": "C[CH]CCC"}
        return Identifier(mech, known, use_names=False, thermo=None).run()

    def test_equivalent_h(self):
        from rxnsolve.identify import equivalent_h
        self.assertEqual(equivalent_h(S("CCCCC"), S("[CH2]CCCC")), 6)
        self.assertEqual(equivalent_h(S("CCCCC"), S("C[CH]CCC")), 4)
        self.assertEqual(equivalent_h(S("CCCCC"), S("CC[CH]CC")), 2)
        self.assertEqual(equivalent_h(S("CCCCC"), S("[CH2]C(C)CC")), 0)

    def test_competing_channels_decide(self):
        # X forms with a smaller A (fewer equivalent H) and a slightly lower Ea than
        # 2-pentyl: 3-pentyl agrees with both signals, 1-pentyl contradicts both
        ident = self._run()
        st = ident.species["X"]
        cands = {c.smiles: c for c in st.candidates.values()}
        self.assertEqual(cands["CC[CH]CC"].log_kin, 0.0)
        self.assertLess(cands["[CH2]CCCC"].log_kin, -4.0)
        self.assertEqual(st.status, "identified")
        self.assertEqual(st.chosen.smiles, "CC[CH]CC")
        self.assertIn("kinetics", st.method)

    def test_without_rates_it_stays_open(self):
        ident = self._run(kinetics=False)
        self.assertNotEqual(ident.species["X"].status, "identified")


class TestEntropyHeatCapacity(unittest.TestCase):
    def test_ring_beats_chain_on_entropy_and_cp(self):
        from rxnsolve.thermo import EntropyCpModel
        m = EntropyCpModel.load()
        self.assertIsNotNone(m, "saved entropy/Cp group values should ship with the package")
        # oxirane's own NASA values: S298 58.0 cal/mol/K, Cp 11.4 / 17.9 / 24.6 / 27.5 / 31.8
        s_mech, cp_mech = 58.0, [11.35, 17.90, 24.55, 27.49, 31.79]
        ll_ring, z_ring, note = m.log_likelihood(S("C1CO1"), s_mech, cp_mech, 1)
        ll_ald, z_ald, _ = m.log_likelihood(S("CC=O"), s_mech, cp_mech, 1)
        self.assertIn("S298", note)
        self.assertLess(z_ring, z_ald)
        self.assertGreater(ll_ring, ll_ald + 2.0)

    def test_spin_degeneracy_adds_r_ln_multiplicity(self):
        import math
        from rxnsolve.thermo import EntropyCpModel, R_CAL
        m = EntropyCpModel.load()
        s1, _ = m.predict(S("[CH2]"), 1)
        s3, _ = m.predict(S("[CH2]"), 3)
        self.assertAlmostEqual(s3 - s1, R_CAL * math.log(3), places=6)


def parse_chemkin_text(name, comp):
    txt = (f"ELEMENTS H O C END\nSPECIES {name} END\nTHERMO ALL\n300.000  1000.000  5000.000\n"
           + _card(name, comp) + "END\nREACTIONS\nEND\n")
    with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as fh:
        fh.write(txt)
        path = fh.name
    try:
        return parse_chemkin(path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
