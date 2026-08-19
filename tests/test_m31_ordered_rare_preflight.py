import csv
import gzip
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(os.environ.get("M31_PREFLIGHT_SCRIPT_PATH", Path(__file__).parents[1] / "bin" / "m31_ordered_rare_preflight.py"))
SPEC = importlib.util.spec_from_file_location("m31_ordered_rare_preflight", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class M31OrderedRarePreflightTest(unittest.TestCase):
    def test_minor_presence_known_answers_cover_both_minor_codes(self):
        self.assertEqual(MODULE.minor_presence(0, 0), 1)
        self.assertEqual(MODULE.minor_presence(1, 0), 0)
        self.assertEqual(MODULE.minor_presence(0, 1), 0)
        self.assertEqual(MODULE.minor_presence(1, 1), 1)
        self.assertIsNone(MODULE.minor_presence(None, 0))
        self.assertEqual(MODULE.known_answers()["state0_minor0"], 1)

    def test_m29_legacy_sum_is_wrong_when_minor_code_is_zero(self):
        h0, h1, minor_code = 0, 0, 0
        legacy = h0 + h1
        corrected = MODULE.minor_presence(h0, minor_code) + MODULE.minor_presence(h1, minor_code)
        self.assertEqual(legacy, 0)
        self.assertEqual(corrected, 2)

    def test_materializer_preserves_order_key_and_missingness(self):
        selected = [
            MODULE.SelectedSite(0, "22", 100, 0, 2, 100, 0.02, 2),
            MODULE.SelectedSite(1, "22", 200, 1, 2, 100, 0.02, 2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "target.tsv.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("chrom", "position", "minor_code", "T001_h0", "T001_h1", "T000_h0", "T000_h1"))
                writer.writerow(("22", 100, 0, 0, 1, 0, 0))
                writer.writerow(("22", 200, 1, 1, ".", 0, 1))
            audit = MODULE.materialize_target(source, selected, 17, root)
            self.assertEqual(audit["samples"], 2)
            self.assertEqual(audit["target_rows"], 4)
            self.assertEqual(audit["minor_code_zero_sites"], 1)
            self.assertTrue(audit["m29_semantic_bug_present"])
            with gzip.open(root / "m31_ordered_rare.target.tsv.gz", "rt", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([(row["position"], row["sample_id"]) for row in rows], [("100", "T000"), ("100", "T001"), ("200", "T000"), ("200", "T001")])
            first = rows[0]
            self.assertEqual((first["h0_minor_presence"], first["h1_minor_presence"], first["minor_dosage"]), ("1", "1", "2"))
            self.assertEqual(rows[-1]["missing_haplotypes"], "1")

    def test_contract_is_smoke_only_and_prohibits_target_selection(self):
        path = Path(__file__).parents[1] / "conf" / "m31_ordered_rare_preflight_preregistration.json"
        contract = json.loads(path.read_text())
        self.assertEqual(contract["status"], "PREFLIGHT_ONLY_NOT_SCIENTIFIC_EVIDENCE")
        self.assertEqual(contract["rare_universe"]["selector"], "FREQ_only")
        self.assertIn("TARGET", contract["rare_universe"]["prohibited_selectors"])
        self.assertIn("truth", contract["rare_universe"]["prohibited_selectors"])
        self.assertFalse(contract["materialization"]["model_frozen"])
        self.assertFalse(contract["materialization"]["metrics_frozen"])
        self.assertEqual(contract["materialization"]["sample_identity_key"], ["root_seed", "sample_id"])
        self.assertEqual(contract["materialization"]["row_primary_key"], ["root_seed", "sample_id", "locus_index"])
        self.assertEqual({root["root_seed"] for root in contract["roots"].values()}, {20260817, 20260818})

    def test_code_provenance_validation_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "git_commit"):
            MODULE.validate_git_commit("unknown")
        self.assertEqual(MODULE.validate_git_commit("a" * 40), "a" * 40)
        self.assertIsNone(MODULE.validate_git_commit(None))
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "script.py"
            script.write_text("print('fixed')\n")
            observed = MODULE.sha256_file(script)
            self.assertEqual(MODULE.authenticate_script(script, observed), observed)
            with self.assertRaisesRegex(ValueError, "script sha256 mismatch"):
                MODULE.authenticate_script(script, "0" * 64)

    def test_invalid_allele_states_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "0/1/missing"):
            MODULE.parse_observed_state("2")
        with self.assertRaisesRegex(ValueError, "minor_code"):
            MODULE.minor_presence(0, 2)


if __name__ == "__main__":
    unittest.main()
