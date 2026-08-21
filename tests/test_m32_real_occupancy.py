import gzip
import json
import tempfile
import unittest
from pathlib import Path

import sys
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m32_prepare_coordinates as prepare  # noqa: E402
import m32_real_occupancy as real  # noqa: E402


class M32RealOccupancyTest(unittest.TestCase):
    def test_genetic_map_interpolation_matches_known_answers(self):
        genetic_map = prepare.GeneticMap([100, 200, 300], [0.0, 1.0, 1.0])
        self.assertEqual(genetic_map.cm_at(100), 0.0)
        self.assertEqual(genetic_map.cm_at(150), 0.5)
        self.assertEqual(genetic_map.cm_at(250), 1.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            genetic_map.cm_at(99)

    def test_coordinate_validation_allows_cm_ties_but_not_bp_ties(self):
        prepare.validate_rows([("chr22", 100, 0.1, "a"), ("chr22", 101, 0.1, "b")], "fixture")
        with self.assertRaisesRegex(ValueError, "bp"):
            prepare.validate_rows([("chr22", 100, 0.1, "a"), ("chr22", 100, 0.1, "b")], "fixture")

    def test_load_coordinates_rejects_extra_or_reordered_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coords.tsv"
            path.write_text("chrom\tcm\tbp\tlocus_id\nchr22\t0.1\t100\ta\n")
            with self.assertRaisesRegex(ValueError, "header"):
                real.load_coordinates(path)

    def test_contract_is_truth_free_and_inputs_are_sha_locked(self):
        contract = json.loads((ROOT / "conf" / "m32_real_occupancy_preregistration.json").read_text())
        self.assertFalse(contract["select_radius"])
        self.assertFalse(contract["scientific_run_authorized"])
        self.assertEqual(set(contract["roots"]), {"root17", "root18"})
        self.assertEqual(len(contract["genetic_map"]["sha256"]), 64)
        for root in contract["roots"].values():
            self.assertEqual(root["role"], "consumed_technical_known_answer_only")
            self.assertEqual(len(root["rare_sites_sha256"]), 64)
            self.assertEqual(len(root["flare_grid_sha256"]), 64)

    def test_nextflow_contract_has_two_processes_no_truth_and_bounded_resources(self):
        module = (ROOT / "modules" / "32_REAL_OCCUPANCY.nf").read_text().lower()
        workflow = (ROOT / "workflows" / "m32_real_occupancy.nf").read_text().lower()
        config = (ROOT / "conf" / "m32_real_occupancy.config").read_text()
        self.assertIn("process m32_materialize_coordinates", module)
        self.assertIn("process m32_real_occupancy_screen", module)
        for prohibited in ("truth", "target_genotype", "posterior", "boundary", "king"):
            self.assertNotIn(prohibited, module)
            self.assertNotIn(prohibited, workflow)
        self.assertIn("overwrite: false", module)
        self.assertIn("m32_occ_cpus = 1", config)
        self.assertIn("m32_occ_memory = '2 GB'", config)
        self.assertIn("maxForks = 2", config)


if __name__ == "__main__":
    unittest.main()
