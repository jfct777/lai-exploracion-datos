import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
SPEC = importlib.util.spec_from_file_location(
    "m33_safe_bridge_technical_under_test", ROOT / "bin/m33_safe_bridge_technical_kat.py"
)
M33 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M33)


def write_flare(path: Path, gt: str, hard1: int, hard2: int,
                anp1: str = "0.8,0.1,0.1", anp2: str = "0.1,0.2,0.7") -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##ANCESTRY=<AFR=0,EUR=1,ASIA=2>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT0\n")
        handle.write(
            "22\t100\tm28s1\tA\tC\t.\tPASS\tTSID=1\tGT:AN1:AN2:ANP1:ANP2\t"
            f"{gt}:{hard1}:{hard2}:{anp1}:{anp2}\n"
        )


class TechnicalKatTests(unittest.TestCase):
    def test_contract_remains_non_consumable_and_truth_free(self):
        contract = json.loads((ROOT / "conf/m33_safe_bridge_technical_kat_contract.json").read_text())
        M33.validate_contract(contract)
        self.assertTrue(all(value is False for value in contract["gates"].values()))

    def test_authorization_has_exact_closed_inventory(self):
        authorization = json.loads(
            (ROOT / "conf/m33_safe_bridge_technical_kat_authorization.json").read_text()
        )
        self.assertEqual(set(authorization["roots"]), {"root17", "root18"})
        for root in authorization["roots"].values():
            self.assertEqual(set(root["assets"]), M33.INPUT_NAMES)
            for asset in root["assets"].values():
                self.assertRegex(asset["generation"], r"^[0-9]+$")
                self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")

    def test_f0_projection_is_invariant_to_gt_and_hard_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.vcf.gz"
            second = Path(directory) / "second.vcf.gz"
            write_flare(first, "0|0", 0, 2)
            write_flare(second, "1|1", 2, 0)
            loci_a, values_a = M33.load_f0_projection(first, ("T0",))
            loci_b, values_b = M33.load_f0_projection(second, ("T0",))
            self.assertEqual(loci_a, loci_b)
            np.testing.assert_array_equal(values_a, values_b)

    def test_f0_projection_changes_when_anp_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.vcf.gz"
            second = Path(directory) / "second.vcf.gz"
            write_flare(first, "0|0", 0, 2)
            write_flare(second, "0|0", 0, 2, anp1="0.1,0.8,0.1")
            _loci_a, values_a = M33.load_f0_projection(first, ("T0",))
            _loci_b, values_b = M33.load_f0_projection(second, ("T0",))
            self.assertFalse(np.array_equal(values_a, values_b))

    def test_runner_has_no_oracle_or_truth_arguments(self):
        source = (ROOT / "bin/m33_safe_bridge_technical_kat.py").read_text()
        parser = source.split("def parse_args", 1)[1]
        for forbidden in ("a0-receipt", "i0-receipt", "truth", "READY"):
            self.assertNotIn(forbidden, parser)

    def test_outputs_use_only_technical_test_namespace(self):
        self.assertEqual(len(M33.SCHEMAS), 7)
        self.assertTrue(all(name.startswith("technical_kat_") for name in M33.SCHEMAS))
        self.assertTrue(all(schema.startswith("tests_") for schema in M33.SCHEMAS.values()))

    def test_ref_label_sham_is_add_only_and_keeps_legacy_schemas_exact(self):
        legacy = {
            "technical_kat_selected_loci_incremental.npz":
                "tests_m33_safe_bridge_technical_kat_selected_loci_incremental_v1",
            "technical_kat_target_rare_diploid_incremental.npz":
                "tests_m33_safe_bridge_technical_kat_target_rare_diploid_incremental_v1",
            "technical_kat_reference_rare_summary_incremental.npz":
                "tests_m33_safe_bridge_technical_kat_reference_rare_summary_incremental_v1",
            "technical_kat_flare_f0_sanitized.npz":
                "tests_m33_safe_bridge_technical_kat_flare_f0_sanitized_v1",
        }
        self.assertEqual({name: M33.SCHEMAS[name] for name in legacy}, legacy)
        expected_shams = {
            f"technical_kat_reference_rare_summary_ref_label_sham_{seed}.npz"
            for seed in M33.bridge_core.REF_LABEL_SHAM_SEEDS
        }
        self.assertEqual(set(M33.SCHEMAS) - set(legacy), expected_shams)
        self.assertEqual(
            {M33.SCHEMAS[name] for name in expected_shams},
            {"tests_m33_safe_bridge_technical_kat_reference_rare_summary_ref_label_sham_v1"},
        )

    def test_ref_label_sham_contract_freezes_real_technical_firewall(self):
        contract = json.loads((ROOT / "conf/m33_safe_bridge_technical_kat_contract.json").read_text())
        M33.validate_contract(contract)
        sham = contract["ref_label_sham_technical_integration"]
        self.assertEqual(sham["seeds"], list(M33.bridge_core.REF_LABEL_SHAM_SEEDS))
        self.assertEqual(sham["expected_people_by_ancestry"],
                         {"AFR": 30, "EUR": 30, "ASIA": 30})
        self.assertFalse(sham["consumable"])
        self.assertFalse(sham["truth_read"])
        self.assertFalse(sham["materialize_or_training"])

    def test_runner_authenticates_diploid_ref_axis_before_sham_and_exports_no_ids(self):
        source = (ROOT / "bin/m33_safe_bridge_technical_kat.py").read_text()
        self.assertIn("map_ref_people_to_authenticated_samples", source)
        self.assertIn("set(mapped) == set(ref_nodes) == set(panel_labels)", source)
        self.assertIn("panel_labels[person] == label", source)
        self.assertIn("expected_people_by_ancestry={name: 30 for name in ANCESTRIES}", source)
        self.assertIn('"raw_identifiers_exported": False', source)
        self.assertIn('"ref_label_sham_real_reference_summary_unchanged": True', source)

    def test_private_ref_people_map_to_pseudonyms_only_through_node_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            pools = Path(directory) / "pools.tsv"
            pools.write_text(
                "role\tancestry\tindividual_id\tnode_id\tnode_identity_sha256\n"
                "REF_LAI\tAFR\tprivate_a\t10\tx\n"
                "REF_LAI\tAFR\tprivate_a\t11\ty\n"
                "REF_LAI\tEUR\tprivate_b\t20\tz\n"
                "REF_LAI\tEUR\tprivate_b\t21\tw\n",
                encoding="utf-8",
            )
            observed = M33.map_ref_people_to_authenticated_samples(
                pools,
                ("private_a", "private_b"),
                ("AFR", "EUR"),
                {"REF_EUR_000": (21, 20), "REF_AFR_000": (10, 11)},
                {"REF_AFR_000": "AFR", "REF_EUR_000": "EUR"},
            )
            self.assertEqual(observed, ("REF_AFR_000", "REF_EUR_000"))

    def test_ref_identity_mapping_rejects_wrong_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            pools = Path(directory) / "pools.tsv"
            pools.write_text(
                "role\tancestry\tindividual_id\tnode_id\tnode_identity_sha256\n"
                "REF_LAI\tAFR\tprivate_a\t10\tx\n"
                "REF_LAI\tAFR\tprivate_a\t11\ty\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "node pair or ancestry"):
                M33.map_ref_people_to_authenticated_samples(
                    pools, ("private_a",), ("AFR",),
                    {"REF_EUR_000": (10, 11)}, {"REF_EUR_000": "EUR"},
                )

    def test_runner_requires_precreated_empty_output_directory(self):
        source = (ROOT / "bin/m33_safe_bridge_technical_kat.py").read_text()
        self.assertIn("existing empty isolated directory", source)
        self.assertNotIn("args.output_dir.mkdir", source)
        self.assertNotIn("os.chmod(args.output_dir", source)

    def test_runner_measures_and_enforces_peak_rss(self):
        source = (ROOT / "bin/m33_safe_bridge_technical_kat.py").read_text()
        self.assertIn("resource.getrusage(resource.RUSAGE_SELF).ru_maxrss", source)
        self.assertIn("peak_rss_gib <= stop_rss_gib", source)
        self.assertIn('"rss_gate_passed": True', source)

    def test_input_isolation_claim_matches_nextflow_staging(self):
        contract = json.loads((ROOT / "conf/m33_safe_bridge_technical_kat_contract.json").read_text())
        isolation = contract["isolation"]
        self.assertFalse(isolation["physical_bind_read_only"])
        self.assertTrue(isolation["effective_read_only_probes"])
        self.assertIn("staged_copy_effectively_read_only", isolation["inputs"])


if __name__ == "__main__":
    unittest.main()
