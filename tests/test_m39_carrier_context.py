"""Synthetic known-answer tests; no production genotypes, truth or training."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m39_carrier_context as B


def fixture(root: Path) -> tuple[Path, dict]:
    refs = [f"private_ref_{a}_{i}" for a in B.ANCESTRIES for i in range(2)]
    targets = ["M34_R0_FIT_0000", "M34_R0_FIT_0001"]
    split = root / "split.tsv"
    text = "sample_id\tancestry\tcanonical_population\tatomic_unit_id\trole\n"
    for a, ancestry in enumerate(B.ANCESTRIES):
        for sample in refs[2 * a:2 * a + 2]:
            text += f"{sample}\t{ancestry}\tpop_{sample}\tunit_{sample}\tREF_TRAIN\n"
    text += "private_test\tAFR\tpop_test\tunit_test\tSOURCE_TEST\n"
    text += "private_valid\tEUR\tpop_valid\tunit_valid\tSOURCE_VALID\n"
    split.write_text(text)
    map_path = root / "chr22.map"
    map_path.write_text("22\t100\t0.0\n22\t1000\t0.9\n")
    ref_rows = {
        100: ["0|1"] * 6,
        200: ["0|1", ".|1", "0|0", "0|0", "0|0", "0|0"],
        300: ["0|0", "1|1", "0|1", "0|1", ".|0", "1|1"],
        500: ["0|1"] * 6,
        600: ["0|0", "1|1", "1|1", "1|1", "1|1", "1|1"],
        700: ["0|0"] * 6,
        900: ["0|1"] * 6,
    }
    query_rows = {p: ["0|1", "1|0"] for p in ref_rows}
    query_rows[200] = ["0|1", ".|1"]
    query_rows[600] = ["0|1", "0|0"]
    query_rows[300] = [".|1", "1|0"]

    def write_vcf(path, samples, rows, role):
        metadata = ["##fileformat=VCFv4.2", "##m34_bridge_scope=exploratory_only",
                    f"##m34_bridge_vcf_role={role}", "##m34_reference_and_frequency_role=REF_TRAIN",
                    "##m34_mosaic_donor_role_upstream=SOURCE_VALID",
                    "##m34_non_reference_panel_genotypes_opened=false",
                    "##m34_source_test_mosaic_donors_upstream=false"]
        lines = metadata + ["#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
        for pos, values in rows.items():
            lines.append(f"22\t{pos}\t.\tA\tG\t.\tPASS\t.\tDP:GT\t" + "\t".join("9:" + v for v in values))
        with gzip.open(path, "wt") as handle:
            handle.write("\n".join(lines) + "\n")

    ref_vcf, query_vcf = root / "ref.vcf.gz", root / "query.vcf.gz"
    write_vcf(ref_vcf, refs, ref_rows, "REFERENCE_REF_TRAIN")
    write_vcf(query_vcf, targets, query_rows, "TARGET_SOURCE_VALID_MOSAICS")
    pos = np.asarray([200, 600], dtype=np.int64)
    ids = np.asarray([B._locus_id("22", int(p), "A", "G") for p in pos], dtype=np.uint64)
    selected = {"locus_id": ids, "chrom": np.asarray([22, 22], dtype=np.uint8), "pos": pos,
                "ref": np.asarray([b"A", b"A"]), "alt": np.asarray([b"G", b"G"]),
                "cM": np.asarray([0.1, 0.5], dtype=np.float64)}
    target = {"sample_key_sha256": np.asarray([B.sample_key(s) for s in targets], dtype="S64"),
              "locus_id": ids, "minor_dosage": np.asarray([[1, 1], [0, 2]], dtype=np.int8),
              "observed_mask": np.asarray([[1, 1], [0, 1]], dtype=np.uint8)}
    ac = np.asarray([[2, 2], [0, 0], [0, 0]], dtype=np.uint16)
    an = np.asarray([[3, 4], [4, 4], [4, 4]], dtype=np.uint16)
    reference = {"ancestry": np.asarray(B.ANCESTRIES, dtype="S4"), "locus_id": ids,
                 "minor_ac": ac, "callable_an": an, "minor_af": ac / an,
                 "observed_mask": np.ones((3, 2), dtype=np.uint8),
                 "no_support": (ac == 0).astype(np.uint8)}
    paths = {"reference_vcf": ref_vcf, "target_vcf": query_vcf, "split_manifest": split,
             "genetic_map": map_path}
    for name, arrays in (("selected_loci", selected), ("target_rare", target),
                         ("reference_summary", reference)):
        paths[name] = root / f"{name}.npz"
        B.write_deterministic_npz(paths[name], arrays)
    config = {"schema_version": B.SCHEMA, "scope": "technical_only_chr22_R0_FIT",
              "inputs": {name: {"path": p.name, "uri": "gs://synthetic-fixture/" + p.name,
                                "sha256": B.sha256_file(p)} for name, p in paths.items()},
              "parameters": {"chromosome": "22", "expected_reference_people": 6,
                             "expected_target_people": 2, "expected_full_loci": 7,
                             "expected_excluded_loci": 2, "loci": 2, "K": 3,
                             "radii_cm": [0.05, 0.2, 0.5], "common_maf_min": 0.01,
                             "anchor_selection": "evenly_spaced_selected_locus_indices",
                             "frequency_role": "REF_TRAIN",
                             "minor_orientation": "REF_TRAIN_ALT_count_le_REF_count_tie_ALT",
                             "rare_phase": "discarded_diploid_dosage_only", "target_partition": "FIT"},
              "claims": {"allowed": ["technical_materialization"], "not_evaluated": ["LAI"]}}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, config


class CarrierBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config_path, self.config = fixture(self.root)
        self.config, self.paths = B.load_config(self.config_path, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def data(self):
        return B.extract_inputs(self.paths, self.config)

    def rewrite_vcf(self, name, old, new):
        path = self.paths[name]
        with gzip.open(path, "rt") as handle:
            text = handle.read()
        self.assertIn(old, text)
        with gzip.open(path, "wt") as handle:
            handle.write(text.replace(old, new))
        self.config["inputs"][name]["sha256"] = B.sha256_file(path)

    def test_reconciliation_partial_calls_minor_ref_and_full_exclusion(self):
        data = self.data()
        self.assertEqual(data["counts"]["common_loci"], 4)
        self.assertEqual(data["counts"]["excluded_selected_loci"], 2)
        self.assertEqual(data["counts"]["reference_partial_rare_genotypes"], 1)
        self.assertEqual(data["counts"]["target_partial_rare_genotypes"], 1)
        np.testing.assert_array_equal(data["minor_is_alt"], [True, False])
        np.testing.assert_array_equal(data["query_dosage"], [[1, 1], [0, 2]])
        np.testing.assert_array_equal(data["query_observed"], [[1, 1], [0, 1]])
        self.assertEqual(data["ref_dosage"][1, 0], 2)
        self.assertFalse(data["ref_observed"][0, 1])
        self.assertEqual(data["ref_dosage"][0, 1], 0)
        self.assertFalse(np.any(np.isin(data["common_cm"], [0.1, 0.5])))

    def test_axes_features_and_diploid_attachment_not_rare_phase(self):
        data = self.data()
        arrays, _ = B.materialize_radius(data, np.arange(2), 3, 0.5)
        self.assertEqual(arrays["common_context"].shape, (2, 2, 2, 3, 3, 4))
        for q, j, h, a, k in np.argwhere(arrays["candidate_mask"]):
            person = arrays["candidate_ref_index"][q, j, h, a, k]
            self.assertEqual(data["reference_ancestry"][person], a)
            self.assertEqual(arrays["ref_dosage"][q, j, h, a, k], data["ref_dosage"][j, person])
            self.assertEqual(arrays["rare_ref_observed"][q, j, h, a, k], data["ref_observed"][j, person])
        self.assertEqual(arrays["rare_semantics"][0].decode(), B.RARE_SEMANTICS)

    def test_missing_joint_calls_and_empty_windows_never_choose_arbitrary_neighbours(self):
        data = self.data()
        empty, stats = B.materialize_radius(data, np.arange(2), 8, 0.05)
        self.assertFalse(empty["candidate_mask"].any())
        self.assertTrue((empty["candidate_ref_index"] == -1).all())
        self.assertEqual(stats["zero_common_window_anchors"], 2)
        data["common_target"][:] = -1
        unsupported, _ = B.materialize_radius(data, np.arange(2), 8, 0.5)
        self.assertFalse(unsupported["candidate_mask"].any())
        self.assertTrue(np.isfinite(unsupported["common_context"]).all())

    def test_rare_perturbation_cannot_change_common_retrieval(self):
        data = self.data()
        baseline, _ = B.materialize_radius(data, np.arange(2), 3, 0.5)
        data["ref_dosage"] = 2 - data["ref_dosage"]
        data["query_dosage"] = 2 - data["query_dosage"]
        data["ref_observed"][:] = True
        data["query_observed"][:] = True
        changed, _ = B.materialize_radius(data, np.arange(2), 3, 0.5)
        for key in ("common_context", "candidate_mask", "candidate_ref_index", "candidate_hap", "locus_id"):
            np.testing.assert_array_equal(baseline[key], changed[key])
        self.assertFalse(np.array_equal(baseline["ref_dosage"], changed["ref_dosage"]))

    def test_pooled_summary_counts_individual_once_not_selected_haplotypes(self):
        data = self.data()
        summary = B.pooled_summary(data, np.arange(2))
        np.testing.assert_array_equal(summary[0, 0, 0], [0, 1, 0, 0.5])
        np.testing.assert_array_equal(summary[0, 1, 0], [0.5, 0, 0.5, 1])
        np.testing.assert_array_equal(summary[0], summary[1])
        small, _ = B.materialize_radius(data, np.arange(2), 1, 0.2)
        big, _ = B.materialize_radius(data, np.arange(2), 8, 0.5)
        np.testing.assert_array_equal(small["pooled_summary"], big["pooled_summary"])

    def test_ties_invariant_to_reference_row_permutation(self):
        data = self.data()
        original, _ = B.materialize_radius(data, np.arange(2), 3, 0.5)
        permutation = np.asarray([5, 0, 3, 1, 4, 2])
        for key in ("reference_ancestry", "reference_sample_key_sha256"):
            data[key] = data[key][permutation]
        data["common_ref"] = data["common_ref"][:, permutation]
        for key in ("ref_dosage", "ref_observed"):
            data[key] = data[key][:, permutation]
        reordered, _ = B.materialize_radius(data, np.arange(2), 3, 0.5)
        mask = original["candidate_mask"].astype(bool)
        np.testing.assert_array_equal(original["candidate_ref_index"][mask],
                                      permutation[reordered["candidate_ref_index"][mask]])
        for key in ("candidate_hap", "common_context", "ref_dosage", "rare_ref_observed"):
            np.testing.assert_array_equal(original[key], reordered[key])

    def test_m34_parser_rejects_unphased_and_multiallelic_gt(self):
        for invalid in ("0/1", "0|2", "0"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                B._gt_from_sample_field("DP:GT", "9:" + invalid, label="fixture", sample="synthetic", line_number=1)
        self.rewrite_vcf("reference_vcf", "9:0|1", "9:0/1")
        with self.assertRaisesRegex(ValueError, "unphased"):
            self.data()

    def test_validation_or_test_targets_rejected_before_genotype_parse(self):
        self.rewrite_vcf("target_vcf", "M34_R0_FIT_", "M34_R0_VALID_")
        with patch.object(B, "parse_states", side_effect=AssertionError("genotypes opened")):
            with self.assertRaisesRegex(ValueError, "only R0 FIT"):
                self.data()

    def test_source_test_reference_rejected_before_genotype_parse(self):
        self.rewrite_vcf("reference_vcf", "private_ref_AFR_0", "private_test")
        with patch.object(B, "parse_states", side_effect=AssertionError("genotypes opened")):
            with self.assertRaisesRegex(ValueError, "REF_TRAIN header"):
                self.data()

    def test_hash_failure_and_undeclared_truth_input_rejected(self):
        broken = copy.deepcopy(self.config)
        broken["inputs"]["reference_vcf"]["sha256"] = "0" * 64
        self.config_path.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            B.load_config(self.config_path, self.root)
        broken["inputs"]["truth"] = broken["inputs"]["target_rare"]
        self.config_path.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "seven bridge inputs"):
            B.load_config(self.config_path, self.root)

    def test_mismatched_vcf_axis_and_wrong_canonical_dosage_fail_closed(self):
        self.rewrite_vcf("target_vcf", "22\t200\t.\tA\tG", "22\t200\t.\tA\tT")
        with self.assertRaisesRegex(ValueError, "allele axes"):
            self.data()
        self.rewrite_vcf("target_vcf", "22\t200\t.\tA\tT", "22\t200\t.\tA\tG")
        self.rewrite_vcf("target_vcf", "9:0|1\t9:.|1", "9:1|1\t9:.|1")
        with self.assertRaisesRegex(ValueError, "TARGET dosage/mask"):
            self.data()

    def test_atomic_publication_private_keys_only_npz_deterministic_nooverwrite(self):
        out = self.root / "output"
        receipt = B.run_bridge(self.config_path, self.root, out)
        self.assertEqual(receipt["decision"], "PASS_TECHNICAL_BRIDGE_ONLY")
        public = (out / "receipt.json").read_text()
        for forbidden in ("private_ref_", "M34_R0_FIT_000", "private_test", B.sample_key("private_ref_AFR_0").decode()):
            self.assertNotIn(forbidden, public)
        for profile in receipt["profiles"]:
            path = out / profile["path"]
            self.assertEqual(B.sha256_file(path), profile["sha256"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        second = B.run_bridge(self.config_path, self.root, self.root / "second")
        self.assertEqual([x["sha256"] for x in receipt["profiles"]], [x["sha256"] for x in second["profiles"]])
        with self.assertRaisesRegex(ValueError, "already exists"):
            B.run_bridge(self.config_path, self.root, out)

    def test_failed_write_leaves_no_published_partial_output(self):
        out = self.root / "failed"
        with patch.object(B, "materialize_radius", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                B.run_bridge(self.config_path, self.root, out)
        self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob(".m39-stage-*")))
        self.assertFalse((self.root / ".failed.m39.lock").exists())

    def test_anchor_grid_is_equally_spaced_and_target_independent(self):
        np.testing.assert_array_equal(B.anchor_indices(660, 64), np.linspace(0, 659, 64, dtype=np.int64))
        self.assertEqual(len(set(B.anchor_indices(660, 64))), 64)


if __name__ == "__main__":
    unittest.main()
