#!/usr/bin/env python3
"""Adversarial known-answer tests for the contract-only M33 M0 boundary."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_m0_contract", ROOT / "bin" / "m33_m0_contract.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT_PATH = ROOT / "conf" / "m33_m0_materializer_contract.json"


def locus(pos: int, locus_id: int, cm: float, ref: str = "A", alt: str = "C", chrom: int = 22) -> dict:
    return {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "locus_id": locus_id, "cM": cm}


def f0(sample: str = "S1", pos: int = 100, **extra: object) -> dict:
    record = {
        "root_seed": 20260817, "sample_id": sample, "chrom": 22, "pos": pos,
        "ref": "A", "alt": "C", "ANP1": [0.49, 0.30, 0.20], "ANP2": [0.10, 0.40, 0.50],
    }
    record.update(extra)
    return record


class StrictContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = MODULE.load_json(CONTRACT_PATH)

    def test_frozen_contract_and_both_official_technical_anchors_pass(self) -> None:
        MODULE.validate_contract(self.contract)
        self.assertEqual(self.contract["root_registry"]["consumed_technical_roots"],
                         {"root17": 20260817, "root18": 20260818})
        self.assertFalse(self.contract["root_registry"]["scientific_selection"])

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        for payload in ('{"a":1,"a":2}', '{"outer":{"x":1,"x":2}}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                MODULE.loads_strict_json(payload)

    def test_recursive_exact_keys_rejects_nested_extra(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["persistence_contract"]["semantic_hash"]["surprise"] = True
        with self.assertRaises(ValueError):
            MODULE.validate_contract(changed)

    def test_exact_a0_and_kat_anchors_fail_closed(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["anchors"]["known_answer_root18"]["counts"]["flare_loci"] += 1
        with self.assertRaises(ValueError):
            MODULE.validate_contract(changed)

    def test_i0_is_pinned_append_only_and_receipted(self) -> None:
        i0 = self.contract["process_contracts"]["I0_DERIVE_AUTHENTICATE_FLARE_INDEX"]
        self.assertEqual(i0["tool"]["exact_version"], "1.16")
        self.assertEqual(i0["status"], "BLOCKED_PENDING_PULLABLE_TABIX_OCI")
        self.assertTrue(i0["tool"]["local_image_id_technical_anchor"].startswith("sha256:"))
        self.assertEqual(i0["exact_command_argv"], ["tabix", "-p", "vcf", "flare.anc.vcf.gz"])
        self.assertIn("append_only", i0["write_policy"])
        self.assertIn("independent_tbi_sha256", i0["receipt_required_keys"])
        self.assertIn("query_parity_sha256", i0["receipt_required_keys"])

    def test_safe_bridge_physical_sources_and_outputs_are_explicit(self) -> None:
        bridge = self.contract["process_contracts"]["SAFE_BRIDGE"]
        self.assertEqual(bridge["physical_inputs"]["pools"]["format"], "tsv")
        for name in ("rare_catalog", "rare_haplotypes", "selected_sites", "target_calls"):
            self.assertEqual(bridge["physical_inputs"][name]["format"], "tsv_gzip")
        self.assertIn("tree_sequence_plus_pools_plus_ref_pairs", bridge["derivations"]["reference_rare_summary_incremental"])
        self.assertIn("never_source", bridge["derivations"]["common_reference_crosscheck_scope"])
        self.assertTrue(all(not artifact["contains_raw_input_payload"] for artifact in bridge["output_artifacts"].values()))

    def test_axes_dtypes_endianness_and_atomic_reopen_are_frozen(self) -> None:
        outputs = self.contract["process_contracts"]["SAFE_BRIDGE"]["output_artifacts"]
        self.assertEqual(outputs["target_rare_diploid_incremental"]["axes"], ["sample", "locus"])
        self.assertEqual(outputs["reference_rare_summary_incremental"]["dtypes"]["callable_an"], "<u2")
        self.assertEqual(outputs["reference_rare_summary_incremental"]["dtypes"]["minor_af"], "<f8")
        self.assertEqual(outputs["selected_loci_incremental"]["byte_order"], "little_endian_for_multibyte_numeric_fields")
        persistence = self.contract["persistence_contract"]
        self.assertIn("no_overwrite", persistence["write"])
        self.assertIn("reopen", persistence["reopen"])
        self.assertTrue(persistence["semantic_hash"]["archive_metadata_excluded"])

    def test_physical_packed_schema_and_transaction_are_frozen(self) -> None:
        materialize = self.contract["process_contracts"]["MATERIALIZE"]
        arrays = materialize["output_artifacts"]["packed_rare_context_shard"]["arrays"]
        self.assertEqual(arrays["rare_tokens"],
                         {"axes": ["valid_token", "channel"], "shape": ["N", 13], "dtype": "<f4"})
        self.assertEqual(arrays["F0"]["shape"], ["B", 2, "M", 3])
        self.assertEqual(arrays["row_ptr"]["dtype"], "<u8")
        self.assertEqual(materialize["output_artifacts"]["packed_rare_context_shard"]["schema_id"],
                         "m33_m0_packed_rare_context_shard_v1")
        manifest = materialize["output_artifacts"]["bundle_manifest"]
        self.assertTrue(manifest["forbid_extra_keys"])
        self.assertEqual(manifest["ordered_shard_entry_schema"]["exact_keys"][0], "schema_id")
        receipt = materialize["output_artifacts"]["materialization_receipt"]
        self.assertNotIn("READY_sha256", receipt["required_keys"])
        self.assertEqual(receipt["radius_manifest_map"]["exact_order"], ["0.05", "0.1", "0.2", "0.5"])
        ready = materialize["output_artifacts"]["READY"]
        self.assertEqual(ready["schema_id"], "m33_m0_READY_v1")
        self.assertIn("materialization_receipt_sha256", ready["required_keys"])
        self.assertIn("READY", self.contract["persistence_contract"]["bundle_transaction"])
        self.assertIn("ifGenerationMatch_equals_0", self.contract["persistence_contract"]["gcs_policy"])

    def test_safe_bridge_receipt_proves_ref_role_firewall(self) -> None:
        receipt = self.contract["process_contracts"]["SAFE_BRIDGE"]["output_artifacts"]["safe_bridge_receipt"]
        self.assertEqual(receipt["schema_id"], "m33_m0_safe_bridge_receipt_v1")
        for key in ("expected_ref_node_count", "contributing_ref_node_count",
                    "rejected_non_ref_node_count", "expected_ref_nodes_semantic_sha256",
                    "contributing_ref_nodes_semantic_sha256", "role_firewall_pass"):
            self.assertIn(key, receipt["required_keys"])

    def test_load_bearing_mutations_all_fail_closed(self) -> None:
        mutations = []

        def changed(path: tuple[object, ...], value: object) -> dict:
            payload = copy.deepcopy(self.contract)
            cursor = payload
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            return payload

        mutations.extend([
            changed(("process_contracts", "I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "requirements"), []),
            changed(("process_contracts", "I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "input_logical_ids"), ["truth"]),
            changed(("process_contracts", "SAFE_BRIDGE", "physical_inputs", "pools", "authentication"), "NONE"),
            changed(("process_contracts", "SAFE_BRIDGE", "output_artifacts", "target_rare_diploid_incremental", "dtypes", "minor_dosage"), "<f8"),
            changed(("process_contracts", "SAFE_BRIDGE", "output_artifacts", "target_rare_diploid_incremental", "axes"), ["locus", "sample"]),
            changed(("process_contracts", "MATERIALIZE", "input_logical_ids"), []),
            changed(("incremental_partition", "exact_key"), ["pos"]),
            changed(("f0_contract", "operation"), "raw"),
            changed(("f0_contract", "ancestry_order"), ["EUR", "AFR", "ASIA"]),
            changed(("f0_contract", "forbidden_dependencies"), []),
            changed(("primary_transferable_input", "conceptual_shapes", "rare_tokens"), "anything"),
            changed(("packed_loader", "person_batch"), 999),
            changed(("packed_loader", "interval"), "nearest_bp"),
            changed(("persistence_contract", "semantic_hash", "algorithm"), "md5"),
        ])
        for payload in mutations:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MODULE.validate_contract(payload)

    def test_materialize_namespace_and_authorization_are_safe(self) -> None:
        process = self.contract["process_contracts"]["MATERIALIZE"]
        namespace = " ".join(process["input_logical_ids"] + process["output_logical_ids"]).lower()
        for token in process["forbidden_namespace_tokens_case_insensitive"]:
            self.assertNotIn(token.lower(), namespace)
        authorization = self.contract["execution_authorization"]
        for action in ("real_asset_read", "derive_index", "safe_bridge", "materialize", "write_READY",
                       "forward", "backward", "training", "truth_scoring"):
            self.assertFalse(authorization[action])


class DataSemanticsTests(unittest.TestCase):
    def test_reference_af_raw_an_and_no_support(self) -> None:
        self.assertEqual(MODULE.reference_summary(0, 0), {"minor_af": 0.0, "observed_mask": 0, "no_support": 0})
        self.assertEqual(MODULE.reference_summary(0, 60), {"minor_af": 0.0, "observed_mask": 1, "no_support": 1})
        self.assertEqual(MODULE.reference_summary(3, 60), {"minor_af": 0.05, "observed_mask": 1, "no_support": 0})
        for args in ((True, 60), (1, False), (-1, 60), (61, 60)):
            with self.assertRaises(ValueError):
                MODULE.reference_summary(*args)

    def test_missing_and_boolean_dosage_are_rejected(self) -> None:
        self.assertEqual(MODULE.validate_target_cell(0, 0), (0, 0))
        self.assertEqual(MODULE.validate_target_cell(2, 1), (2, 1))
        for args in ((1, 0), (3, 1), (True, 1), (1, True), (-1, 1)):
            with self.assertRaises(ValueError):
                MODULE.validate_target_cell(*args)
        self.assertEqual(MODULE.diploid_minor_dosage(None, 1), (0, 0))
        self.assertEqual(MODULE.diploid_minor_dosage(0, 1), MODULE.diploid_minor_dosage(1, 0))
        with self.assertRaises(ValueError):
            MODULE.diploid_minor_dosage(True, 0)

    def test_nonempty_overlap_partition_is_disjoint_and_reconstructs_union(self) -> None:
        rows = [locus(100, 1, 0.1), locus(200, 2, 0.2), locus(300, 3, 0.3)]
        overlap_key = (22, 200, "A", "C")
        incremental, overlap = MODULE.partition_incremental(rows, [overlap_key])
        self.assertEqual([row["locus_id"] for row in incremental], [1, 3])
        self.assertEqual([row["locus_id"] for row in overlap], [2])
        self.assertEqual(len(incremental) + len(overlap), len(rows))
        with self.assertRaises(ValueError):
            MODULE.partition_incremental(rows, [overlap_key, overlap_key])

    def test_duplicate_key_and_locus_id_fail(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_locus_rows([locus(100, 1, 0.1), locus(100, 2, 0.2)], (1, 1000), (0, 1))
        with self.assertRaises(ValueError):
            MODULE.validate_locus_rows([locus(100, 1, 0.1), locus(200, 1, 0.2)], (1, 1000), (0, 1))
        with self.assertRaises(ValueError):
            MODULE.partition_incremental([locus(100, 1, 0.1), locus(200, 1, 0.2)], [])

    def test_empty_locus_and_f0_domains_fail(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_locus_rows([], (1, 1000), (0, 1))
        with self.assertRaises(ValueError):
            MODULE.validate_f0_join([], set(), set())

    def test_chrom_ref_alt_and_locus_id_fail_closed(self) -> None:
        invalid = [
            locus(100, 1, 0.1, chrom="22"), locus(100, 1, 0.1, ref="AA"),
            locus(100, 1, 0.1, ref="A", alt="A"), locus(100, 1, 0.1, ref=None),
            locus(100, True, 0.1), locus(True, 1, 0.1),
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                MODULE.validate_locus_rows([row], (1, 1000), (0, 1))

    def test_locus_cm_nan_order_ties_and_domain(self) -> None:
        MODULE.validate_locus_rows([locus(100, 1, 0.1), locus(200, 2, 0.1)], (100, 200), (0.1, 0.1))
        for rows in (
            [locus(100, 1, math.nan)],
            [locus(200, 2, 0.2), locus(100, 1, 0.1)],
            [locus(200, 2, 0.1), locus(100, 1, 0.1)],
            [locus(99, 1, 0.1)], [locus(100, 1, 0.3)],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                MODULE.validate_locus_rows(rows, (100, 200), (0.0, 0.2))

    def test_genetic_map_nan_order_ties_and_domain(self) -> None:
        MODULE.validate_genetic_map([{"chrom": 22, "pos": 100, "cM": 0.1}, {"chrom": 22, "pos": 200, "cM": 0.1}], (100, 200))
        invalid = (
            [{"chrom": 22, "pos": 100, "cM": math.nan}],
            [{"chrom": 22, "pos": 100, "cM": 0.2}, {"chrom": 22, "pos": 200, "cM": 0.1}],
            [{"chrom": 22, "pos": 100, "cM": 0.1}, {"chrom": 22, "pos": 100, "cM": 0.2}],
            [{"chrom": 22, "pos": 101, "cM": 0.1}, {"chrom": 22, "pos": 199, "cM": 0.2}],
        )
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                MODULE.validate_genetic_map(rows, (100, 200))

    def test_f0_exact_key_sample_set_duplicates_and_contamination(self) -> None:
        record = f0()
        key = tuple(record[field] for field in MODULE.F0_KEY)
        MODULE.validate_f0_join([record], {key}, {"S1"})
        h0, h1 = MODULE.normalize_f0(record)
        self.assertTrue(math.isclose(sum(h0), 1.0) and math.isclose(sum(h1), 1.0))
        with self.assertRaises(ValueError):
            MODULE.normalize_f0(f0(GT="0|1"))
        with self.assertRaises(ValueError):
            MODULE.validate_f0_join([record, record], {key}, {"S1"})
        with self.assertRaises(ValueError):
            MODULE.validate_f0_join([record], {key}, {"S2"})
        with self.assertRaises(ValueError):
            MODULE.normalize_f0(f0(sample=""))
        with self.assertRaises(ValueError):
            MODULE.normalize_f0(f0(ANP1=[True, 0.0, 0.0]))

    def test_sample_key_and_ref_node_firewall(self) -> None:
        self.assertEqual(MODULE.sample_key_sha256("S1"), MODULE.sample_key_sha256("S1"))
        self.assertNotEqual(MODULE.sample_key_sha256("S1"), MODULE.sample_key_sha256("S2"))
        MODULE.validate_ref_node_firewall([1, 2, 3], [1, 2], [1, 2])
        for contributors in ([1], [1, 2, 3]):
            with self.assertRaises(ValueError):
                MODULE.validate_ref_node_firewall([1, 2, 3], [1, 2], contributors)


class PackedEquivalenceTests(unittest.TestCase):
    def test_four_radii_and_inclusive_endpoints(self) -> None:
        rare = [0.95, 1.00, 1.05, 1.50]
        self.assertEqual(MODULE.context_intervals(rare, [1.0], 0.05), [(0, 3)])
        for radius in MODULE.EXPECTED_RADII:
            self.assertEqual(len(MODULE.context_intervals(rare, [1.0], radius)), 1)
        for radius in (0.0, 0.25, math.nan):
            with self.assertRaises(ValueError):
                MODULE.context_intervals(rare, [1.0], radius)
        with self.assertRaises(ValueError):
            MODULE.context_intervals(rare, [1.1, 1.0], 0.1)

    def test_contiguous_packing_never_truncates(self) -> None:
        self.assertEqual(MODULE.pack_contiguous([2, 0, 3, 4], person_batch=2, token_budget=10),
                         [(0, 3, 10), (3, 4, 8)])
        self.assertEqual(MODULE.pack_contiguous([], person_batch=2, token_budget=10), [])
        for lengths in ([6], [True]):
            with self.assertRaises(ValueError):
                MODULE.pack_contiguous(lengths, person_batch=2, token_budget=10)

    def test_packed_equals_masked_padded_with_poison_and_k0(self) -> None:
        contexts = [[[1.0] * 13, [3.0] * 13], [], [[5.0] * 13]]
        flat, row_ptr = MODULE.pack_contexts(contexts)
        padded, mask = MODULE.padded_contexts(contexts, poison=math.nan)
        recovered = MODULE.unpack_masked_padded(padded, mask)
        recovered_flat, recovered_ptr = MODULE.pack_contexts(recovered)
        self.assertEqual((flat, row_ptr), (recovered_flat, recovered_ptr))
        self.assertEqual(row_ptr, [0, 2, 2, 3])
        poisoned_valid_mask = copy.deepcopy(mask)
        poisoned_valid_mask[1][0] = 1
        with self.assertRaises(ValueError):
            MODULE.unpack_masked_padded(padded, poisoned_valid_mask)
        boolean_mask = copy.deepcopy(mask)
        boolean_mask[0][0] = True
        with self.assertRaises(ValueError):
            MODULE.unpack_masked_padded(padded, boolean_mask)
        with self.assertRaises(ValueError):
            MODULE.pack_contexts([[[1.0] * 12]])

    def test_fixture_semantic_hash_is_stable_and_nan_rejected(self) -> None:
        payload = {"axes": ["row", "channel"], "dtype": "<f8", "values": [[1.0, 2.0], []]}
        self.assertEqual(MODULE.canonical_fixture_sha256(payload),
                         "02625fee51edd8ef4bf14fa9ca23069f858ddc6cc4d839057d526cbb841719d5")
        with self.assertRaises(ValueError):
            MODULE.canonical_fixture_sha256({"value": math.nan})

    def test_physical_array_semantic_hash_includes_schema_shape_dtype_and_bytes(self) -> None:
        arrays = {
            "x": {"axes": ["row", "channel"], "shape": [1, 2], "dtype": "<f4",
                  "data": struct.pack("<ff", 1.0, 2.0)},
            "mask": {"axes": ["row"], "shape": [1], "dtype": "|u1", "data": bytes([1])},
        }
        observed = MODULE.canonical_array_bundle_sha256("m33_fixture_v1", arrays)
        self.assertEqual(observed, "f5a7e0488066afd00939ad746737620160bdac333fc55a9b95e4df8ecccddfcd")
        changed = copy.deepcopy(arrays)
        changed["mask"]["data"] = bytes([0])
        self.assertNotEqual(observed, MODULE.canonical_array_bundle_sha256("m33_fixture_v1", changed))
        with self.assertRaises(ValueError):
            MODULE.canonical_array_bundle_sha256("m33_fixture_v1", {
                "x": {"axes": ["row"], "shape": [2], "dtype": "<f8", "data": bytes(8)},
            })


if __name__ == "__main__":
    unittest.main()
