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
        self.assertIn("raw_binary_state_equals_authenticated_rare_catalog_minor_code",
                      bridge["derivations"]["reference_rare_summary_incremental"])
        self.assertIn("sum_indicator_raw_REF_haplotype_state_equals_minor_code",
                      bridge["reference_minor_orientation"]["minor_ac_formula"])
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
        for key in ("selected_all_count", "selected_incremental_count", "selected_overlap_count",
                    "partition_disjoint_union_pass", "minor_code_0_locus_count",
                    "minor_code_1_locus_count", "minor_orientation_source_semantic_sha256",
                    "reference_minor_summary_semantic_sha256"):
            self.assertIn(key, receipt["required_keys"])

    def test_fit_manifest_sample_axis_tolerance_and_privacy_are_frozen(self) -> None:
        materialize = self.contract["process_contracts"]["MATERIALIZE"]
        self.assertIn("authenticated_fit_callable_normalization_manifest", materialize["input_logical_ids"])
        self.assertEqual(materialize["fit_callable_normalization_manifest"]["score_only_or_eval_contribution"],
                         "STOP_BEFORE_ANY_SHARD_WRITE")
        self.assertEqual(self.contract["f0_contract"]["float32_simplex_absolute_tolerance"], 5e-06)
        self.assertFalse(self.contract["privacy_contract"]["external_sharing"])
        self.assertIn("sample_axis_semantic_sha256",
                      materialize["output_artifacts"]["bundle_manifest"]["required_keys"])
        self.assertEqual(self.contract["anchors"]["pre4_contract"]["preregistration_sha256"],
                         "4308bbf33ae28f554f701da33efdc185264f9f407d62661e7048e0345687eb8b")
        controls = self.contract["control_views"]
        self.assertEqual(controls["target_same_locus_sham"]["seeds"],
                         [1277457345, 943666774, 1858042568])
        self.assertEqual(controls["REF_label_sham"]["seeds"], [79351217, 202307732, 1737132171])

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

    def test_reference_minor_orientation_uses_minor_code_and_excludes_missing(self) -> None:
        states = [0, 0, 1, None]
        self.assertEqual(MODULE.reference_minor_summary(states, 0), {
            "minor_ac": 2, "callable_an": 3, "minor_af": 2 / 3,
            "observed_mask": 1, "no_support": 0})
        self.assertEqual(MODULE.reference_minor_summary(states, 1), {
            "minor_ac": 1, "callable_an": 3, "minor_af": 1 / 3,
            "observed_mask": 1, "no_support": 0})
        self.assertEqual(MODULE.reference_minor_summary([None, None], 0), {
            "minor_ac": 0, "callable_an": 0, "minor_af": 0.0,
            "observed_mask": 0, "no_support": 0})
        for bad_states, minor_code in (([0, 2], 0), ([False, 1], 1), ([0, 1], 2), ([0, 1], True)):
            with self.assertRaises(ValueError):
                MODULE.reference_minor_summary(bad_states, minor_code)

    def test_missing_and_boolean_dosage_are_rejected(self) -> None:
        self.assertEqual(MODULE.validate_target_cell(0, 0), (0, 0))
        self.assertEqual(MODULE.validate_target_cell(2, 1), (2, 1))
        for args in ((1, 0), (3, 1), (True, 1), (1, True), (-1, 1)):
            with self.assertRaises(ValueError):
                MODULE.validate_target_cell(*args)
        self.assertEqual(MODULE.diploid_minor_dosage(None, 1, 1), (0, 0))
        self.assertEqual(MODULE.diploid_minor_dosage(0, 1, 1), MODULE.diploid_minor_dosage(1, 0, 1))
        with self.assertRaises(ValueError):
            MODULE.diploid_minor_dosage(True, 0, 1)

    def test_minor_orientation_uses_authenticated_minor_code(self) -> None:
        self.assertEqual(MODULE.diploid_minor_dosage(0, 0, 0), (2, 1))
        self.assertEqual(MODULE.diploid_minor_dosage(0, 1, 0), (1, 1))
        self.assertEqual(MODULE.diploid_minor_dosage(1, 1, 0), (0, 1))
        self.assertEqual(MODULE.diploid_minor_dosage(0, 0, 1), (0, 1))
        self.assertEqual(MODULE.diploid_minor_dosage(0, 1, 1), (1, 1))
        self.assertEqual(MODULE.diploid_minor_dosage(1, 1, 1), (2, 1))
        for minor_code in (-1, 2, True):
            with self.assertRaises(ValueError):
                MODULE.diploid_minor_dosage(0, 1, minor_code)

    def test_nonempty_overlap_partition_is_disjoint_and_reconstructs_union(self) -> None:
        rows = [locus(100, 1, 0.1), locus(200, 2, 0.2), locus(300, 3, 0.3)]
        overlap_key = (22, 200, "A", "C")
        incremental, overlap = MODULE.partition_incremental(rows, [overlap_key])
        self.assertEqual([row["locus_id"] for row in incremental], [1, 3])
        self.assertEqual([row["locus_id"] for row in overlap], [2])
        self.assertEqual(len(incremental) + len(overlap), len(rows))
        with self.assertRaises(ValueError):
            MODULE.partition_incremental(rows, [overlap_key, overlap_key])
        with self.assertRaises(ValueError):
            MODULE.partition_incremental([locus(200, 2, 0.2, ref="C", alt="A")], [overlap_key])

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

    def test_fit_manifest_excludes_score_only_and_sample_axis_is_ordered(self) -> None:
        digest = "a" * 64
        sources = {"2024931463": digest, "1324432253": "b" * 64}
        maxima = {"AFR": 60, "EUR": 58, "ASIA": 56}
        manifest = {
            "stage": "M33_M0_FIT_NORMALIZATION", "schema_id": "m33_m0_fit_callable_normalization_manifest_v1",
            "status": "PASS", "profile": "DEVELOPMENT_ROTATION", "rotation_id": "R0",
            "fit_root_seeds": [2024931463, 1324432253], "score_only_root_seed": 386357765,
            "max_callable_an_by_ancestry": maxima,
            "source_reference_summary_sha256_by_fit_root": sources,
            "semantic_sha256": "", "source_auth_sha256": "d" * 64,
        }
        manifest["semantic_sha256"] = MODULE.fit_manifest_semantic_sha256(manifest)
        source_auth = manifest["source_auth_sha256"]
        MODULE.validate_fit_normalization_manifest(manifest, sources, maxima, source_auth)
        leaked = copy.deepcopy(manifest)
        leaked["fit_root_seeds"] = [2024931463, 386357765]
        leaked["source_reference_summary_sha256_by_fit_root"] = {
            "2024931463": digest, "386357765": "b" * 64}
        leaked["semantic_sha256"] = MODULE.fit_manifest_semantic_sha256(leaked)
        with self.assertRaises(ValueError):
            MODULE.validate_fit_normalization_manifest(
                leaked, leaked["source_reference_summary_sha256_by_fit_root"], maxima, source_auth)
        wrong_maxima = copy.deepcopy(manifest)
        wrong_maxima["max_callable_an_by_ancestry"]["AFR"] = 61
        wrong_maxima["semantic_sha256"] = MODULE.fit_manifest_semantic_sha256(wrong_maxima)
        with self.assertRaises(ValueError):
            MODULE.validate_fit_normalization_manifest(wrong_maxima, sources, maxima, source_auth)
        with self.assertRaises(ValueError):
            MODULE.validate_fit_normalization_manifest(
                manifest, {**sources, "2024931463": "e" * 64}, maxima, source_auth)
        stale_semantic = copy.deepcopy(manifest)
        stale_semantic["source_auth_sha256"] = "e" * 64
        stale_semantic["semantic_sha256"] = MODULE.fit_manifest_semantic_sha256(stale_semantic)
        with self.assertRaises(ValueError):
            MODULE.validate_fit_normalization_manifest(stale_semantic, sources, maxima, source_auth)
        sample_ids = ["S1", "S2"]
        keys = [MODULE.sample_key_sha256(value) for value in sample_ids]
        MODULE.validate_sample_axis_join(keys, sample_ids)
        with self.assertRaises(ValueError):
            MODULE.validate_sample_axis_join(list(reversed(keys)), sample_ids)

    def test_float32_simplex_and_cross_radius_nested_loci(self) -> None:
        MODULE.validate_float32_simplex([0.2, 0.3, 0.500004])
        with self.assertRaises(ValueError):
            MODULE.validate_float32_simplex([0.2, 0.3, 0.500006])
        MODULE.validate_cross_radius_loci({0.05: [2], 0.1: [1, 2], 0.2: [1, 2, 3], 0.5: [0, 1, 2, 3]})
        with self.assertRaises(ValueError):
            MODULE.validate_cross_radius_loci({0.05: [2], 0.1: [2, 1], 0.2: [1, 2, 3], 0.5: [0, 1, 2, 3]})
        loci = {0.05: [2], 0.1: [1, 2], 0.2: [1, 2, 3], 0.5: [0, 1, 2, 3]}
        token = [0.1] * 13
        payloads = {radius: [token[:] for _ in values] for radius, values in loci.items()}
        MODULE.validate_cross_radius_payloads(loci, payloads)
        for non_geometry_channel in range(11):
            broken_mask = copy.deepcopy(payloads)
            broken_mask[0.5][2][non_geometry_channel] = 0.0
            with self.assertRaises(ValueError):
                MODULE.validate_cross_radius_payloads(loci, broken_mask)

    def test_materialization_chain_binds_rotation_root_role_and_manifests(self) -> None:
        fit_hash, source_auth = "a" * 64, "b" * 64
        sample_axis, marker_axis, channel_hash = "c" * 64, "d" * 64, "e" * 64
        bundle_hashes = {radius: str(index + 1) * 64 for index, radius in enumerate(MODULE.RADIUS_KEYS)}
        prefixes = {radius: f"gs://projects-usp/dnaBr-lai/datalake/m33/{radius}/"
                    for radius in MODULE.RADIUS_KEYS}
        authenticated_shards = {}
        bundles = {}
        for index, (radius_key, radius) in enumerate(zip(MODULE.RADIUS_KEYS, MODULE.EXPECTED_RADII)):
            shard = {
                "schema_id": "m33_m0_packed_rare_context_shard_v1", "shard_ordinal": 0,
                "person_start": 0, "person_end_exclusive": 2, "marker_start": 0,
                "marker_end_exclusive": 3, "valid_token_count": 10,
                "gcs_uri": f"{prefixes[radius_key]}part-00000.npz", "gcs_generation": index + 1,
                "raw_sha256": format(index + 10, "x") * 64,
                "semantic_sha256": format(index + 5, "x") * 64,
            }
            authenticated_shards[radius_key] = [shard]
            bundles[radius_key] = {
                "stage": "M33_M0_MATERIALIZE_BUNDLE", "schema_id": "m33_m0_bundle_manifest_v1",
                "status": "PASS", "root_label": "development-root", "root_seed": 386357765,
                "rotation_id": "R0", "role_in_rotation": "SCORE", "radius_cM": radius,
                "fit_callable_normalization_manifest_sha256": fit_hash, "sample_count": 2,
                "sample_axis_semantic_sha256": sample_axis, "marker_count": 3,
                "marker_axis_semantic_sha256": marker_axis, "ordered_shards": [shard],
                "raw_semantic_sha256": "f" * 64, "channel_semantic_sha256": channel_hash,
                "source_auth_sha256": source_auth,
            }
        receipt = {
            "stage": "M33_M0_MATERIALIZATION_RECEIPT",
            "schema_id": "m33_m0_materialization_receipt_v1", "status": "PASS",
            "root_label": "development-root", "root_seed": 386357765, "rotation_id": "R0",
            "role_in_rotation": "SCORE", "radii_cM": MODULE.EXPECTED_RADII,
            "fit_callable_normalization_manifest_sha256": fit_hash, "sample_count": 2,
            "sample_axis_semantic_sha256": sample_axis, "marker_count": 3,
            "marker_axis_semantic_sha256": marker_axis,
            "ordered_bundle_manifest_sha256_by_radius": bundle_hashes,
            "raw_semantic_sha256": "0" * 64, "channel_semantic_sha256": channel_hash,
            "source_auth_sha256": source_auth, "reopen_verified": True, "append_only": True,
        }
        receipt_hash = "9" * 64
        ready = {
            "stage": "M33_M0_READY", "schema_id": "m33_m0_READY_v1", "status": "PASS",
            "root_label": "development-root", "root_seed": 386357765, "rotation_id": "R0",
            "role_in_rotation": "SCORE", "fit_callable_normalization_manifest_sha256": fit_hash,
            "sample_count": 2, "sample_axis_semantic_sha256": sample_axis, "marker_count": 3,
            "marker_axis_semantic_sha256": marker_axis, "materialization_receipt_sha256": receipt_hash,
            "ordered_bundle_manifest_sha256_by_radius": bundle_hashes, "source_auth_sha256": source_auth,
        }
        args = (bundles, bundle_hashes, authenticated_shards, prefixes,
                receipt, receipt_hash, ready, fit_hash, source_auth,
                "R0", 386357765, "SCORE")
        MODULE.validate_materialization_output_chain(*args)

        relabeled_bundles, relabeled_receipt, relabeled_ready = (
            copy.deepcopy(bundles), copy.deepcopy(receipt), copy.deepcopy(ready))
        for bundle in relabeled_bundles.values():
            bundle["role_in_rotation"] = "FIT"
        relabeled_receipt["role_in_rotation"] = "FIT"
        relabeled_ready["role_in_rotation"] = "FIT"
        with self.assertRaises(ValueError):
            MODULE.validate_materialization_output_chain(
                relabeled_bundles, bundle_hashes, authenticated_shards, prefixes,
                relabeled_receipt, receipt_hash, relabeled_ready,
                fit_hash, source_auth, "R0", 386357765, "SCORE")

        drifted_ready = copy.deepcopy(ready)
        drifted_ready["sample_axis_semantic_sha256"] = "8" * 64
        with self.assertRaises(ValueError):
            MODULE.validate_materialization_output_chain(
                bundles, bundle_hashes, authenticated_shards, prefixes, receipt, receipt_hash, drifted_ready,
                fit_hash, source_auth, "R0", 386357765, "SCORE")

        malformed_bundles, malformed_shards = copy.deepcopy(bundles), copy.deepcopy(authenticated_shards)
        malformed_bundles["0.05"]["ordered_shards"] = [{"ordinal": 0}]
        malformed_shards["0.05"] = [{"ordinal": 0}]
        with self.assertRaises(ValueError):
            MODULE.validate_materialization_output_chain(
                malformed_bundles, bundle_hashes, malformed_shards, prefixes,
                receipt, receipt_hash, ready, fit_hash, source_auth, "R0", 386357765, "SCORE")

        reversed_map = dict(reversed(list(bundle_hashes.items())))
        reversed_receipt, reversed_ready = copy.deepcopy(receipt), copy.deepcopy(ready)
        reversed_receipt["ordered_bundle_manifest_sha256_by_radius"] = reversed_map
        reversed_ready["ordered_bundle_manifest_sha256_by_radius"] = reversed_map
        with self.assertRaises(ValueError):
            MODULE.validate_materialization_output_chain(
                bundles, bundle_hashes, authenticated_shards, prefixes,
                reversed_receipt, receipt_hash, reversed_ready,
                fit_hash, source_auth, "R0", 386357765, "SCORE")


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
