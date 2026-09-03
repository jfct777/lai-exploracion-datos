#!/usr/bin/env python3
"""Unit and static contracts for the M27F-b REF-to-VALID support audit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))

import audit_m27f_ref_support as refsupport  # noqa: E402
import audit_m27f_valid_support as validsupport  # noqa: E402
import claim_m27f_validation_opening as claim  # noqa: E402
import project_m27f_ref_panel as projection  # noqa: E402


class TestM27FRefValidSupport(unittest.TestCase):
    def test_private_tsv_is_deterministic_and_empty_safe(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            fields = ["chrom", "pos"]
            rows = [{"chrom": "22", "pos": 10}]
            first_path = Path(first) / "support.tsv"
            second_path = Path(second) / "support.tsv"
            refsupport.write_private_tsv(first_path, rows, fields)
            refsupport.write_private_tsv(second_path, rows, fields)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            empty_path = Path(first) / "empty.tsv"
            refsupport.write_private_tsv(empty_path, [], fields)
            self.assertEqual(empty_path.read_text(encoding="utf-8"), "chrom\tpos\n")
            self.assertEqual(empty_path.stat().st_mode & 0o777, 0o600)

    def test_frozen_orientation_and_phase_carrier_logic(self):
        self.assertTrue(refsupport.usable_carrier(1, True, True, True))
        self.assertFalse(refsupport.usable_carrier(1, False, True, True))
        self.assertFalse(refsupport.usable_carrier(0, True, True, True))
        self.assertTrue(refsupport.usable_carrier(0, False, True, False))
        self.assertFalse(refsupport.usable_carrier(2, True, True, False))
        self.assertFalse(refsupport.usable_carrier(0, True, False, False))

    def test_missing_is_not_reference_and_frequency_does_not_require_phase(self):
        samples = ["a", "b", "c"]
        metadata = {
            "a": {"atomic_unit_id": "u1"},
            "b": {"atomic_unit_id": "u1"},
            "c": {"atomic_unit_id": "u2"},
        }
        parsed = [(0, False, False), (1, False, True), (0, False, True)]
        metrics = refsupport.role_metrics(
            parsed, [0, 1, 2], samples, metadata, minor_is_alt=True
        )
        self.assertEqual(metrics["called_samples"], 2)
        self.assertEqual(metrics["minor_ac"], 1)
        self.assertEqual(metrics["minor_an"], 4)
        self.assertEqual(metrics["carrier_samples"], 0)
        self.assertEqual(metrics["unphased_het_carriers_excluded"], 1)
        self.assertEqual(metrics["fully_callable_atomic_units"], 1)
        self.assertEqual(metrics["carrier_atomic_units"], 0)
        self.assertEqual(metrics["carrier_atomic_units_upper_bound"], 1)
        self.assertEqual(metrics["unresolved_noncarrier_atomic_units"], 1)

    def test_callability_bounds_prevent_false_negative(self):
        samples = ["called_ref", "missing"]
        metadata = {
            "called_ref": {"atomic_unit_id": "same_unit"},
            "missing": {"atomic_unit_id": "same_unit"},
        }
        metrics = refsupport.role_metrics(
            [(0, False, True), (0, False, False)],
            [0, 1],
            samples,
            metadata,
            minor_is_alt=True,
        )
        self.assertEqual(metrics["carrier_atomic_units"], 0)
        self.assertEqual(metrics["fully_callable_atomic_units"], 0)
        self.assertEqual(metrics["carrier_atomic_units_upper_bound"], 1)

    def test_catalog_digest_is_order_invariant_and_orientation_sensitive(self):
        first = {("22", 20, "A", "G"): True, ("22", 10, "C", "T"): False}
        second = dict(reversed(list(first.items())))
        self.assertEqual(refsupport.catalog_digest(first), refsupport.catalog_digest(second))
        changed = dict(first)
        changed[("22", 20, "A", "G")] = False
        self.assertNotEqual(refsupport.catalog_digest(first), refsupport.catalog_digest(changed))

    def test_role_extraction_keeps_discovery_core_distinct_from_quarantine(self):
        rows = [
            {"sample_id": "d", "role": "DISCOVERY", "exclusion_reason": "DISCOVERY_CORE"},
            {"sample_id": "q", "role": "DISCOVERY", "exclusion_reason": "DISCOVERY_IBD_CLOSURE"},
            {"sample_id": "r", "role": "REF_TRAIN", "exclusion_reason": ""},
            {"sample_id": "v", "role": "SOURCE_VALID", "exclusion_reason": ""},
            {"sample_id": "t", "role": "SOURCE_TEST", "exclusion_reason": ""},
        ]
        roles = projection.samples_by_role(rows)
        self.assertEqual(roles["DISCOVERY_CORE"], ["d"])
        self.assertNotIn("q", set().union(*map(set, roles.values())))

    def test_preregistration_pins_canonical_split_and_fixed_threshold(self):
        prereg = json.loads(
            (REPO / "conf/m27f_ref_support_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        upstream = prereg["upstream_contract"]
        support = prereg["support_contract"]
        self.assertEqual(prereg["version"], 2)
        self.assertEqual(
            upstream["m27f_split_private_sha256"],
            "7e4abe2aa57c7375023268e56e3c1dbf9c7bbedc31c3acf1e07b4b07dabfdd07",
        )
        self.assertEqual(upstream["m27f_split_generator_commit"], "3b27440feb929e3229932be854a4c116befcb2a1")
        self.assertEqual(upstream["expected_ref_native_american_atomic_units"], 4)
        self.assertEqual(upstream["expected_valid_native_american_atomic_units"], 2)
        self.assertEqual(support["primary_ref_min_atomic_units"], 2)
        self.assertEqual(support["required_valid_atomic_units"], 2)
        self.assertFalse(support["select_using_validation"])
        self.assertFalse(support["source_test_opened"])

    def test_threshold_sensitivity_never_changes_primary(self):
        rows = [
            {"ref_nam_carrier_atomic_units": value, "in_frozen_baseline": inside}
            for value, inside in [(1, True), (2, False), (3, True), (4, False)]
        ]
        summary = refsupport.threshold_summary(rows, [1, 2, 3, 4])
        self.assertEqual(summary["1"]["all"], 4)
        self.assertEqual(summary["2"]["all"], 3)
        self.assertEqual(summary["2"]["outside_frozen_baseline"], 2)
        self.assertEqual(summary["4"]["all"], 1)

    def test_validation_sensitivity_reports_but_does_not_tune(self):
        rows = [
            {
                "ref_nam_carrier_atomic_units": 1,
                "valid_nam_carrier_atomic_units": 2,
                "in_frozen_baseline": False,
            },
            {
                "ref_nam_carrier_atomic_units": 2,
                "valid_nam_carrier_atomic_units": 2,
                "in_frozen_baseline": False,
            },
            {
                "ref_nam_carrier_atomic_units": 4,
                "valid_nam_carrier_atomic_units": 1,
                "in_frozen_baseline": True,
            },
        ]
        summary = validsupport.sensitivity_summary(rows, [1, 2, 3, 4], 2)
        self.assertEqual(summary["1"]["observed_in_both_valid_units"], 2)
        self.assertEqual(summary["2"]["observed_in_both_valid_units"], 1)
        self.assertEqual(summary["2"]["validated_outside_frozen_baseline"], 1)

    def test_negative_decisions_become_inconclusive_when_missing_can_rescue(self):
        self.assertEqual(
            refsupport.classify_ref_decision(0, 1),
            "INCONCLUSIVE_REF_CALLABILITY",
        )
        self.assertEqual(
            refsupport.classify_ref_decision(0, 0),
            "STOP_REF_NO_TRANSFERABLE_SUPPORT",
        )
        self.assertEqual(
            refsupport.classify_ref_decision(3, 0),
            "GO_VALID_SUPPORT_AUDIT",
        )
        self.assertEqual(
            validsupport.classify_valid_decision(0, 0, 1, 1),
            "INCONCLUSIVE_VALID_CALLABILITY",
        )
        self.assertEqual(
            validsupport.classify_valid_decision(1, 0, 0, 1),
            "INCONCLUSIVE_ADDITIONAL_VARIANT_KEY_CALLABILITY",
        )
        self.assertEqual(
            validsupport.classify_valid_decision(1, 1, 0, 0),
            "GO_SUPPORT_CANDIDATES_ONLY",
        )

    def test_joint_missingness_must_rescue_the_same_site(self):
        rows = [
            {
                "ref_nam_carrier_atomic_units": 2,
                "ref_nam_carrier_atomic_units_upper_bound": 2,
                "valid_nam_carrier_atomic_units": 0,
                "valid_nam_carrier_atomic_units_upper_bound": 0,
                "in_frozen_baseline": True,
            },
            {
                "ref_nam_carrier_atomic_units": 1,
                "ref_nam_carrier_atomic_units_upper_bound": 2,
                "valid_nam_carrier_atomic_units": 0,
                "valid_nam_carrier_atomic_units_upper_bound": 0,
                "in_frozen_baseline": False,
            },
        ]
        observed, possible, observed_b, possible_b = validsupport.joint_support_sets(
            rows, 2, 2
        )
        self.assertEqual((len(observed), len(possible)), (0, 0))
        self.assertEqual((len(observed_b), len(possible_b)), (0, 0))
        self.assertEqual(
            validsupport.classify_valid_decision(0, 0, 0, 0),
            "STOP_VALID_NO_TRANSFERABLE_SUPPORT",
        )
        rows[1]["valid_nam_carrier_atomic_units_upper_bound"] = 2
        _observed, possible, _observed_b, possible_b = (
            validsupport.joint_support_sets(rows, 2, 2)
        )
        self.assertEqual((len(possible), len(possible_b)), (1, 1))

    def test_sliding_span_concentration_is_boundary_independent(self):
        positions = [499_999, 500_001, 1_400_000]
        self.assertEqual(validsupport.max_sites_in_span(positions, 500_000), 2)
        summary = validsupport.spatial_concentration(
            [{"pos": value} for value in positions], [500_000]
        )
        self.assertAlmostEqual(
            summary["max_fraction_in_any_500000_bp_span"], 2 / 3
        )

    def test_projection_is_mechanical_and_never_creates_test_bcf(self):
        source = (REPO / "bin/project_m27f_ref_panel.py").read_text(encoding="utf-8")
        self.assertIn('"--samples-file"', source)
        self.assertIn('"--no-update"', source)
        self.assertIn('"--remove",\n            "INFO"', source)
        self.assertIn('"--no-version"', source)
        self.assertNotIn('"--include"', source)
        self.assertNotIn('"--exclude"', source)
        self.assertNotIn('"SOURCE_TEST": "m27f_test"', source)
        self.assertNotIn("KING", source)

    def test_nextflow_orders_valid_after_ref_and_keeps_test_absent(self):
        module = (REPO / "modules/27F_REF_SUPPORT_AUDIT.nf").read_text(encoding="utf-8")
        workflow = (REPO / "workflows/m27f_ref_support_audit.nf").read_text(
            encoding="utf-8"
        )
        self.assertIn("AUDIT_M27F_REF_SUPPORT.out.private_primary_catalog", workflow)
        self.assertIn("AUDIT_M27F_REF_SUPPORT.out.public_support", workflow)
        self.assertIn("CLAIM_M27F_VALIDATION_OPENING.out.receipt", workflow)
        self.assertIn("PROJECT_M27F_SUPPORT_PANEL.out.valid_projection", workflow)
        self.assertNotIn('path("m27f_test.chr22', module + workflow)
        self.assertNotIn("m27f_test.samples", module + workflow)
        self.assertNotIn("KING", module + workflow)
        self.assertEqual(
            module.count("container params.m27f_support_container_image"), 4
        )
        self.assertIn("container false", module)
        self.assertIn("--claim-uri", module)
        claim_source = (REPO / "bin/claim_m27f_validation_opening.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--if-generation-match=0"', claim_source)
        self.assertIn("os.O_EXCL", claim_source)
        self.assertLess(
            claim_source.index("if not authorized:"),
            claim_source.rindex("publish_gcs_claim(candidate"),
        )
        self.assertIn("def authorizedOpening", workflow)
        self.assertIn("'VALIDATION_OPENING_FROZEN'", workflow)
        self.assertIn("--claim-registry-dir", module)
        self.assertIn("--claim-key", module)
        self.assertIn(
            "chmod 600 m27f_ref_valid_support.private.tsv", module
        )
        self.assertNotIn("--run-provenance-ref ../run_provenance.json", module)
        valid_source = (REPO / "bin/audit_m27f_valid_support.py").read_text(
            encoding="utf-8"
        )
        run_body = valid_source[valid_source.index("def run(args:") :]
        self.assertLess(
            run_body.index("authenticate_before_validation(args, prereg)"),
            run_body.index("read_bcf_samples(args.valid_bcf"),
        )

    def test_validation_opening_authenticates_manifest_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preregistration = root / "m27f_ref_support_preregistration.json"
            preregistration.write_bytes(
                (REPO / "conf/m27f_ref_support_preregistration.json").read_bytes()
            )
            fixture_prereg = json.loads(preregistration.read_text(encoding="utf-8"))
            fixture_prereg["support_contract"]["validation_claim_key"] = (
                "m27fb_fixture_once"
            )
            fixture_prereg["support_contract"]["validation_claim_uri"] = (
                "gs://fixture/claims/m27fb_fixture_once.json"
            )
            preregistration.write_text(
                json.dumps(fixture_prereg, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            support = root / "m27f_ref_site_support.private.tsv"
            catalog = root / "m27f_ref_primary_catalog.private.tsv"
            support.write_text("site\n", encoding="utf-8")
            catalog.write_text("selected\n", encoding="utf-8")
            projection_public = root / "m27f_projection.public.json"
            projection_public.write_text(
                json.dumps(
                    {
                        "stage": "M27F_REF_VALID_MECHANICAL_PROJECTION",
                        "decision": "GO_REF_SUPPORT_AUDIT",
                        "gates": {"P0": "PASS", "P1": "PASS"},
                        "source_test_projection_created": False,
                        "source_test_samples_in_projected_outputs": 0,
                        "projections": {
                            "DISCOVERY_CORE": {
                                "bcf_sha256": "discovery-hash",
                                "n_records_with_nonempty_info": 0,
                            },
                            "REF_TRAIN": {
                                "bcf_sha256": "ref-hash",
                                "n_records_with_nonempty_info": 0,
                            },
                            "SOURCE_VALID": {
                                "bcf_sha256": "valid-hash",
                                "n_records_with_nonempty_info": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            projection_manifest = root / "m27f_projection.manifest.json"
            projection_manifest.write_text(
                json.dumps(
                    {
                        "stage": "M27F_REF_VALID_MECHANICAL_PROJECTION",
                        "sha256": {
                            projection_public.name: claim.sha256_file(projection_public),
                            "m27f_discovery_core.chr22.private.bcf": "discovery-hash",
                            "m27f_ref.chr22.private.bcf": "ref-hash",
                            "m27f_valid.chr22.private.bcf": "valid-hash",
                        },
                    }
                ),
                encoding="utf-8",
            )
            ref_public = root / "m27f_ref_support.public.json"
            ref_public.write_text(
                json.dumps(
                    {
                        "stage": "M27F_REF_SUPPORT_SELECTION",
                        "decision": "GO_VALID_SUPPORT_AUDIT",
                        "private_ref_support_sha256": claim.sha256_file(support),
                        "private_primary_catalog_sha256": claim.sha256_file(catalog),
                    }
                ),
                encoding="utf-8",
            )
            ref_manifest = root / "m27f_ref_support.manifest.json"
            ref_manifest.write_text(
                json.dumps(
                    {
                        "stage": "M27F_REF_SUPPORT_SELECTION",
                        "inputs": {
                            preregistration.name: claim.sha256_file(preregistration),
                            projection_public.name: claim.sha256_file(
                                projection_public
                            ),
                            projection_manifest.name: claim.sha256_file(
                                projection_manifest
                            ),
                        },
                        "sha256": {
                            support.name: claim.sha256_file(support),
                            catalog.name: claim.sha256_file(catalog),
                            ref_public.name: claim.sha256_file(ref_public),
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "m27f_validation_opening.receipt.json"
            valid_script = root / "audit_m27f_valid_support.py"
            valid_script.write_bytes(
                (REPO / "bin/audit_m27f_valid_support.py").read_bytes()
            )
            arguments = SimpleNamespace(
                run_id="fixture",
                claim_registry_dir=root / "claims",
                claim_key="m27fb_fixture_once",
                claim_uri="gs://fixture/claims/m27fb_fixture_once.json",
                gcloud="gcloud",
                claim_py=REPO / "bin/claim_m27f_validation_opening.py",
                validation_contract_py=REPO / "bin/m27f_validation_contract.py",
                valid_audit_py=valid_script,
                ref_audit_py=REPO / "bin/audit_m27f_ref_support.py",
                m27e_py=REPO / "bin/audit_m27e_ibd_rare_transfer.py",
                bridge_py=REPO / "bin/audit_rare_scaffold_bridge.py",
                container_digest=(
                    "sha256:3baca87521291075c1ea5c7b17a3706a5030adda65174313fbc203c1a1324b35"
                ),
                container_image=(
                    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/dnabr-qc@"
                    "sha256:3baca87521291075c1ea5c7b17a3706a5030adda65174313fbc203c1a1324b35"
                ),
                projection_public=projection_public,
                projection_manifest=projection_manifest,
                ref_support_private=support,
                ref_primary_catalog=catalog,
                ref_public=ref_public,
                ref_manifest=ref_manifest,
                preregistration=preregistration,
                out=output,
            )
            with mock.patch.object(claim, "publish_gcs_claim") as publish:
                receipt = claim.run(arguments)
                claim.run(arguments)
                original_valid_script = valid_script.read_bytes()
                valid_script.write_bytes(original_valid_script + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "STOP_VALIDATION_REOPENING"):
                    claim.run(arguments)
                valid_script.write_bytes(original_valid_script)
                arguments.run_id = "different-run"
                with self.assertRaisesRegex(ValueError, "STOP_VALIDATION_REOPENING"):
                    claim.run(arguments)
            self.assertEqual(publish.call_count, 4)
            self.assertEqual(receipt["decision"], "VALIDATION_OPENING_FROZEN")
            self.assertEqual(receipt["authorized_analytical_openings"], 1)
            self.assertFalse(receipt["source_valid_genotypes_read_to_create_receipt"])
            self.assertEqual(
                (root / "claims/m27fb_fixture_once.json").read_bytes(),
                output.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
