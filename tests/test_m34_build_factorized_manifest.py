#!/usr/bin/env python3
"""Known-answer and failure tests for the M34 factor-manifest boundary."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_build_factorized_manifest as subject  # noqa: E402
import m34_train_factorized as trainer  # noqa: E402


def write_npz(path: Path, values: dict[str, np.ndarray]) -> Path:
    np.savez(path, **values)
    return path


def factor_files(root: Path, split: str, people: int,
                 reference: Path) -> dict[str, Path]:
    prefix = f"M34_R0_{split}"
    keys = np.asarray(
        [f"{prefix}-{index:04d}".encode().ljust(64, b"0") for index in range(people)],
        dtype="|S64",
    )
    loci = np.asarray([101, 202], dtype="<u8")
    selected = write_npz(root / f"{split}.selected.npz", {
        "locus_id": loci,
        "chrom": np.full(2, 22, dtype="|u1"),
        "pos": np.asarray([100, 250], dtype="<i8"),
        "ref": np.asarray([b"A", b"C"], dtype="|S1"),
        "alt": np.asarray([b"G", b"T"], dtype="|S1"),
        "cM": np.asarray([0.0, 0.3], dtype="<f8"),
    })
    target = write_npz(root / f"{split}.target.npz", {
        "sample_key_sha256": keys,
        "locus_id": loci,
        "minor_dosage": np.zeros((people, 2), dtype="|i1"),
        "observed_mask": np.ones((people, 2), dtype="|u1"),
    })
    marker_pos = np.asarray([150, 200, 300], dtype="<i8")
    f0 = np.full((people, 2, 3, 3), 1.0 / 3.0, dtype="<f4")
    f0_path = write_npz(root / f"{split}.f0.npz", {
        "sample_key_sha256": keys,
        "marker_chrom": np.full(3, 22, dtype="|u1"),
        "marker_pos": marker_pos,
        "marker_ref": np.asarray([b"A", b"C", b"G"], dtype="|S1"),
        "marker_alt": np.asarray([b"G", b"T", b"A"], dtype="|S1"),
        "F0": f0,
    })
    marker = write_npz(root / f"{split}.marker.npz", {
        "marker_cM": np.asarray([0.1, 0.2, 0.4], dtype="<f8"),
    })
    truth = write_npz(root / f"{split}.truth.npz", {
        "sample_key_sha256": keys,
        "marker_pos": marker_pos,
        "labels": np.zeros((people, 2, 3), dtype="|i1"),
    })
    return {
        "selected_variant": selected,
        "target": target,
        "reference": reference,
        "f0": f0_path,
        "marker_cm": marker,
        "truth": truth,
    }


def output(path: Path) -> dict[str, object]:
    return {"name": path.name, "sha256": subject.sha256_file(path),
            "bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def split_receipts(root: Path, split: str, files: dict[str, Path]) -> dict[str, Path]:
    people = 24 if split == "FIT" else 8
    donor_role = "SOURCE_VALID" if split == "FIT" else "SOURCE_TEST"
    prefix = f"M34_R0_{split}"
    mosaic_target = {"sha256": ("1" if split == "FIT" else "2") * 64, "bytes": 101}
    mosaic = {
        "stage": "M34_NAM_EXPLORATORY_MOSAICS",
        "decision": "PASS_EXPLORATORY_MOSAICS_WITH_LOCAL_TRUTH",
        "scope": {"exploratory_only": True, "confirmatory_validation": False,
                  "generalizes_to_dnabr": False},
        "parameters": {
            "chromosome": "22", "rotation": 0, "target_prefix": prefix,
            "target_individuals": people,
            "seed": 1439610605 if split == "FIT" else 1702577247,
            "transition_parameterization": "pulse_generations",
            "transitions_per_morgan": 12.0,
            "mixture_proportions": {"AFR": 0.25, "EUR": 0.60, "NAM": 0.15},
        },
        "role_audit": {
            "donor_role": donor_role, "atomic_units_crossing_forbidden_roles": 0,
            "ref_train_used_as_donor": False,
            "source_test_used_as_donor": split == "VALID",
            "forbidden_roles": [
                "REF_TRAIN", "SOURCE_TEST" if split == "FIT" else "SOURCE_VALID"
            ],
            "unit_partition": "all", "unit_partition_rotation": 0,
            "selected_atomic_units": 15,
            "selected_people": 306 if split == "FIT" else 297,
            "donor_people_by_ancestry": (
                {"AFR": 141, "EUR": 151, "NAM": 14}
                if split == "FIT" else {"AFR": 135, "EUR": 151, "NAM": 11}
            ),
            "donor_atomic_units_by_ancestry": {"AFR": 5, "EUR": 8, "NAM": 2},
        },
        "counts": {"target_individuals": people},
        "inputs": {
            "phased_vcf": {"sha256": "7" * 64},
            "split_tsv": {"sha256": "8" * 64},
            "genetic_map": {"sha256": "9" * 64},
        },
        "outputs": {"m34_target.chr22.vcf.gz": mosaic_target},
    }
    reference_vcf = {"name": "m34_ref_train.chr22.vcf.gz", "sha256": "a" * 64,
                     "bytes": 200}
    target_vcf = {"name": "m34_target.chr22.vcf.gz", "sha256": "b" * 64,
                  "bytes": 150}
    sample_map = {"name": "m34_ref_train.sample_map.tsv", "sha256": "c" * 64,
                  "bytes": 50}
    bridge = {
        "schema_version": "m34_panel_factors_receipt_v1",
        "stage": "M34_EXPLORATORY_VCF_TO_FACTORS_BRIDGE",
        "decision": f"PASS_EXPLORATORY_PANEL_FACTORS_{donor_role}_MOSAICS",
        "scope": {"exploratory_only": True, "confirmatory_validation": False,
                  "generalizes_to_dnabr": False},
        "roles": {
            "reference_role": "REF_TRAIN", "frequency_role": "REF_TRAIN",
            "mosaic_donor_role_upstream": donor_role,
            "source_valid_panel_genotypes_opened": False,
            "source_test_panel_genotypes_opened": False,
            "source_test_open": split == "VALID",
            "source_test_mosaic_donors_upstream": split == "VALID",
        },
        "inputs": {
            "mosaic_vcf": mosaic_target,
            "panel_vcf": {"sha256": "7" * 64},
            "split_tsv": {"sha256": "8" * 64},
            "genetic_map": {"sha256": "9" * 64},
        },
        "outputs": [
            reference_vcf, target_vcf, sample_map,
            {**output(files["selected_variant"]), "name": "m34_selected_loci.npz"},
            {**output(files["target"]), "name": "m34_target_rare_diploid.npz"},
            {**output(files["reference"]), "name": "m34_reference_rare_summary.npz"},
        ],
        "counts": {
            "target_samples": people, "reference_samples": 753,
            "reference_samples_by_ancestry": {"AFR": 341, "EUR": 387, "NAM": 25},
            "split_biological_roles": {
                "REF_TRAIN": 753, "SOURCE_VALID": 306,
                "SOURCE_TEST": 297, "DISCOVERY": 149,
            },
        },
    }
    flare = {
        "schema_version": "1.0.0", "stage": "M34_AFR_EUR_NAM_FLARE",
        "status": "PASS_TRUTH_BLIND_FLARE", "claim_level": "exploratory",
        "chromosome": "22", "ancestry_names": ["AFR", "EUR", "NAM"],
        "shape": {"target_sample_count": people, "marker_count": 3},
        "input_sha256": {
            "reference_vcf": reference_vcf["sha256"],
            "target_vcf": target_vcf["sha256"],
            "sample_map": sample_map["sha256"],
            "genetic_map": "9" * 64,
        },
        "parameters": {
            "array": False, "probs": True, "em": True,
            "min-mac": 1, "min-maf": 0.0, "gen": 12.0,
            "update-p": False, "panel-probs": False,
            "seed": 3401103, "nthreads": 4,
        },
        "ancestry_vcf_audit": {
            "sample_count": people, "marker_count": 3,
            "sha256": ("d" if split == "FIT" else "e") * 64,
        },
        "truth_argument_available": False, "truth_accessed": False,
        "scoring_performed": False, "preflight_only": False,
    }
    return {
        "mosaic_receipt": write_json(root / f"{split}.mosaic.json", mosaic),
        "bridge_receipt": write_json(root / f"{split}.bridge.json", bridge),
        "flare_receipt": write_json(root / f"{split}.flare.json", flare),
    }


def fixture(root: Path) -> tuple[subject.SplitPaths, subject.SplitPaths]:
    loci = np.asarray([101, 202], dtype="<u8")
    reference = write_npz(root / "reference.shared.npz", {
        "ancestry": np.asarray([b"AFR", b"EUR", b"NAM"], dtype="|S4"),
        "locus_id": loci,
        "minor_ac": np.ones((3, 2), dtype="<u2"),
        "callable_an": np.full((3, 2), 20, dtype="<u2"),
        "minor_af": np.full((3, 2), 0.05, dtype="<f8"),
        "observed_mask": np.ones((3, 2), dtype="|u1"),
        "no_support": np.zeros((3, 2), dtype="|u1"),
    })
    result = []
    for split, people in (("FIT", 24), ("VALID", 8)):
        files = factor_files(root, split, people, reference)
        receipts = split_receipts(root, split, files)
        result.append(subject.SplitPaths(**files, **receipts))
    return result[0], result[1]


def mutate(path: Path, callback) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    callback(value)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class M34FactorizedManifestTests(unittest.TestCase):
    def test_known_answer_emits_manifest_consumed_by_current_trainer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fit, valid = fixture(root)
            manifest, receipt = subject.build(fit, valid, root)
            manifest_path = root / "manifest.json"
            receipt_path = root / "manifest.receipt.json"
            subject.write_outputs(manifest, receipt, manifest_path, receipt_path)

            loaded = trainer.load_manifest(manifest_path)
            self.assertEqual(loaded["rotation"], "R0")
            self.assertFalse(Path(manifest["splits"]["FIT"][0]["f0"]).is_absolute())
            self.assertEqual(set(loaded["splits"]), {"FIT", "VALID"})
            self.assertEqual(receipt["split_people"], {"FIT": 24, "VALID": 8})
            self.assertEqual(receipt["split_donor_roles"],
                             {"FIT": "SOURCE_VALID", "VALID": "SOURCE_TEST"})
            self.assertTrue(receipt["axes"]["fit_valid_samples_disjoint"])
            self.assertFalse(receipt["scientific_evidence"])

    def test_changed_fit_role_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            fit, valid = fixture(Path(raw))
            mutate(fit.mosaic_receipt,
                   lambda value: value["role_audit"].update(donor_role="SOURCE_TEST"))
            with self.assertRaisesRegex(ValueError, "donor role"):
                subject.build(fit, valid, Path(raw))

    def test_changed_generation_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            fit, valid = fixture(Path(raw))
            mutate(valid.mosaic_receipt,
                   lambda value: value["parameters"].update(transitions_per_morgan=17.0))
            with self.assertRaisesRegex(ValueError, "generation"):
                subject.build(fit, valid, Path(raw))

    def test_changed_root_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            fit, valid = fixture(Path(raw))
            mutate(fit.mosaic_receipt,
                   lambda value: value["parameters"].update(rotation=1))
            with self.assertRaisesRegex(ValueError, "root"):
                subject.build(fit, valid, Path(raw))

    def test_changed_hash_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            fit, valid = fixture(Path(raw))
            mutate(valid.bridge_receipt,
                   lambda value: value["outputs"][3].update(sha256="f" * 64))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                subject.build(fit, valid, Path(raw))

    def test_changed_flare_parameter_fails(self):
        changes = {
            "array": True, "probs": False, "em": False,
            "min-mac": 2, "min-maf": 0.01, "gen": 17.0,
            "update-p": True, "panel-probs": True,
            "seed": 1, "nthreads": 8,
        }
        for name, changed in changes.items():
            with self.subTest(parameter=name), tempfile.TemporaryDirectory() as raw:
                fit, valid = fixture(Path(raw))
                mutate(fit.flare_receipt,
                       lambda value: value["parameters"].update({name: changed}))
                with self.assertRaisesRegex(ValueError, "FLARE parameters"):
                    subject.build(fit, valid, Path(raw))

    def test_changed_cross_split_source_hash_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            fit, valid = fixture(Path(raw))
            mutate(valid.mosaic_receipt,
                   lambda value: value["inputs"]["phased_vcf"].update(sha256="0" * 64))
            mutate(valid.bridge_receipt,
                   lambda value: value["inputs"]["panel_vcf"].update(sha256="0" * 64))
            with self.assertRaisesRegex(ValueError, "sources differ"):
                subject.build(fit, valid, Path(raw))

    def test_manifest_rejects_factors_outside_its_bundle(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fit, valid = fixture(root)
            bundle = root / "separate" / "bundle"
            with self.assertRaisesRegex(ValueError, "outside the co-staged"):
                subject.build(fit, valid, bundle)

    def test_overlapping_fit_valid_sample_axis_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fit, valid = fixture(root)
            with np.load(fit.target, allow_pickle=False) as archive:
                fit_keys = archive["sample_key_sha256"][:8]
            for path, sample_member in ((valid.target, "sample_key_sha256"),
                                        (valid.f0, "sample_key_sha256"),
                                        (valid.truth, "sample_key_sha256")):
                with np.load(path, allow_pickle=False) as archive:
                    values = {name: archive[name] for name in archive.files}
                values[sample_member] = fit_keys
                np.savez(path, **values)
            mutate(valid.bridge_receipt, lambda value: [
                row.update(sha256=subject.sha256_file(valid.target),
                           bytes=valid.target.stat().st_size)
                for row in value["outputs"] if row.get("name") == "m34_target_rare_diploid.npz"
            ])
            with self.assertRaisesRegex(ValueError, "overlap"):
                subject.build(fit, valid, root)


if __name__ == "__main__":
    unittest.main()
