import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from m38_stratify_rare_loci import run, validate_exact_locus_partition  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_inputs(root: Path) -> dict[str, Path]:
    loci = 660
    locus_id = np.arange(10_000, 10_000 + loci, dtype=np.uint64)
    selected = root / "m34_selected_loci.npz"
    np.savez(
        selected, alt=np.full(loci, b"G", dtype="S1"),
        cM=np.linspace(1.0, 70.0, loci), chrom=np.full(loci, 22, dtype=np.uint8),
        locus_id=locus_id, pos=np.arange(16_000_000, 16_000_000 + loci, dtype=np.int64),
        ref=np.full(loci, b"A", dtype="S1"),
    )
    an = np.vstack((np.full(loci, 100), np.full(loci, 100), np.full(loci, 50))).astype(np.uint16)
    ac = np.zeros((3, loci), dtype=np.uint16)
    ac[2, 0] = 5
    ac[0, 1] = 3
    ac[1, 2] = 4
    af = ac / an
    reference = root / "m34_reference_rare_summary.npz"
    np.savez(
        reference, ancestry=np.asarray([b"AFR", b"EUR", b"NAM"]),
        callable_an=an, locus_id=locus_id, minor_ac=ac, minor_af=af,
        no_support=(ac == 0).astype(np.uint8), observed_mask=np.ones_like(ac, dtype=np.uint8),
    )
    fields = [
        "chrom", "position", "ref", "alt", "pooled_minor_ac",
        "pooled_callable_an", "pooled_maf",
        *(f"{ancestry}_{suffix}" for ancestry in ("AFR", "EUR", "NAM")
          for suffix in ("minor_ac", "callable_an", "minor_af", "carrier_people",
                         "carrier_populations", "carrier_units",
                         "max_unit_carrier_share", "unit_hhi")),
    ]
    audit = root / "m34_rare_loci.audit.tsv"
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for locus in range(loci):
            row = {
                "chrom": 22, "position": 16_000_000 + locus, "ref": "A", "alt": "G",
                "pooled_minor_ac": int(ac[:, locus].sum()),
                "pooled_callable_an": int(an[:, locus].sum()),
                "pooled_maf": float(ac[:, locus].sum() / an[:, locus].sum()),
            }
            for index, ancestry in enumerate(("AFR", "EUR", "NAM")):
                row[f"{ancestry}_minor_ac"] = int(ac[index, locus])
                row[f"{ancestry}_callable_an"] = int(an[index, locus])
                row[f"{ancestry}_minor_af"] = float(af[index, locus])
                row[f"{ancestry}_carrier_people"] = 3 if ac[index, locus] else 0
                row[f"{ancestry}_carrier_populations"] = 2 if ac[index, locus] else 0
                row[f"{ancestry}_carrier_units"] = 3 if ac[index, locus] else 0
                row[f"{ancestry}_max_unit_carrier_share"] = 1 / 3 if ac[index, locus] else 0
                row[f"{ancestry}_unit_hhi"] = 1 / 3 if ac[index, locus] else 0
            writer.writerow(row)
    summary = root / "m34_rare_loci.audit.summary.json"
    summary.write_text(json.dumps({
        "stage": "M34_RARE_LOCUS_DISTRIBUTION_AUDIT",
        "status": "PASS_DESCRIPTIVE_AUDIT_NO_MODEL_SELECTION",
        "scope": {"frequency_population": "REF_TRAIN_only", "target_mosaics_read": False,
                  "local_ancestry_truth_read": False, "predictions_read": False,
                  "king_used": False},
        "selection": {"selected_loci": loci},
        "outputs": {"per_locus_tsv_sha256": digest(audit)},
    }, sort_keys=True), encoding="utf-8")
    return {"selected": selected, "reference": reference, "audit": audit, "summary": summary}


def arguments(inputs: dict[str, Path], output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        selected=inputs["selected"], reference=inputs["reference"],
        audit_tsv=inputs["audit"], audit_summary=inputs["summary"],
        selected_sha256=digest(inputs["selected"]), reference_sha256=digest(inputs["reference"]),
        audit_tsv_sha256=digest(inputs["audit"]), audit_summary_sha256=digest(inputs["summary"]),
        expected_loci=660, expected_ancestry_an="AFR=100,EUR=100,NAM=50",
        beta_priors="0.5,1.0", rare_af_cutoff=0.01,
        q_top_thresholds="0.8,0.9,0.95", q_rare_thresholds="0.5,0.8,0.95",
        unit_thresholds="2,3", q_top_draws=4096, seed=3801103,
        f0_contains_selected_rare_loci=True,
        f0_overlap_assertion_source="FIXTURE_CONTRACT",
        output_tsv=output / "m38_rare_locus_strata.tsv",
        output_npz=output / "m38_rare_locus_strata.npz",
        output_summary=output / "m38_rare_locus_stratification.summary.json",
        output_receipt=output / "m38_rare_locus_stratification.receipt.json",
    )


class M38StratificationTest(unittest.TestCase):
    def test_ref_train_only_outputs_and_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_inputs(root)
            args = arguments(inputs, root / "out")
            summary = run(args)
            self.assertEqual(summary["scope"]["loci"], 660)
            self.assertFalse(summary["scope"]["target_read"])
            self.assertFalse(summary["scope"]["local_ancestry_truth_read"])
            self.assertFalse(summary["scope"]["F0_predictions_read"])
            self.assertTrue(summary["contractual_assertions"]["M37_F0_CONTAINS_SELECTED_RARE_LOCI"])
            self.assertFalse(summary["incremental_value_estimated"])
            self.assertEqual(summary["sweeps"]["mask_counts"]
                             ["ANCHOR_OBS_NAM_AF_GE_0P05_AFR_EUR_LT_0P01"], 1)
            self.assertEqual(summary["M38B_pending_control"], "leave_one_atomic_unit_out")
            with np.load(args.output_npz, allow_pickle=False) as archive:
                self.assertEqual(archive["locus_id"].shape, (660,))
                self.assertTrue(np.allclose(archive["q_top_prior_0p5"].sum(axis=0), 1.0))
                self.assertIn("NAM_UNRESOLVED", archive["nam_status_prior_0p5"])
            receipt = json.loads(args.output_receipt.read_text())
            self.assertFalse(receipt["selection_used_target_truth_F0_or_scores"])

    def test_deterministic_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_inputs(root)
            first = arguments(inputs, root / "one")
            second = arguments(inputs, root / "two")
            run(first)
            run(second)
            self.assertEqual(digest(first.output_tsv), digest(second.output_tsv))
            self.assertEqual(digest(first.output_npz), digest(second.output_npz))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run(first)

    def test_rejects_hash_and_axis_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_inputs(root)
            bad_hash = arguments(inputs, root / "hash")
            bad_hash.selected_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                run(bad_hash)
            with np.load(inputs["reference"], allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["locus_id"] = arrays["locus_id"].copy()
            arrays["locus_id"][0] += 1
            np.savez(inputs["reference"], **arrays)
            bad_axis = arguments(inputs, root / "axis")
            with self.assertRaisesRegex(ValueError, "locus axes"):
                run(bad_axis)

    def test_accepts_nextflow_staged_symlinks_with_authenticated_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            inputs = make_inputs(source_root)
            staged = root / "staged"
            staged.mkdir()
            linked_inputs: dict[str, Path] = {}
            for name, source in inputs.items():
                link = staged / source.name
                link.symlink_to(source)
                linked_inputs[name] = link
            args = arguments(linked_inputs, root / "out")
            summary = run(args)
            self.assertEqual(summary["status"],
                             "PASS_REF_TRAIN_ONLY_DESCRIPTIVE_STRATIFICATION")

    def test_workflow_is_personal_bucket_and_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        module = (root / "modules/38_RARE_LOCUS_STRATIFICATION.nf").read_text()
        workflow = (root / "workflows/m38_rare_locus_stratification.nf").read_text()
        config = (root / "conf/m38_rare_locus_stratification.config").read_text()
        self.assertIn("overwrite: false", module)
        self.assertIn("team: 'frank'", config)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", config)
        self.assertIn("--network none", config)
        self.assertIn("m38_f0_contains_selected_rare_loci", workflow)
        self.assertNotIn("TARGET", workflow)
        self.assertNotIn("truth", workflow.lower())

    def test_m38b_exact_partition_rejects_rare_common_overlap(self) -> None:
        result = validate_exact_locus_partition(
            np.asarray([1, 2, 3, 4]), np.asarray([1, 2]), np.asarray([3, 4]))
        self.assertEqual(result["overlap_loci"], 0)
        with self.assertRaisesRegex(ValueError, "intersects F_common"):
            validate_exact_locus_partition(
                np.asarray([1, 2, 3, 4]), np.asarray([1, 2, 3]), np.asarray([3, 4]))


if __name__ == "__main__":
    unittest.main()
