import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from m32_locus_contract import EXPECTED_RADII  # noqa: E402
from m32_locus_occupancy import context_counts, occupancy_report, read_positions, validate_positions  # noqa: E402
import m32_locus_smoke as smoke  # noqa: E402
from m32_locus_tensor import (  # noqa: E402
    apply_phase_switches,
    build_ordered_sequence,
    pad_locus_axis,
    pad_ragged_indices,
    permute_reference_labels,
    phase_aware_minor_presence,
    primary_diploid_channels,
    ragged_context_indices,
    swap_homologues,
)


class M32LocusSequenceSmokeTest(unittest.TestCase):
    def test_minor_dosage_missingness_and_phase_invariance(self):
        minor = [0, 1, 1]
        h = [[[0, 0], [0, 1], [-1, 1]], [[1, 0], [1, 1], [0, 0]]]
        primary = primary_diploid_channels(h, minor)
        self.assertEqual(primary["minor_dosage"][0][:2], [2, 1])
        self.assertIsNone(primary["minor_dosage"][0][2])
        self.assertFalse(primary["callable_mask"][0][2])
        swapped = primary_diploid_channels(swap_homologues(h), minor)
        self.assertEqual(primary, swapped)
        self.assertNotEqual(phase_aware_minor_presence(h, minor), phase_aware_minor_presence(swap_homologues(h), minor))
        switched = apply_phase_switches(h, [[True, False, True], [False, True, False]])
        self.assertEqual(primary, primary_diploid_channels(switched, minor))
        self.assertNotEqual(phase_aware_minor_presence(h, minor), phase_aware_minor_presence(switched, minor))

    def test_reference_label_sham_is_diploid_count_preserving_and_deterministic(self):
        labels = ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"]
        first = permute_reference_labels(labels, 17)
        second = permute_reference_labels(labels, 17)
        other = permute_reference_labels(labels, 18)
        self.assertEqual(first, second)
        self.assertNotEqual(first, labels)
        self.assertNotEqual(first, other)
        self.assertEqual(sorted(first), sorted(labels))

    def test_occupancy_uses_symmetric_radius_and_does_not_drop_grid(self):
        grid = [0.0, 0.1, 0.2, 0.4]
        rare = [0.05, 0.11, 0.39]
        self.assertEqual(context_counts(grid, rare, 0.05), [1, 2, 0, 1])
        report = occupancy_report(grid, rare, EXPECTED_RADII)
        self.assertEqual(report["grid_marker_count"], 4)
        self.assertFalse(report["selects_radius"])
        self.assertEqual([row["radius_cm"] for row in report["contexts"]], list(EXPECTED_RADII))

    def test_occupancy_accepts_cm_ties_but_rejects_bad_coordinates(self):
        self.assertEqual(validate_positions([0.1, 0.1, 0.2], "ties"), [0.1, 0.1, 0.2])
        for values in ([0.2, 0.1], [0.1, float("nan")]):
            with self.assertRaises(ValueError):
                validate_positions(values, "bad")
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.tsv"
            valid.write_text("chrom\tbp\tcm\nchr22\t100\t0.1\nchr22\t101\t0.1\n")
            self.assertEqual(read_positions(valid), [0.1, 0.1])
            invalid = Path(tmp) / "invalid.tsv"
            invalid.write_text("chrom\tbp\tcm\nchr22\t100\t0.1\nchr22\t100\t0.1\n")
            with self.assertRaisesRegex(ValueError, "bp"):
                read_positions(invalid)

    def test_padding_preserves_interior(self):
        values = [[0, 1, 2, 3], [4, 5, 6, 7]]
        padded = pad_locus_axis(values, 2, fill=-1)
        self.assertEqual(values, [person[2:-2] for person in padded])

    def test_ordered_tensor_and_ragged_mask_preserve_full_identity(self):
        target = [[[0, 1], [1, 1]], [[0, 0], [0, 1]]]
        reference = [
            [[0, 0], [1, 1]], [[0, 1], [1, 0]],
            [[1, 1], [0, 0]], [[1, 0], [0, 1]],
            [[0, 1], [1, 1]], [[1, 1], [0, 1]],
        ]
        flare = [[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]]
        result = build_ordered_sequence(
            ["m0", "m1"], [100, 200], [0.1, 0.1], flare,
            ["r0", "r1"], [110, 120], [0.1, 0.1], [0, 1],
            target, reference, ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"],
        )
        self.assertEqual(result["grid"]["marker_id"], ["m0", "m1"])
        self.assertEqual(result["rare_sequence"]["locus_id"], ["r0", "r1"])
        rows = ragged_context_indices([0.1, 0.2], [0.1, 0.1, 0.25], 0.1)
        padded, mask = pad_ragged_indices(rows)
        self.assertEqual(rows, [[index for index, keep in zip(row, row_mask) if keep] for row, row_mask in zip(padded, mask)])

    @staticmethod
    def _source_arguments():
        relative = [
            "bin/m32_locus_contract.py", "bin/m32_locus_tensor.py",
            "bin/m32_locus_occupancy.py", "bin/m32_locus_smoke.py",
            "conf/m32_locus_sequence_smoke_preregistration.json",
            "conf/m32_locus_sequence_smoke.config",
            "modules/32_LOCUS_SEQUENCE_SMOKE.nf",
            "workflows/m32_locus_sequence_smoke.nf",
        ]
        return [item for path in relative for item in ("--source", f"{path}={ROOT / path}")]

    def test_smoke_is_deterministic_and_fails_on_overwrite(self):
        script = ROOT / "bin" / "m32_locus_smoke.py"
        contract = ROOT / "conf" / "m32_locus_sequence_smoke_preregistration.json"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
            commands = []
            for out, run_id in ((base / "a", "same"), (base / "b", "same")):
                commands.append([
                    sys.executable, str(script), "--preregistration", str(contract),
                    "--run-id", run_id, "--seed", "17", "--git-commit", head,
                    "--repository-root", str(ROOT), *self._source_arguments(), "--outdir", str(out),
                ])
            for command in commands:
                subprocess.run(command, check=True)
            names = ["m32_locus_sequence.occupancy_and_invariants.json", "m32_locus_sequence.provenance.json", "m32_locus_sequence.manifest.json", "m32_locus_sequence.receipt.json"]
            for name in names:
                self.assertEqual((base / "a" / "same" / name).read_bytes(), (base / "b" / "same" / name).read_bytes())
            result = subprocess.run(commands[0], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_smoke_cli_has_no_truth_argument(self):
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        result = subprocess.run([
            sys.executable, str(ROOT / "bin" / "m32_locus_smoke.py"),
            "--preregistration", str(ROOT / "conf" / "m32_locus_sequence_smoke_preregistration.json"),
            "--run-id", "forbidden", "--git-commit", head,
            "--repository-root", str(ROOT), *self._source_arguments(),
            "--outdir", "/tmp", "--truth", "forbidden",
        ], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_run_id_and_source_set_fail_closed(self):
        for value in ("", "../bad", "bad space", "x" * 129):
            self.assertIsNone(smoke.RUN_ID_PATTERN.fullmatch(value))
        with self.assertRaisesRegex(ValueError, "complete M32"):
            smoke.parse_source_specs(["bin/m32_locus_smoke.py=/tmp/x"])


if __name__ == "__main__":
    unittest.main()
