"""Device report reconciliation on synthetic JSON, without torch or genotypes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import m39_compare_device_profiles as C


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def fixture(self, name, device, batch=2, policy="grouped", seconds=1.0):
        path = self.root / device.replace(":", "-") / name / "profile.json"
        pairs = [[i, i // 2] for i in range(32)]
        sample = {"anchor_samples": [{"pairs": pairs[i:i + 2]} for i in range(0, 32, 2)],
                  "strata": [{"pair_count": 7920} for _ in range(4)]}
        self.write(path.with_name("sampling-manifest.json"), sample)
        rows = [{"step": 0, "warmup": True, "pairs": [[32, 659]],
                 "stratum": None, "end_to_end_seconds": 9999.0}]
        for i in range(0, 32, batch):
            rows.append({"step": len(rows), "warmup": False, "pairs": pairs[i:i + batch],
                         "stratum": i // 8 if policy == "grouped" else None,
                         "end_to_end_seconds": seconds})
        for row in rows:
            self.write(path.parent / "steps" / f"step-{row['step']:04d}.json", row)
        report = {"decision": "PASS_ORDERED_THROUGHPUT_TECHNICAL_ONLY",
                  "case": {"id": name, "batch_size": batch, "policy": policy},
                  "runtime": {"device": device, "torch": "2.12.1+cu126"},
                  "architecture": {"width": 32}, "parameter_count": 10, "store_shape": [48, 660, 8],
                  "sampling_manifest_sha256": C.digest(path.with_name("sampling-manifest.json")),
                  "scope": {"biological_training": False, "truth_opened": False}, "steps": rows,
                  "summary": {"measured_end_to_end_seconds": seconds * 32 / batch,
                              "grouped_fullpass_estimate": None if policy == "random" else {
                                  "seconds": seconds * 31680 / batch}},
                  "provenance": {"parent_sha256": "a", "manifest_sha256": "b", "folds_sha256": "c",
                                 "source_sha256": {"m39_ordered_models.py": "d"}}}
        if device == "cuda:0":
            report["device_parity"] = {
                "decision": "PASS_CPU_DEVICE_WARMUP_GRADIENT_PARITY", "relative_tolerance_reference": "cpu",
                "max_tolerance_ratio_by_tensor": {"logits": 0.25, "input": 0.1}}
            self.write(path.with_name("device-parity.json"), report["device_parity"])
            report["resources"] = {"device_memory": {"peak_reserved_bytes": 1024**3}}
        self.write(path, report)
        return path, report

    def test_recomputes_weighted_time_excluding_warmup(self):
        path, _ = self.fixture("one", "cpu", seconds=2)
        _, value = C.inspect(path, "cpu")
        self.assertEqual(value["sample_seconds"], 32)
        self.assertAlmostEqual(value["pass_hours"], 31680 / 3600)

    def test_rejects_primary_step_mismatch_and_missing_report(self):
        path, report = self.fixture("one", "cpu")
        report["steps"][1]["end_to_end_seconds"] = 12
        self.write(path, report)
        with self.assertRaisesRegex(ValueError, "step disagrees"):
            C.inspect(path, "cpu")

    def test_rejects_failed_or_empty_parity(self):
        path, original = self.fixture("one", "cuda:0")
        for value in ({}, {"logits": 1.01}, {"logits": float("nan")}):
            report = copy.deepcopy(original)
            report["device_parity"]["max_tolerance_ratio_by_tensor"] = value
            self.write(path, report)
            self.write(path.with_name("device-parity.json"), report["device_parity"])
            with self.assertRaises(ValueError):
                C.inspect(path, "cuda:0")

    def test_rejects_changed_sampling_and_population_estimate_for_random(self):
        path, report = self.fixture("one", "cpu", policy="random")
        self.assertIsNone(C.inspect(path, "cpu")[1]["pass_hours"])
        report["summary"]["grouped_fullpass_estimate"] = {"seconds": 1}
        self.write(path, report)
        with self.assertRaisesRegex(ValueError, "random replay"):
            C.inspect(path, "cpu")

    def test_six_matched_cases_and_model_drift(self):
        for i in range(6):
            self.fixture(str(i), "cpu", seconds=4)
            gpu_path, report = self.fixture(str(i), "cuda:0", seconds=1)
        result = C.compare(self.root / "cpu", self.root / "cuda-0")
        self.assertEqual(len(result["rows"]), 6)
        self.assertTrue(all(r["observed_cpu_gpu_time_ratio"] == 4 for r in result["rows"]))
        report["provenance"]["source_sha256"]["m39_ordered_models.py"] = "changed"
        self.write(gpu_path, report)
        with self.assertRaisesRegex(ValueError, "model changed"):
            C.compare(self.root / "cpu", self.root / "cuda-0")


if __name__ == "__main__":
    unittest.main()
