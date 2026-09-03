from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m36_cora_audit_materialization", ROOT / "bin/m36_cora_audit_materialization.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class M36MaterializationAuditTests(unittest.TestCase):
    def test_balanced_component_disjoint_materialization_is_trainable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples = [f"S{index}" for index in range(9)]
            covariates = [{
                "sample_id": sample, "cohort": f"C{index % 2}",
                "rare_burden": str(index + 1), "rare_callability": "0.99",
                "Q_AFR": "0.2", "Q_EUR": "0.5", "Q_NAM": "0.3", "Q_EAS": "0",
            } for index, sample in enumerate(samples)]
            components = [{"sample_id": sample, "pcrelate_component": f"P{index}"}
                          for index, sample in enumerate(samples)]
            targets = []
            for left, right in combinations(samples, 2):
                left_index, right_index = samples.index(left), samples.index(right)
                value = "1" if (left_index + right_index) % 2 else "0"
                targets.append({
                    "sample_i": left, "sample_j": right,
                    "target_chrom": "outside_chr22_total",
                    "target_source": "asibd_refined_ibd_gnomix_stratified_exploratory",
                    "target_stratum": "between_component", "target_cm": value,
                    "target_positive": value, "target": value,
                })
            paths = {
                "covariates": root / "covariates.tsv",
                "components": root / "components.tsv",
                "targets": root / "targets.tsv",
            }
            write_tsv(paths["covariates"], covariates)
            write_tsv(paths["components"], components)
            write_tsv(paths["targets"], targets)
            descriptors = {
                name: {"uri": path.name, "generation": "TEST",
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in paths.items()
            }
            descriptors.update({name: {"uri": name, "generation": "TEST", "sha256": "0" * 64}
                                for name in ("loci", "carriers", "missing")})
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "stage": "M36_CORA_MATERIALIZE", "status": "MATERIALIZED_PASS",
                "synthetic": False, "input_descriptors": descriptors,
            }), encoding="utf-8")
            result = AUDIT.audit(receipt, paths["covariates"], paths["components"],
                                 paths["targets"], 3)
            self.assertEqual(result["status"], "PASS_TRAINABLE")
            self.assertEqual(result["n_pcrelate_components"], 9)
            self.assertGreater(result["target_partition_coverage"]["validation_covered_pairs"], 0)

    def test_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {name: root / f"{name}.tsv" for name in ("covariates", "components", "targets")}
            for path in paths.values():
                path.write_text("x\n", encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "stage": "M36_CORA_MATERIALIZE", "status": "MATERIALIZED_PASS",
                "synthetic": False,
                "input_descriptors": {name: {"sha256": "0" * 64} for name in paths},
            }), encoding="utf-8")
            with self.assertRaisesRegex(AUDIT.ContractError, "hash differs"):
                AUDIT.audit(receipt, paths["covariates"], paths["components"],
                            paths["targets"], 3)


if __name__ == "__main__":
    unittest.main()
