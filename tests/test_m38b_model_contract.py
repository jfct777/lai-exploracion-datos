#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m38b_validate_model_contract import (  # noqa: E402
    LOAD_BEARING_SOURCE_NAMES, M38BModelContractError, validate,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M38BModelContractTests(unittest.TestCase):
    @staticmethod
    def sources() -> list[Path]:
        locations = (ROOT / "bin", ROOT / "conf", ROOT / "modules", ROOT / "workflows")
        found = {path.name: path for location in locations for path in location.iterdir()
                 if path.name in LOAD_BEARING_SOURCE_NAMES}
        return [found[name] for name in sorted(LOAD_BEARING_SOURCE_NAMES)]

    def test_canonical_contract_and_amendment_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "receipt.json"
            contract = ROOT / "conf/m38b_r0_oof_contract.json"
            amendment = ROOT / "conf/m38b_r0_oof_amendment_1.json"
            amendment_2 = ROOT / "conf/m38b_r0_oof_amendment_2.json"
            observed = validate(contract, amendment, amendment_2,
                                digest(contract), digest(amendment), digest(amendment_2), output,
                                self.sources())
            self.assertEqual(observed["status"], "PASS_BASE_AND_PRE_OUTCOME_AMENDMENT_BOUND")
            self.assertEqual(observed["base_contract_sha256"], digest(contract))
            self.assertEqual(observed["amendment_sha256"], digest(amendment))
            self.assertEqual(observed["amendment_2_sha256"], digest(amendment_2))
            self.assertEqual(observed["source_binding"],
                             "DETERMINISTIC_LOAD_BEARING_SOURCE_MANIFEST")
            self.assertEqual(len(observed["source_manifest"]), 28)
            self.assertEqual(len(observed["source_manifest_sha256"]), 64)

    def test_declared_python_sources_cover_local_import_closure(self) -> None:
        """Every local transitive import must be staged and authenticated."""
        declared = {name for name in LOAD_BEARING_SOURCE_NAMES if name.endswith(".py")}
        closure = set(declared)
        frontier = list(declared)
        while frontier:
            name = frontier.pop()
            parsed = ast.parse((ROOT / "bin" / name).read_text(encoding="utf-8"))
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module.split(".", 1)[0]]
                else:
                    continue
                for module in modules:
                    dependency = f"{module}.py"
                    if (ROOT / "bin" / dependency).is_file() and dependency not in closure:
                        closure.add(dependency)
                        frontier.append(dependency)
        self.assertEqual(closure, declared, f"unstaged local imports: {sorted(closure - declared)}")

    def test_production_entrypoints_import_from_staged_sources_only(self) -> None:
        """Mirror the isolated staged/bin environment used by Google Batch."""
        with tempfile.TemporaryDirectory(prefix="m38b-import-smoke-") as raw:
            staged = Path(raw) / "staged" / "bin"
            staged.mkdir(parents=True)
            python_sources = sorted(
                name for name in LOAD_BEARING_SOURCE_NAMES if name.endswith(".py")
            )
            for name in python_sources:
                shutil.copy2(ROOT / "bin" / name, staged / name)
            entrypoints = (
                "m38b_validate_model_contract", "m38b_subset_factors",
                "m38b_bind_marker_axis", "m38b_strict_sham", "m38b_materialize_arm",
                "m38b_make_folds", "m38b_positive_control", "m38b_partition_fold",
                "m38b_train_fold", "m38b_collect_oof", "m38b_pack_scoring",
                "m38b_score_oof", "m38b_score_positive", "m38b_decide",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(staged)
            environment["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run(
                [sys.executable, "-c", ";".join(f"import {name}" for name in entrypoints)],
                cwd=staged,
                env=environment,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tampered_amendment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = ROOT / "conf/m38b_r0_oof_contract.json"
            amendment = root / "amendment.json"
            document = json.loads(
                (ROOT / "conf/m38b_r0_oof_amendment_1.json").read_text(encoding="utf-8")
            )
            document["delta"]["arm_binding"] = "disabled"
            amendment.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(M38BModelContractError, "amendment differs"):
                canonical_2 = ROOT / "conf/m38b_r0_oof_amendment_2.json"
                validate(contract, amendment, canonical_2, digest(contract), digest(amendment),
                         digest(canonical_2),
                         root / "receipt.json", self.sources())

    def test_omitted_or_tampered_source_set_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = ROOT / "conf/m38b_r0_oof_contract.json"
            amendment = ROOT / "conf/m38b_r0_oof_amendment_1.json"
            amendment_2 = ROOT / "conf/m38b_r0_oof_amendment_2.json"
            with self.assertRaisesRegex(M38BModelContractError, "source set differs"):
                validate(contract, amendment, amendment_2, digest(contract), digest(amendment),
                         digest(amendment_2), root / "receipt.json", self.sources()[:-1])


if __name__ == "__main__":
    unittest.main()
