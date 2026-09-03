#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
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
            self.assertEqual(len(observed["source_manifest"]), 27)
            self.assertEqual(len(observed["source_manifest_sha256"]), 64)

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
