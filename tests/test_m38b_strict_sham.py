from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m33_safe_bridge_core import write_deterministic_npz  # noqa: E402
from m38b_strict_sham import make_strict_sham  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M38BStrictShamTests(unittest.TestCase):
    def fixture(self, root: Path, tamper: bool = False) -> tuple[Path, Path]:
        source = root / "reference.npz"
        loci = 123
        ac = np.vstack((np.arange(loci) % 3, np.arange(loci) % 4, np.arange(loci) % 5)).astype("<u2")
        an = np.full((3, loci), 20, dtype="<u2")
        af = ac.astype(float) / an
        if tamper:
            af[0, 0] += 0.01
        write_deterministic_npz(source, {
            "ancestry": np.asarray([b"AFR", b"EUR", b"NAM"], dtype="|S4"),
            "locus_id": np.arange(loci, dtype="<u8"),
            "minor_ac": ac, "callable_an": an, "minor_af": af,
            "observed_mask": (an > 0).astype("|u1"),
            "no_support": ((an > 0) & (ac == 0)).astype("|u1"),
        })
        receipt = root / "reference.receipt.json"
        receipt.write_text(json.dumps({
            "stage": "M38B_APPLY_FROZEN_LOO_PRIMARY_MASK",
            "decision": "PASS_PRIMARY_FACTORS_FROZEN_FOR_MODEL",
            "counts": {"primary_loci": loci},
            "outputs": {source.name: {"sha256": digest(source)}},
        }), encoding="utf-8")
        return source, receipt

    def test_derived_frequency_fields_are_recomputed_after_derangement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, receipt = self.fixture(root)
            output, audit = root / "sham.npz", root / "sham.receipt.json"
            make_strict_sham(source, receipt, output, audit, 3401103)
            with np.load(output, allow_pickle=False) as archive:
                ac, an = archive["minor_ac"], archive["callable_an"]
                np.testing.assert_allclose(archive["minor_af"], ac / an, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(archive["observed_mask"], an > 0)
                np.testing.assert_array_equal(archive["no_support"], (an > 0) & (ac == 0))

    def test_inconsistent_source_derived_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, receipt = self.fixture(root, tamper=True)
            with self.assertRaisesRegex(ValueError, "derived frequency fields"):
                make_strict_sham(source, receipt, root / "sham.npz",
                                 root / "sham.receipt.json", 3401103)


if __name__ == "__main__":
    unittest.main()
