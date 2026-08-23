#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_f0_sanitize", ROOT / "bin" / "m33_f0_sanitize.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_inputs(directory: Path, *, second_sum: str = "0.1,0.2,0.7",
                 samples: tuple[str, ...] = ("T1", "T2"), duplicate: bool = False,
                 hard_calls: tuple[str, str] = ("99", "-8")) -> tuple[Path, Path]:
    vcf = directory / "flare.anc.vcf"
    rows = [
        (100, "A", "C", "0.7,0.2,0.1", second_sum),
        (200, "G", "T", "0.2,0.3,0.5", "0.3,0.3,0.4"),
    ]
    if duplicate:
        rows.append(rows[-1])
    with vcf.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##ANCESTRY=<AFR=0,EUR=1,ASIA=2>\n")
        handle.write('##FORMAT=<ID=ANP1,Number=3,Type=Float,Description="first">\n')
        handle.write('##FORMAT=<ID=ANP2,Number=3,Type=Float,Description="second">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for pos, ref, alt, p1, p2 in rows:
            values = [f"1|1:{hard_calls[0]}:{hard_calls[1]}:{p1}:{p2}" for _ in samples]
            handle.write(f"22\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT:AN1:AN2:ANP1:ANP2\t" + "\t".join(values) + "\n")
    target = directory / "target.npz"
    keys = np.asarray([MODULE.core.sample_key(sample) for sample in samples], dtype="|S64")
    np.savez(target, sample_key_sha256=keys, minor_dosage=np.zeros((len(samples), 1), dtype="|i1"))
    return vcf, target


class SanitizedF0Tests(unittest.TestCase):
    def test_source_auth_binds_exact_commit_inventory_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")
            relative = "bin/source.py"
            commit = "1" * 40
            auth = root / "source_auth.json"
            auth.write_text(json.dumps({
                "schema_version": "1.0.0",
                "stage": "M33_F0_SANITIZE_SOURCE_AUTH",
                "status": "AUTHORIZED_EXACT_SANITIZER_SOURCES",
                "implementation_commit": commit,
                "files": {relative: hashlib.sha256(source.read_bytes()).hexdigest()},
            }), encoding="utf-8")
            observed = MODULE.validate_source_auth(auth, {relative: source}, commit)
            self.assertEqual(observed, hashlib.sha256(auth.read_bytes()).hexdigest())
            source.write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                MODULE.validate_source_auth(auth, {relative: source}, commit)

    def test_probability_only_artifact_is_reopened_and_axes_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            vcf, target = write_inputs(root)
            output = root / "out"
            output.mkdir()
            receipt = MODULE.run(vcf, target, output, 386357765, "a" * 64)
            self.assertEqual(receipt["status"], "PASS_PROBABILITY_ONLY_REOPENED")
            self.assertEqual(receipt["sample_count"], 2)
            self.assertEqual(receipt["marker_count"], 2)
            self.assertFalse(receipt["contains_raw_genotypes"])
            with np.load(output / "flare_f0_sanitized.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {
                    "sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt", "F0"
                })
                self.assertEqual(archive["F0"].shape, (2, 2, 2, 3))
                self.assertEqual(archive["F0"].dtype, np.dtype("<f4"))
                np.testing.assert_allclose(archive["F0"].sum(axis=3), 1.0, atol=5e-6, rtol=0)

    def test_hard_calls_and_gt_do_not_affect_output(self) -> None:
        artifacts = []
        for hard in (("0", "1"), ("99", "-8")):
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                vcf, target = write_inputs(root, hard_calls=hard)
                output = root / "out"
                output.mkdir()
                MODULE.run(vcf, target, output, 386357765, "b" * 64)
                with np.load(output / "flare_f0_sanitized.npz", allow_pickle=False) as archive:
                    artifacts.append(np.asarray(archive["F0"]).copy())
        np.testing.assert_array_equal(artifacts[0], artifacts[1])

    def test_raw_sample_ids_do_not_appear_in_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            vcf, target = write_inputs(root)
            output = root / "out"
            output.mkdir()
            MODULE.run(vcf, target, output, 386357765, "c" * 64)
            payload = b"".join(path.read_bytes() for path in output.iterdir())
            self.assertNotIn(b"T1", payload)
            self.assertNotIn(b"T2", payload)

    def test_sample_order_mismatch_fails_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            vcf, target = write_inputs(root)
            with np.load(target, allow_pickle=False) as archive:
                keys = archive["sample_key_sha256"][::-1]
            target.unlink()
            np.savez(target, sample_key_sha256=keys)
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "sample axes differ"):
                MODULE.run(vcf, target, output, 1, "d" * 64)
            self.assertEqual(list(output.iterdir()), [])

    def test_duplicate_marker_and_bad_probability_fail(self) -> None:
        for kwargs in ({"duplicate": True}, {"second_sum": "0.1,0.2,0.1"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                vcf, target = write_inputs(root, **kwargs)
                output = root / "out"
                output.mkdir()
                with self.assertRaises(ValueError):
                    MODULE.run(vcf, target, output, 1, "e" * 64)
                self.assertEqual(list(output.iterdir()), [])

    def test_output_directory_must_be_empty_and_source_auth_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            vcf, target = write_inputs(root)
            output = root / "out"
            output.mkdir()
            (output / "existing").write_text("x")
            with self.assertRaises(ValueError):
                MODULE.run(vcf, target, output, 1, "f" * 64)
            (output / "existing").unlink()
            with self.assertRaises(ValueError):
                MODULE.run(vcf, target, output, 1, "bad")


if __name__ == "__main__":
    unittest.main()
