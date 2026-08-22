import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_adapter():
    numpy = types.ModuleType("numpy")
    linear = types.ModuleType("m31_ordered_linear")
    linear.load_ordered_rare = lambda *_args, **_kwargs: None
    linear.load_genetic_map = lambda *_args, **_kwargs: None
    linear.load_ref_minor_dosage = lambda *_args, **_kwargs: None
    linear.ancestry_support = lambda *_args, **_kwargs: None
    preflight = types.ModuleType("m31_ordered_rare_preflight")
    preflight.derive_freq_sites = lambda *_args, **_kwargs: None
    with mock.patch.dict(sys.modules, {
        "numpy": numpy,
        "m31_ordered_linear": linear,
        "m31_ordered_rare_preflight": preflight,
    }):
        spec = importlib.util.spec_from_file_location("m33_a0_under_test", ROOT / "bin/m33_a0_real_adapter.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


A0 = load_adapter()


class M33A0ContractTests(unittest.TestCase):
    def test_source_auth_covers_complete_A0_implementation(self):
        source_auth = (ROOT / "bin/m33_a0_source_auth.py").read_text(encoding="utf-8")
        for relative in (
            "bin/m33_a0_real_adapter.py", "bin/m33_a0_source_auth.py", "bin/m33_a0_tabix_audit.py",
            "bin/m31_ordered_linear.py", "bin/m31_ordered_rare_preflight.py",
            "conf/m33_a0_legacy_assets.json", "conf/m33_a0_real_adapter.config",
            "conf/m33_a0_real_adapter_preregistration.json",
            "modules/33_A0_REAL_ADAPTER.nf", "workflows/m33_a0_real_adapter.nf",
            "tests/test_m33_a0_real_adapter.py", "tests/test_m33_a0_real_adapter_nextflow.py",
        ):
            self.assertIn(f'"{relative}"', source_auth)

    def test_contract_is_technical_only_and_registry_is_add_only(self):
        contract = json.loads((ROOT / "conf/m33_a0_real_adapter_preregistration.json").read_text())
        registry = json.loads((ROOT / "conf/m33_a0_legacy_assets.json").read_text())
        self.assertEqual(contract["status"], "TECHNICAL_COMPATIBILITY_ONLY")
        self.assertEqual(set(registry["roots"]), {"root17", "root18"})
        root17_canonical = json.dumps(
            registry["roots"]["root17"], sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(
            hashlib.sha256(root17_canonical).hexdigest(),
            "4a9c6fe8d7cc9cddc9e44f1d8f0595c122d0b653774bef7813ca9f913f6067d8",
        )
        root18_canonical = json.dumps(
            registry["roots"]["root18"], sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(
            hashlib.sha256(root18_canonical).hexdigest(),
            "132e133031f81956ecb11fae5861c895b21108edeb03f823e57b4155833aa39a",
        )
        self.assertEqual(registry["roots"]["root17"]["root_seed"], 20260817)
        self.assertEqual(registry["roots"]["root18"]["root_seed"], 20260818)
        self.assertIsNone(registry["roots"]["root17"]["sha256"]["flare_anc_tbi"])
        self.assertIsNone(registry["roots"]["root18"]["sha256"]["flare_anc_tbi"])
        self.assertFalse(contract["execution_authorization"]["write_READY"])
        for key in ("materialize_tensor", "forward", "backward", "training", "truth_scoring"):
            self.assertFalse(contract["execution_authorization"][key])

    def test_contract_accepts_reconciled_root18(self):
        _contract, root = A0.load_contract(
            ROOT / "conf/m33_a0_real_adapter_preregistration.json",
            ROOT / "conf/m33_a0_legacy_assets.json", "root18", 20260818,
        )
        self.assertEqual(root["expected_counts"]["selected_rare_sites"], 94703)
        self.assertEqual(root["expected_counts"]["rare_overlap_flare_sites"], 0)

    def test_contract_rejects_mismatched_root_seed_pair(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            A0.load_contract(
                ROOT / "conf/m33_a0_real_adapter_preregistration.json",
                ROOT / "conf/m33_a0_legacy_assets.json", "root18", 20260817,
            )

    def test_contract_drift_is_rejected_by_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            prereg = json.loads((ROOT / "conf/m33_a0_real_adapter_preregistration.json").read_text())
            prereg["vcf_contract"]["phased_binary_GT"] = False
            changed = Path(directory) / "changed.json"
            changed.write_text(json.dumps(prereg))
            with self.assertRaisesRegex(ValueError, "preregistration hash drifted"):
                A0.load_contract(changed, ROOT / "conf/m33_a0_legacy_assets.json", "root17", 20260817)

    def test_incomplete_source_auth_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps({
                "stage": "M33_A0_SOURCE_AUTH",
                "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
                "git_commit": "a" * 40,
            }))
            with self.assertRaisesRegex(ValueError, "keys drifted"):
                A0.load_source_auth(path, "a" * 40, {})

    def test_complete_but_false_source_hashes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            sources = {relative: ROOT / relative for relative in A0.REQUIRED_SOURCE_PATHS}
            path.write_text(json.dumps({
                "stage": "M33_A0_SOURCE_AUTH",
                "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
                "git_commit": "a" * 40,
                "source_sha256": {relative: "0" * 64 for relative in sources},
            }))
            with self.assertRaisesRegex(ValueError, "staged A0 sources differ"):
                A0.load_source_auth(path, "a" * 40, sources)

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"stage":"one","stage":"two"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                A0.load_json_strict(path)

    def test_tbi_magic_is_checked_after_gzip_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.tbi"
            bad = Path(directory) / "bad.tbi"
            with gzip.open(good, "wb") as handle:
                handle.write(b"TBI\x01payload")
            with gzip.open(bad, "wb") as handle:
                handle.write(b"NOPEpayload")
            A0.audit_tbi(good)
            with self.assertRaisesRegex(ValueError, "Tabix"):
                A0.audit_tbi(bad)

    def test_vcf_rejects_unphased_and_grid_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.vcf.gz"
            with gzip.open(path, "wt") as handle:
                handle.write("##fileformat=VCFv4.2\n")
                handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT000\n")
                handle.write("22\t100\tm28s1\tA\tC\t.\tPASS\tTSID=1\tGT\t0/1\n")
            with self.assertRaisesRegex(ValueError, "phased"):
                A0.audit_vcf(path, ("T000",))

    def test_vcf_accepts_phased_binary_known_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.vcf.gz"
            with gzip.open(path, "wt") as handle:
                handle.write("##fileformat=VCFv4.2\n")
                handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT000\n")
                handle.write("22\t100\tm28s1\tA\tC\t.\tPASS\tTSID=1\tGT\t0|1\n")
            audit = A0.audit_vcf(path, ("T000",))
            self.assertEqual(audit["loci"], ((100, "m28s1", "A", "C", 1),))

    def test_vcf_requires_exact_fileformat_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-header.vcf.gz"
            with gzip.open(path, "wt") as handle:
                handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT000\n")
                handle.write("22\t100\tm28s1\tA\tC\t.\tPASS\tTSID=1\tGT\t0|1\n")
            with self.assertRaisesRegex(ValueError, "fileformat"):
                A0.audit_vcf(path, ("T000",))

    def test_genotype_digest_detects_flare_target_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.vcf.gz"
            flare = Path(directory) / "flare.vcf.gz"
            with gzip.open(target, "wt") as handle:
                handle.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT000\n")
                handle.write("22\t100\tm28s1\tA\tC\t.\tPASS\tTSID=1\tGT\t0|0\n")
            with gzip.open(flare, "wt") as handle:
                handle.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT000\n")
                handle.write("22\t100\tm28s1\tA\tC\t.\tPASS\tTSID=1\tGT:AN1:AN2:ANP1:ANP2\t1|1:0:0:1,0,0:1,0,0\n")
            target_audit = A0.audit_vcf(target, ("T000",))
            flare_audit = A0.audit_vcf(flare, ("T000",), flare=True)
            self.assertNotEqual(target_audit["gt_sha256"], flare_audit["gt_sha256"])

    def test_ref_mapping_rejects_nodes_not_in_ref_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.tsv"
            panel = Path(directory) / "panel.tsv"
            with pairs.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("sample_id", "ancestry", "haplotype_0_node", "haplotype_1_node"))
                writer.writerow(("REF_AFR_000", "AFR", 900000, 900001))
            panel.write_text("REF_AFR_000\tAFR\n")
            expected = {"ref_people": 1}
            with self.assertRaisesRegex(ValueError, "exact REF_LAI pool nodes"):
                A0.audit_ref_mapping(pairs, panel, expected, {("AFR", 1, 2)})

    def test_receipt_schema_forbids_private_identifiers(self):
        source = (ROOT / "bin/m33_a0_real_adapter.py").read_text()
        self.assertIn('"scientific_evidence": False', source)
        self.assertIn('"ready_emitted": False', source)
        receipt_block = source.split("receipt = {", 1)[1].split("write_exclusive_json", 1)[0]
        for forbidden in ('"paths"', '"sample_ids"', '"node_ids"', '"genotypes"', '"truth_segments"'):
            self.assertNotIn(forbidden, receipt_block)


if __name__ == "__main__":
    unittest.main()
