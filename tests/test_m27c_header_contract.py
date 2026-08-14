import sys
import subprocess
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from audit_gvcf_header_contract import summarize  # noqa: E402


def header(length=50818468, source="HaplotypeCaller", formats=None):
    return {
        "samples": ["redacted"],
        "chr22_length": length,
        "source": source,
        "format_ids": formats or ["GT", "DP", "GQ", "MIN_DP"],
        "has_required_fields": set(formats or ["GT", "DP", "GQ", "MIN_DP"]).issuperset(
            {"GT", "DP", "GQ", "MIN_DP"}
        ),
    }


class TestM27CHeaderContract(unittest.TestCase):
    def test_entrypoint_imports_core_from_nextflow_symlink_stage(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for name in ("audit_gvcf_header_contract.py", "m27c_gvcf_core.py"):
                (work / name).symlink_to(repo_root / "bin" / name)
            completed = subprocess.run(
                [sys.executable, "audit_gvcf_header_contract.py", "--help"],
                cwd=work,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_all_headers_must_pass_every_requirement(self):
        result = summarize([header(), header()], expected_samples=2)
        self.assertTrue(result["header_contract_pass"])
        self.assertFalse(result["sample_ids_emitted"])

    def test_failure_is_reported_as_aggregate_without_sample_ids(self):
        result = summarize([header(), header(length=150754)], expected_samples=2)
        self.assertFalse(result["header_contract_pass"])
        self.assertEqual(result["pass_counts"]["chr22_length"], 1)
        self.assertNotIn("redacted", json_text(result))


def json_text(value):
    import json

    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
