"""Static regression checks for the isolated, hash-locked M39 CUDA image."""

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ImageContractTests(unittest.TestCase):
    def test_base_is_pinned_and_context_has_no_project_sources(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertRegex(dockerfile.splitlines()[0], r"^FROM .*m28-lai-sim@sha256:[0-9a-f]{64}$")
        copies = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
        self.assertEqual(copies, ["COPY requirements.lock /opt/m39-image/requirements.lock",
                                 "COPY verify_image.py /opt/m39-image/verify_image.py"])

    def test_every_wheel_has_a_single_hash_and_exact_version(self):
        rows = [line for line in (ROOT / "requirements.lock").read_text().splitlines()
                if line and not line.startswith("#")]
        self.assertEqual(len(rows), 30)
        names = []
        for row in rows:
            self.assertRegex(row, r" --hash=sha256:[0-9a-f]{64}$")
            self.assertEqual(row.count("--hash="), 1)
            name = re.split(r"==| @ ", row)[0]
            names.append(name.lower().replace("_", "-"))
            if name != "torch":
                self.assertRegex(row, r"^[A-Za-z0-9_-]+==[0-9][^ ]+ --hash=")
        self.assertEqual(len(names), len(set(names)))

    def test_torch_requires_explicit_cuda_build_and_authenticated_wheel(self):
        lock = (ROOT / "requirements.lock").read_text()
        self.assertIn("torch-2.12.1%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl", lock)
        self.assertIn("0a40c1d3f014d3d6ea3e9b4fd24ac9dcbdb19e4c1e778f9b50f62c0f345584dd", lock)
        self.assertNotIn("torch==2.12.1 ", lock)
        self.assertIn("numpy==2.4.6 ", lock)

    def test_install_is_locked_without_runtime_resolution(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        for guard in ("--require-hashes", "--only-binary=:all:", "--no-deps", "pip check"):
            self.assertIn(guard, dockerfile)
        self.assertIn("--index-url https://pypi.org/simple", dockerfile)
        self.assertNotIn("apt-get", dockerfile)

    def test_uid_safe_cache_and_deterministic_environment(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache", dockerfile)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", dockerfile)
        self.assertNotRegex(dockerfile, r"(?:ENV |\s)(?:HOME|USER)=")
        self.assertIn("USER runner", dockerfile)

    def test_cuda_smoke_is_explicit_not_a_build_time_claim(self):
        source = (ROOT / "verify_image.py").read_text()
        ast.parse(source)
        self.assertIn('"cuda_exercised": False', source)
        self.assertIn('parser.add_argument("--require-cuda", action="store_true")', source)
        self.assertIn("if require_cuda and not available:", source)
        self.assertNotIn("--require-cuda", (ROOT / "Dockerfile").read_text())

    def test_context_allowlist_excludes_test_and_private_artifacts(self):
        self.assertEqual((ROOT / ".dockerignore").read_text().splitlines(),
                         ["*", "!Dockerfile", "!requirements.lock", "!verify_image.py"])


if __name__ == "__main__":
    unittest.main()
