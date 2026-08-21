import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M32PackedNextflowTests(unittest.TestCase):
    def test_two_stage_private_tensor_contract(self):
        module = (ROOT / "modules" / "32_PACKED_BENCHMARK.nf").read_text()
        workflow = (ROOT / "workflows" / "m32_packed_benchmark.nf").read_text()
        self.assertIn("process M32_MATERIALIZE_PACKED_TENSOR", module)
        self.assertIn("process M32_BENCHMARK_PACKED_TENSOR", module)
        self.assertIn("process M32_AUTHENTICATE_PACKED_SOURCES", module)
        self.assertNotIn("publishDir", module.split("process M32_MATERIALIZE_PACKED_TENSOR", 1)[1].split("process M32_BENCHMARK", 1)[0])
        self.assertIn("overwrite:false", module)
        self.assertIn("M32_MATERIALIZE_PACKED_TENSOR.out.tensor", workflow)
        self.assertIn("M32_AUTHENTICATE_PACKED_SOURCES.out.auth", workflow)
        self.assertIn("'root17'", workflow)
        self.assertIn("'root18'", workflow)
        self.assertNotIn("truth", workflow.lower())

    def test_all_sources_are_staged_and_authenticated(self):
        module = (ROOT / "modules" / "32_PACKED_BENCHMARK.nf").read_text()
        for source in (
            "bin/m32_source_auth.py", "bin/m32_packed_benchmark.py", "bin/m31_ordered_linear.py",
            "conf/m32_packed_benchmark_preregistration.json",
            "conf/m32_packed_benchmark.config", "modules/32_PACKED_BENCHMARK.nf",
            "workflows/m32_packed_benchmark.nf",
        ):
            self.assertIn(f"--source {source}=", module)
        self.assertIn("--source-auth ${source_auth}", module)

    def test_config_is_local_docker_bounded_and_non_retrying(self):
        config = (ROOT / "conf" / "m32_packed_benchmark.config").read_text()
        self.assertIn("executor = 'local'", config)
        self.assertIn("maxRetries = 0", config)
        self.assertIn("m32_pack_memory = '8 GB'", config)
        self.assertIn("--network none --memory 8g", config)
        self.assertIn("sha256:2c30d018028636ac1b7a4890641e04b3e15be8c79d991dfade35b90db0e17bd1", config)

    def test_contract_does_not_authorize_science_or_training(self):
        contract = json.loads((ROOT / "conf" / "m32_packed_benchmark_preregistration.json").read_text())
        self.assertFalse(contract["truth_policy"]["truth_access"])
        self.assertFalse(contract["truth_policy"]["selects_radius"])
        self.assertFalse(contract["truth_policy"]["training_authorized"])
        self.assertEqual(contract["performance_marker_sample"], 512)


if __name__ == "__main__":
    unittest.main()
