import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import m33_generate_development_root as subject


def contracts(tmp_path: Path) -> tuple[Path, Path]:
    m28 = {
        "stage": "M28_LAI_SIMULATION_PREFLIGHT",
        "version": 2,
        "pools": {
            "allocation_unit": "diploid_individual",
            "target_diploids": 30,
            "frequency_diploids": {"total": 300},
            "lai_reference_diploids_per_ancestry": 30,
            "mosaic_donor_haplotypes_per_ancestry": 512,
        },
        "rare_definition": {"minimum_mac": 2, "maximum_maf_exclusive": 0.01},
    }
    pre4 = {
        "schema_version": "2.0.0",
        "root_registry": {
            "DEVELOPMENT": [11, 12, 13],
            "consumed_technical_only": [17, 18],
            "EVAL_reserved_not_generated": [21, 22],
            "target_diploid_people_per_root": 30,
        },
        "simulation_contract": {
            "generator": "M28_v2_individual_safe",
            "demographic_model": "stdpopsim_HomSap_AmericanAdmixture_4B18",
            "ancestries": ["AFR", "EUR", "ASIA"],
            "ASIA_is_not_NAM": True,
        },
    }
    m28_path = tmp_path / "m28.json"
    pre4_path = tmp_path / "pre4.json"
    m28_path.write_text(json.dumps(m28))
    pre4_path.write_text(json.dumps(pre4))
    return m28_path, pre4_path


class DevelopmentRootContractTests(unittest.TestCase):
    def test_accepts_only_frozen_development_roots(self):
        with TemporaryDirectory() as tmp:
            m28, pre4 = contracts(Path(tmp))
            loaded, _ = subject.load_contracts(m28, pre4, 11)
            self.assertEqual(loaded["pools"]["allocation_unit"], "diploid_individual")
            for forbidden in (17, 21, 999):
                with self.assertRaises(ValueError):
                    subject.load_contracts(m28, pre4, forbidden)

    def test_rejects_generator_parameter_drift(self):
        with TemporaryDirectory() as tmp:
            m28, pre4 = contracts(Path(tmp))
            payload = json.loads(pre4.read_text())
            payload["root_registry"]["target_diploid_people_per_root"] = 31
            pre4.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "TARGET count drift"):
                subject.load_contracts(m28, pre4, 11)


if __name__ == "__main__":
    unittest.main()
