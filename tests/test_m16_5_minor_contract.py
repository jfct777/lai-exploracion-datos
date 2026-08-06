import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M165MinorContractTest(unittest.TestCase):
    def test_canonical_parameters_are_versioned_defaults(self) -> None:
        config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
        expected = (
            "ibd_enhanced_min_edge_bp = 5000000",
            "ibd_enhanced_min_max_segment_bp = 500000",
            'ibd_enhanced_edge_weight_transform = "log1p"',
            'ibd_enhanced_nmf_k_values = "2,3,4,5,6,8,10,12,15,20"',
            "ibd_enhanced_nmf_inits = 30",
            'ibd_enhanced_nmf_init_mode = "random-cophenetic"',
            "ibd_enhanced_nmf_operational_k = 8",
            "ibd_enhanced_kinship_segment_mb = 3.0",
            "ibd_enhanced_founder_intra_inter_ratio = 3.0",
            "ibd_enhanced_founder_min_silhouette = 0.0",
        )
        for declaration in expected:
            self.assertIn(declaration, config)

    def test_workflow_isolated_from_downstream_and_test(self) -> None:
        workflow = (ROOT / "workflows/m16_5_minor.nf").read_text(encoding="utf-8")
        self.assertIn("IBD_COMMUNITY_ENHANCED", workflow)
        self.assertIn("COMPARE_M16_5_ORIENTATION", workflow)
        for forbidden in ("EVALUATE_TEST", "MODEL_PRIMARY_CV", "RARE_BENCH", "COMPARE_ASIBD_COMMON"):
            self.assertNotIn(forbidden, workflow)

    def test_cloud_image_is_digest_pinned_and_labeled(self) -> None:
        cloud = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn("m16_5_analysis_container_image", cloud)
        self.assertIn("@sha256:3e9381d165b1", cloud)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)

    def test_cloud_resources_follow_measured_m16_5_profile(self) -> None:
        cloud = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn(
            "withName: 'IBD_COMMUNITY_ENHANCED'               { "
            "container = params.m16_5_analysis_container_image; cpus = 8; "
            "memory = '16 GB'; time = '1h'; disk = '50 GB'; maxForks = 1 }",
            cloud,
        )


if __name__ == "__main__":
    unittest.main()
