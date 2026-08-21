import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_asset_manifest_contract", ROOT / "bin/m33_asset_manifest_contract.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)
CONTRACT_PATH = ROOT / "conf/m33_asset_manifest_contract.json"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_person(root, role, ancestry, index):
    person_id = f"m33:{root}:{role.lower()}:{index:04d}"
    haplotypes = [
        {
            "haplotype_id": f"{person_id}:h{homologue}",
            "canonical_sha256": digest(f"simulated-haplotype:{root}|{role}|{ancestry}|{index}|{homologue}"),
        }
        for homologue in (0, 1)
    ]
    source_person_sha256 = None if role == "TARGET" else digest(
        ancestry + "|" + "|".join(sorted(hap["canonical_sha256"] for hap in haplotypes))
    )
    return {
        "person_id": person_id,
        "source_person_sha256": source_person_sha256,
        "ancestry": ancestry,
        "haplotypes": haplotypes,
    }


def make_roles(root):
    roles = {}
    specs = {
        "FREQ": (("AFR", 50), ("EUR", 100), ("ASIA", 150)),
        "REF_LAI": (("AFR", 30), ("EUR", 30), ("ASIA", 30)),
        "DONOR": (("AFR", 256), ("EUR", 256), ("ASIA", 256)),
    }
    for role, groups in specs.items():
        rows = []
        index = 0
        for ancestry, count in groups:
            for _ in range(count):
                rows.append(make_person(root, role, ancestry, index))
                index += 1
        roles[role] = rows
    roles["TARGET"] = [make_person(root, "TARGET", "MOSAIC", index) for index in range(30)]
    return roles


def make_asset(contract, root, asset_set_id, logical_id):
    prefix = MOD.expected_prefix(contract, root, asset_set_id)
    if logical_id == "map":
        shared = contract["shared_assets"]
        return {
            "logical_id": "map",
            "gcs_uri": shared["genetic_map_uri"],
            "gcs_generation": shared["genetic_map_gcs_generation"],
            "size_bytes": shared["genetic_map_size_bytes"],
            "sha256_raw": shared["genetic_map_sha256"],
            "crc32c": shared["genetic_map_crc32c"],
            "media_type": shared["genetic_map_media_type"],
            "compression": shared["genetic_map_compression"],
            "schema_version": shared["genetic_map_schema_version"],
            "record_count": shared["genetic_map_record_count"],
        }
    elif logical_id == "generator_source_auth":
        uri = contract["shared_assets"]["generator_source_auth_prefix"] + "source-auth.fixture.json"
        raw_sha = digest("source-auth:shared")
    else:
        uri = f"{prefix}{logical_id}.fixture"
        raw_sha = digest(f"raw:{root}:{logical_id}")
    return {
        "logical_id": logical_id,
        "gcs_uri": uri,
        "gcs_generation": "1720000000000000",
        "size_bytes": 200 if logical_id == "generator_source_auth" else 100 + root % 17,
        "sha256_raw": raw_sha,
        "crc32c": "AAAAAA==",
        "media_type": "application/octet-stream",
        "compression": "none",
        "schema_version": "fixture-1",
        "record_count": 1,
    }


def make_manifest(contract, root):
    generator_receipt = digest(f"generator-receipt:{root}")
    source_auth = digest("source-auth:shared")
    asset_set_id = MOD.asset_set_id_for(root, generator_receipt)
    roles = make_roles(root)
    donors = [hap["haplotype_id"] for person in roles["DONOR"] for hap in person["haplotypes"]]
    targets = [hap["haplotype_id"] for person in roles["TARGET"] for hap in person["haplotypes"]]
    manifest = {
        "schema_version": "1.0.0",
        "stage": "M33_PRIVATE_DEVELOPMENT_ROOT_ASSETS",
        "mode": "DEVELOPMENT_ASSETS",
        "root_seed": root,
        "root_namespace": f"m33-development-root-{root}",
        "asset_set_id": asset_set_id,
        "output_prefix": MOD.expected_prefix(contract, root, asset_set_id),
        "creation_precondition": "ifGenerationMatch=0",
        "base_contract_sha256": MOD.BASE_CONTRACT_SHA256,
        "generator": {
            "repository": contract["generator_anchors"]["repository"],
            "git_commit": contract["generator_anchors"]["source_commit"],
            "clean_source_auth_sha256": source_auth,
            "source_sha256": contract["generator_anchors"]["source_sha256"],
            "nextflow_version": contract["generator_anchors"]["runtime"]["nextflow"],
            "oci_image_digest": contract["generator_anchors"]["oci_image_digest"],
            "python_version": contract["generator_anchors"]["runtime"]["python"],
            "stdpopsim_version": "0.3.0",
            "msprime_version": "1.4.2",
            "tskit_version": "1.0.3",
            "numpy_version": "2.4.6",
            "root_seed": root,
            "rng_streams": MOD.derive_rng_streams(root, contract),
            "rng_seedsequence": contract["rng_contract"]["root_streams"][str(root)],
            "new_tree_sequence_for_root": True,
            "generator_receipt_sha256": generator_receipt,
        },
        "assets": {
            logical_id: make_asset(contract, root, asset_set_id, logical_id)
            for logical_id in contract["asset_inventory"]["required"]
        },
        "roles": roles,
        "donor_to_target_haplotypes": [
            {"donor_haplotype_id": donors[index], "target_haplotype_id": target}
            for index, target in enumerate(targets)
        ],
        "semantic_fingerprints": {
            logical_id: digest(f"semantic:{root}:{logical_id}")
            for logical_id in contract["semantic_fingerprints"]["required"]
        },
        "rare_flare_grid_overlap": {
            "rare_site_count": 100,
            "flare_grid_site_count": 200,
            "overlap_site_count": 5,
            "overlap_fraction_of_rare": 0.05,
        },
        "truth_barrier": {
            "truth_state": "SEALED_PRIVATE_NOT_EXPOSED_TO_PREDICT",
            "predict_view_contains_truth": False,
            "predictor_accepts_truth_argument": False,
        },
        "private_manifest_sha256": "0" * 64,
    }
    manifest["assets"]["generator_manifest"]["sha256_raw"] = generator_receipt
    manifest["assets"]["generator_source_auth"]["sha256_raw"] = source_auth
    manifest["private_manifest_sha256"] = MOD.private_manifest_payload_sha256(manifest)
    return manifest


def reseal(manifest):
    manifest["private_manifest_sha256"] = MOD.private_manifest_payload_sha256(manifest)


class M33AssetManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = MOD.load_contract(CONTRACT_PATH)

    def bundle(self):
        return [make_manifest(self.contract, root) for root in self.contract["root_seeds"]]

    def test_contract_is_immutable_and_authorizes_only_fixture_tests(self):
        self.assertEqual(MOD.sha256_file(CONTRACT_PATH), MOD.EXACT_CONTRACT_SHA256)
        self.assertFalse(self.contract["execution_authorization"]["real_asset_read"])
        self.assertFalse(self.contract["execution_authorization"]["asset_generation"])
        self.assertFalse(self.contract["execution_authorization"]["training"])

    def test_complete_bundle_passes_and_receipt_is_redacted(self):
        receipt = MOD.validate_bundle(self.bundle(), self.contract)
        self.assertEqual(receipt["status"], MOD.PASS_STATUS)
        self.assertEqual(receipt["root_seeds"], self.contract["root_seeds"])
        self.assertEqual(receipt["aggregate_counts"], {
            "FREQ": 900, "REF_LAI": 270, "DONOR": 2304, "TARGET": 90,
        })
        MOD.assert_predict_receipt_redacted(receipt, self.contract)

    def test_missing_extra_duplicate_or_forbidden_root_fails(self):
        bundle = self.bundle()
        cases = [bundle[:2], bundle + [copy.deepcopy(bundle[0])]]
        for value in cases:
            with self.subTest(length=len(value)), self.assertRaises(ValueError):
                MOD.validate_bundle(value, self.contract)
        forbidden = copy.deepcopy(bundle)
        forbidden[0]["root_seed"] = 20260817
        with self.assertRaisesRegex(ValueError, "unregistered or forbidden"):
            MOD.validate_bundle(forbidden, self.contract)

    def test_missing_or_extra_asset_and_bad_hash_fail(self):
        for mutation in ("missing", "extra", "hash"):
            bundle = self.bundle()
            if mutation == "missing":
                del bundle[0]["assets"]["sites"]
            elif mutation == "extra":
                bundle[0]["assets"]["unexpected"] = copy.deepcopy(bundle[0]["assets"]["sites"])
            else:
                bundle[0]["assets"]["sites"]["sha256_raw"] = "bad"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MOD.validate_bundle(bundle, self.contract)

    def test_role_overlap_and_non_diploid_person_fail(self):
        bundle = self.bundle()
        bundle[0]["roles"]["REF_LAI"][0] = copy.deepcopy(bundle[0]["roles"]["FREQ"][0])
        with self.assertRaisesRegex(ValueError, "multiple roles"):
            MOD.validate_bundle(bundle, self.contract)
        bundle = self.bundle()
        bundle[0]["roles"]["TARGET"][0]["haplotypes"].pop()
        with self.assertRaisesRegex(ValueError, "not diploid"):
            MOD.validate_bundle(bundle, self.contract)

    def test_founder_tree_and_rng_reuse_across_roots_fail(self):
        for mutation in ("founder", "tree_raw", "tree_semantic", "rng"):
            bundle = self.bundle()
            if mutation == "founder":
                person = bundle[1]["roles"]["FREQ"][0]
                person["haplotypes"][0]["canonical_sha256"] = (
                    bundle[0]["roles"]["FREQ"][0]["haplotypes"][0]["canonical_sha256"]
                )
                person["source_person_sha256"] = digest(
                    person["ancestry"] + "|" + "|".join(sorted(
                        hap["canonical_sha256"] for hap in person["haplotypes"]
                    ))
                )
            elif mutation == "tree_raw":
                bundle[1]["assets"]["tree_sequence"]["sha256_raw"] = (
                    bundle[0]["assets"]["tree_sequence"]["sha256_raw"]
                )
            elif mutation == "tree_semantic":
                bundle[1]["semantic_fingerprints"]["normalized_tree_tables_sha256"] = (
                    bundle[0]["semantic_fingerprints"]["normalized_tree_tables_sha256"]
                )
            else:
                bundle[1]["generator"]["rng_streams"] = bundle[0]["generator"]["rng_streams"]
            reseal(bundle[1])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MOD.validate_bundle(bundle, self.contract)

    def test_source_person_and_nonmap_asset_reuse_across_roots_fail(self):
        bundle = self.bundle()
        left = bundle[0]["roles"]["FREQ"][0]
        right = bundle[1]["roles"]["FREQ"][0]
        right["haplotypes"] = copy.deepcopy(left["haplotypes"])
        for homologue, hap in enumerate(right["haplotypes"]):
            hap["haplotype_id"] = f"{right['person_id']}:h{homologue}"
        right["source_person_sha256"] = left["source_person_sha256"]
        reseal(bundle[1])
        with self.assertRaisesRegex(ValueError, "source person reused"):
            MOD.validate_bundle(bundle, self.contract)

        bundle = self.bundle()
        bundle[1]["assets"]["sites"] = copy.deepcopy(bundle[0]["assets"]["sites"])
        # Keep the second root's canonical prefix so the root-level check passes,
        # while reusing immutable content/generation is still caught by the bundle.
        bundle[1]["assets"]["sites"]["gcs_uri"] = bundle[1]["output_prefix"] + "sites.fixture"
        reseal(bundle[1])
        with self.assertRaisesRegex(ValueError, "non-shareable asset reused: sites"):
            MOD.validate_bundle(bundle, self.contract)

    def test_generator_receipt_and_source_auth_are_bound_to_assets(self):
        for field, logical_id in (("generator_receipt_sha256", "generator_manifest"),
                                  ("clean_source_auth_sha256", "generator_source_auth")):
            bundle = self.bundle()
            bundle[0]["generator"][field] = digest("unbound")
            if field == "generator_receipt_sha256":
                bundle[0]["asset_set_id"] = MOD.asset_set_id_for(
                    bundle[0]["root_seed"], bundle[0]["generator"][field]
                )
                bundle[0]["output_prefix"] = MOD.expected_prefix(
                    self.contract, bundle[0]["root_seed"], bundle[0]["asset_set_id"]
                )
                for key, asset in bundle[0]["assets"].items():
                    if key not in self.contract["asset_inventory"]["shareable_between_roots"]:
                        asset["gcs_uri"] = bundle[0]["output_prefix"] + f"{key}.fixture"
            reseal(bundle[0])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "not bound"):
                MOD.validate_bundle(bundle, self.contract)

    def test_truth_leak_and_unsafe_uri_fail(self):
        bundle = self.bundle()
        bundle[0]["truth_barrier"]["predict_view_contains_truth"] = True
        with self.assertRaisesRegex(ValueError, "truth barrier"):
            MOD.validate_bundle(bundle, self.contract)
        bundle = self.bundle()
        bundle[0]["assets"]["sites"]["gcs_uri"] = (
            "gs://projects-usp/dnaBr-lai/datalake/%2e%2e/escape"
        )
        with self.assertRaises(ValueError):
            MOD.validate_bundle(bundle, self.contract)

    def test_runtime_asset_set_manifest_hash_generation_and_uri_guards(self):
        mutations = ("runtime", "image", "asset_set", "manifest_hash", "generation", "percent_uri")
        for mutation in mutations:
            bundle = self.bundle()
            root = bundle[0]
            if mutation == "runtime":
                root["generator"]["tskit_version"] = "0.6.4"
            elif mutation == "image":
                root["generator"]["oci_image_digest"] = "registry.invalid/image@sha256:" + digest("x")
            elif mutation == "asset_set":
                root["asset_set_id"] = digest("wrong-asset-set")
            elif mutation == "manifest_hash":
                root["private_manifest_sha256"] = digest("wrong-private-manifest")
            elif mutation == "generation":
                root["assets"]["sites"]["gcs_generation"] = "0"
            else:
                root["assets"]["sites"]["gcs_uri"] = root["output_prefix"] + "%252e%252e/escape"
            if mutation in {"runtime", "image", "generation", "percent_uri"}:
                reseal(root)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MOD.validate_bundle(bundle, self.contract)

    def test_shared_map_is_allowed_but_other_shared_content_is_not(self):
        bundle = self.bundle()
        self.assertEqual(bundle[0]["assets"]["map"]["sha256_raw"],
                         bundle[1]["assets"]["map"]["sha256_raw"])
        MOD.validate_bundle(bundle, self.contract)
        bundle[1]["assets"]["tree_sequence"]["sha256_raw"] = (
            bundle[0]["assets"]["tree_sequence"]["sha256_raw"]
        )
        reseal(bundle[1])
        with self.assertRaisesRegex(ValueError, "tree byte hash reused"):
            MOD.validate_bundle(bundle, self.contract)

    def test_strict_json_rejects_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, raw in enumerate(('{"x":1,"x":2}', '{"x":NaN}')):
                path = Path(directory) / f"bad-{index}.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MOD.strict_json(path)

    def test_redaction_guard_rejects_truth_or_private_uri(self):
        receipt = MOD.validate_bundle(self.bundle(), self.contract)
        for key, value in (("truth_uri", "gs://private/truth"),
                           ("private_gcs_uri", "gs://private/root")):
            changed = copy.deepcopy(receipt)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MOD.assert_predict_receipt_redacted(changed, self.contract)

    def test_source_auth_rejects_fake_commit_dirty_tree_and_changed_staging(self):
        if shutil.which("git") is None:
            self.skipTest("Git is required for source authentication")
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            staged = {}
            for relative in MOD.REQUIRED_SOURCES:
                destination = repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
                staged[relative] = destination
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "jfct777"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "jcalderonta@ime.usp.br"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            hashes = {relative: MOD.sha256_file(path) for relative, path in staged.items()}
            auth = {
                "stage": "M33_ASSET_MANIFEST_SOURCE_AUTH",
                "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
                "git_commit": commit,
                "source_sha256": hashes,
            }
            auth_path = Path(directory) / "auth.json"
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            self.assertEqual(MOD.validate_source_auth(auth_path, commit, staged, repo), hashes)

            changed = Path(directory) / "changed.py"
            changed.write_text("changed", encoding="utf-8")
            changed_staging = dict(staged)
            changed_staging["bin/m33_asset_manifest_contract.py"] = changed
            with self.assertRaisesRegex(ValueError, "changed after authentication"):
                MOD.validate_source_auth(auth_path, commit, changed_staging, repo)

            with self.assertRaisesRegex(ValueError, "source-auth commit drift"):
                MOD.validate_source_auth(auth_path, "a" * 40, staged, repo)

            staged["bin/m33_asset_manifest_contract.py"].write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty or untracked"):
                MOD.validate_source_auth(auth_path, commit, staged, repo)

    def test_exclusive_receipt_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            MOD.write_exclusive(output, {"status": "first"})
            self.assertEqual(json.loads(output.read_text())["status"], "first")
            with self.assertRaises(FileExistsError):
                MOD.write_exclusive(output, {"status": "second"})


if __name__ == "__main__":
    unittest.main()
