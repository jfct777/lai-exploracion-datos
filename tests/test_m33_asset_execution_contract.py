import copy
import hashlib
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
SPEC = importlib.util.spec_from_file_location(
    "m33_asset_execution_contract", ROOT / "bin/m33_asset_execution_contract.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)
REAL_ASSET_CONTRACT = MOD.load_contract(ROOT / "conf/m33_asset_manifest_contract.json")
AMENDMENT = MOD.load_amendment(ROOT / "conf/m33_asset_execution_amendment.json")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def descriptor(logical_id, schema, uri, payload, record_count=1):
    return {
        "logical_id": logical_id,
        "gcs_uri": uri,
        "gcs_generation": "1720000000000001",
        "size_bytes": len(payload),
        "sha256_raw": hashlib.sha256(payload).hexdigest(),
        "crc32c": MOD.crc32c_base64(payload),
        "media_type": "application/octet-stream",
        "compression": "none",
        "schema_version": schema,
        "record_count": record_count,
    }


MAP_BYTES = b"chrom\tbp\tcm\nchr22\t1\t0.0\nchr22\t5000\t50.0\n"
ASSET_CONTRACT = copy.deepcopy(REAL_ASSET_CONTRACT)
ASSET_CONTRACT["shared_assets"].update({
    "genetic_map_size_bytes": len(MAP_BYTES),
    "genetic_map_sha256": hashlib.sha256(MAP_BYTES).hexdigest(),
    "genetic_map_crc32c": MOD.crc32c_base64(MAP_BYTES),
    "genetic_map_record_count": 2,
})


def map_descriptor():
    shared = ASSET_CONTRACT["shared_assets"]
    return {
        "logical_id": "genetic_map",
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


ORCHESTRATOR_COMMIT = "a" * 40
SOURCE_HASHES = {relative: digest(f"source:{relative}") for relative in MOD.REQUIRED_SOURCES}
SOURCE_HASHES["conf/m33_asset_execution_amendment.json"] = MOD.AMENDMENT_SHA256
SOURCE_AUTH_OBJECT = {
    "stage": "M33_ASSET_EXECUTION_SOURCE_AUTH",
    "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
    "git_commit": ORCHESTRATOR_COMMIT,
    "source_sha256": SOURCE_HASHES,
}
SOURCE_AUTH_BYTES = MOD.canonical_json(SOURCE_AUTH_OBJECT)
SOURCE_AUTH_DESCRIPTOR = descriptor(
    "generator_source_auth",
    AMENDMENT["manifest_members"]["generator_source_auth"]["schema"],
    ASSET_CONTRACT["shared_assets"]["generator_source_auth_prefix"] + "auth.json",
    SOURCE_AUTH_BYTES,
)
FIXTURE_PAYLOADS = {}


def selected_site(pos, ref, alt, minor_index=1):
    return {
        "CHROM": "chr22", "POS": pos, "REF": ref, "ALT": alt,
        "minor_allele_index": minor_index, "minor_mac": 2, "minor_an": 600,
        "minor_maf": 2 / 600, "carrier_people": 2,
    }


def fixture_sites(root):
    offset = MOD.DEVELOPMENT_ROOTS.index(root) * 1000
    return (
        [selected_site(100 + offset, "A", "G"), selected_site(300 + offset, "C", "T", 0)],
        [selected_site(200 + offset, "G", "A")],
    )


def selected_sites_document(logical_id, root):
    incremental, overlap = fixture_sites(root)
    rows = incremental if logical_id == "selected_sites_incremental" else overlap
    stage = ("M33_SELECTED_SITES_INCREMENTAL" if logical_id ==
             "selected_sites_incremental" else "M33_SELECTED_SITES_OVERLAP_FLARE")
    return {"schema_version": "1.0.0", "stage": stage, "status": "PASS", "rows": rows}


def target_rare_document(root):
    roles = make_roles(root)
    targets = sorted(
        haplotype["haplotype_id"]
        for person in roles["TARGET"] for haplotype in person["haplotypes"]
    )
    incremental, _ = fixture_sites(root)
    rows = [
        {
            "target_haplotype_id": target,
            "CHROM": site["CHROM"], "POS": site["POS"],
            "REF": site["REF"], "ALT": site["ALT"],
            "minor_allele_presence": (
                (target_index + site_index) % 2 if site["minor_allele_index"] == 1
                else 1 - ((target_index + site_index) % 2)
            ),
        }
        for target_index, target in enumerate(targets)
        for site_index, site in enumerate(incremental)
    ]
    return {
        "schema_version": "1.0.0", "stage": "M33_TARGET_RARE_INCREMENTAL",
        "status": "PASS", "target_haplotype_ids": targets, "rows": rows,
    }


def roles_document(root):
    return {
        "schema_version": "1.0.0", "stage": "M33_COMPLETE_DIPLOID_ROLES",
        "status": "PASS", "roles": make_roles(root),
    }


def flare_grid(root):
    offset = MOD.DEVELOPMENT_ROOTS.index(root) * 1000
    return [
        {"CHROM": "chr22", "POS": 200 + offset, "REF": "G", "ALT": "A"},
        {"CHROM": "chr22", "POS": 400 + offset, "REF": "T", "ALT": "C"},
    ]


def target_vcf_fixture_document(root):
    targets = target_rare_document(root)["target_haplotype_ids"]
    return {
        "schema_version": "1.0.0", "stage": "M33_TARGET_VCF_FIXTURE_VIEW",
        "status": "PASS", "chromosome": "chr22", "target_haplotype_ids": targets,
        "loci": flare_grid(root),
    }


def flare_anc_document(root):
    targets = target_rare_document(root)["target_haplotype_ids"]
    loci = flare_grid(root)
    rows = [
        {
            "target_haplotype_id": target, **locus,
            "probabilities": [1.0, 0.0, 0.0] if index % 3 == 0 else
            [0.0, 1.0, 0.0] if index % 3 == 1 else [0.0, 0.0, 1.0],
        }
        for index, target in enumerate(targets) for locus in loci
    ]
    return {
        "schema_version": "1.0.0", "stage": "M33_FLARE_ANC", "status": "PASS",
        "chromosome": "chr22", "target_haplotype_ids": targets, "loci": loci, "rows": rows,
    }


def flare_global_document(root):
    targets = target_rare_document(root)["target_haplotype_ids"]
    return {
        "schema_version": "1.0.0", "stage": "M33_FLARE_GLOBAL", "status": "PASS",
        "ancestry_order": list(AMENDMENT["flare_contract"]["ancestry_order"]),
        "rows": [
            {"target_haplotype_id": target, "probabilities": [0.4, 0.35, 0.25]}
            for target in targets
        ],
    }


def mosaic_fixture_documents(root):
    roles = make_roles(root)
    targets = sorted(
        haplotype["haplotype_id"]
        for person in roles["TARGET"] for haplotype in person["haplotypes"]
    )
    donors = [
        (haplotype["haplotype_id"], person["ancestry"])
        for person in roles["DONOR"] for haplotype in person["haplotypes"]
    ][:60]
    events = [
        {"target_haplotype_id": target, "start_bp": 1, "end_bp": 5001,
         "donor_haplotype_id": donor, "ancestry": ancestry}
        for target, (donor, ancestry) in zip(targets, donors)
    ]
    truth = [
        {"target_haplotype_id": row["target_haplotype_id"], "start_bp": 1,
         "end_bp": 5001, "ancestry": row["ancestry"]}
        for row in events
    ]
    loci = [
        {key: site[key] for key in ("CHROM", "POS", "REF", "ALT")}
        for site in fixture_sites(root)[0]
    ]
    donor_rows = []
    target_rows = []
    for target_index, (target, (donor, _)) in enumerate(zip(targets, donors)):
        for locus_index, locus in enumerate(loci):
            allele = (target_index + locus_index) % 2
            donor_rows.append({"haplotype_id": donor, **locus, "allele": allele})
            target_rows.append({"haplotype_id": target, **locus, "allele": allele})
    return {
        "mosaic_events": {
            "schema_version": "1.0.0", "stage": "M33_MOSAIC_EVENTS", "status": "PASS",
            "chromosome": "chr22", "rows": events,
        },
        "truth": {
            "schema_version": "1.0.0", "stage": "M33_HAPLOTYPE_TRUTH", "status": "PASS",
            "chromosome": "chr22", "rows": truth,
        },
        "donor_to_target_provenance": {
            "schema_version": "1.0.0", "stage": "M33_DONOR_TARGET_PROVENANCE",
            "status": "PASS", "chromosome": "chr22",
            "donor_ancestry": {donor: ancestry for donor, ancestry in donors},
            "donor_alleles": donor_rows, "target_alleles": target_rows,
        },
    }


def flare_run_document(plan, assets):
    flare = AMENDMENT["flare_contract"]
    return {
        "schema_version": "1.0.0", "stage": "M33_FLARE_RUN_MANIFEST", "status": "PASS",
        "plan_id": plan["plan_id"],
        "interface_sha256": MOD.flare_interface_sha256(assets, AMENDMENT),
        "generator_source_auth_sha256": plan["generator_source_auth_sha256"],
        "simulation_engine_commit": plan["engine"]["git_commit"],
        "orchestrator_commit": plan["orchestrator"]["git_commit"],
        "flare_version": flare["version"], "flare_reported_build": flare["reported_build"],
        "flare_jar_sha256": flare["jar_sha256"], "container_digest": flare["container_digest"],
        "command_argv": list(flare["command_argv"]),
        "input_descriptor_sha256": {
            logical_id: MOD.canonical_json_sha256(assets[logical_id])
            for logical_id in flare["input_logical_ids"]
        },
        "output_descriptor_sha256": {
            logical_id: MOD.canonical_json_sha256(assets[logical_id])
            for logical_id in flare["direct_command_output_logical_ids"]
        },
        "truth_accessed": False,
        "started_at_utc": "2026-08-21T20:00:00Z",
        "finished_at_utc": "2026-08-21T20:01:00Z",
    }


def flare_audit_document(plan, assets):
    return {
        "schema_version": "1.0.0", "stage": "M33_FLARE_TRUTH_BLIND_AUDIT",
        "status": "PASS", "plan_id": plan["plan_id"],
        "interface_sha256": MOD.flare_interface_sha256(assets, AMENDMENT),
        "run_manifest_sha256": assets["flare_run_manifest"]["sha256_raw"],
        "prediction_sha256": assets["flare_anc"]["sha256_raw"],
        "target_haplotype_count": 60, "locus_count": 2,
        "sample_parity_exact": True, "locus_parity_exact": True,
        "probabilities_finite": True,
        "probability_simplex_exact_within_tolerance": True,
        "simplex_tolerance": 1e-6, "truth_accessed": False,
        "started_at_utc": "2026-08-21T20:02:00Z",
        "finished_at_utc": "2026-08-21T20:03:00Z",
    }


def make_plan(root):
    inputs = {
        "genetic_map": map_descriptor(),
        "generator_source_auth": copy.deepcopy(SOURCE_AUTH_DESCRIPTOR),
    }
    rng = ASSET_CONTRACT["rng_contract"]["root_streams"][str(root)]
    plan_id = MOD.derive_plan_id(
        root, SOURCE_AUTH_DESCRIPTOR["sha256_raw"], inputs, rng, MOD.AMENDMENT_SHA256
    )
    return {
        "schema_version": "1.0.0",
        "stage": "M33_PLAN_MANIFEST",
        "mode": "DEVELOPMENT",
        "root_seed": root,
        "plan_id": plan_id,
        "asset_set_id": plan_id,
        "output_prefix": MOD.expected_prefix(AMENDMENT, root, plan_id),
        "base_contract_sha256": MOD.BASE_CONTRACT_SHA256,
        "asset_contract_sha256": MOD.EXACT_CONTRACT_SHA256,
        "amendment_sha256": MOD.AMENDMENT_SHA256,
        "generator_source_auth_sha256": SOURCE_AUTH_DESCRIPTOR["sha256_raw"],
        "input_descriptors": inputs,
        "rng_seedsequence": rng,
        "engine": {
            "repository": "jfct777/lai-exploracion-datos",
            "git_commit": MOD.ENGINE_COMMIT,
            "oci_image_digest": AMENDMENT["code_anchors"]["oci_image_digest"],
        },
        "orchestrator": {
            "repository": "jfct777/lai-exploracion-datos",
            "git_commit": ORCHESTRATOR_COMMIT,
            "source_auth_sha256": SOURCE_AUTH_DESCRIPTOR["sha256_raw"],
        },
        "creation_precondition": "ifGenerationMatch=0",
    }


def make_manifest(plan, ordinal):
    assets = {}
    for logical_id, definition in AMENDMENT["manifest_members"].items():
        if logical_id == "genetic_map":
            assets[logical_id] = map_descriptor()
        elif logical_id == "generator_source_auth":
            assets[logical_id] = copy.deepcopy(SOURCE_AUTH_DESCRIPTOR)
        else:
            payload = asset_payload(plan, logical_id)
            FIXTURE_PAYLOADS[(plan["plan_id"], logical_id)] = payload
            assets[logical_id] = descriptor(
                logical_id, definition["schema"],
                f"{plan['output_prefix']}{logical_id}.fixture", payload,
            )

    def bind_fixture(logical_id, payload, record_count):
        FIXTURE_PAYLOADS[(plan["plan_id"], logical_id)] = payload
        assets[logical_id] = descriptor(
            logical_id, AMENDMENT["manifest_members"][logical_id]["schema"],
            f"{plan['output_prefix']}{logical_id}.fixture", payload, record_count,
        )

    roles_payload_document = roles_document(plan["root_seed"])
    bind_fixture(
        "roles", MOD.canonical_json(roles_payload_document),
        sum(len(rows) for rows in roles_payload_document["roles"].values()),
    )
    for logical_id, document in mosaic_fixture_documents(plan["root_seed"]).items():
        count = len(document["target_alleles"] if logical_id ==
                    "donor_to_target_provenance" else document["rows"])
        bind_fixture(logical_id, MOD.canonical_json(document), count)
    target_vcf_document = target_vcf_fixture_document(plan["root_seed"])
    bind_fixture("target_vcf", MOD.canonical_json(target_vcf_document),
                 len(target_vcf_document["loci"]))
    anc_document = flare_anc_document(plan["root_seed"])
    bind_fixture("flare_anc", MOD.canonical_json(anc_document), len(anc_document["rows"]))
    global_document = flare_global_document(plan["root_seed"])
    bind_fixture("flare_global", MOD.canonical_json(global_document),
                 len(global_document["rows"]))
    root_bytes = str(plan["root_seed"]).encode()
    tbi_payload = b"TBI\x01" + b"fixture-index" * 4 + root_bytes
    for logical_id in ("ref_tbi", "target_tbi", "flare_anc_tbi"):
        bind_fixture(logical_id, tbi_payload + logical_id.encode(), 1)
    bind_fixture("flare_model", b"FLARE_MODEL\t0.6.0\n" + b"model-fixture\n" * 8 + root_bytes, 1)
    bind_fixture(
        "flare_log",
        b"flare version 0.6.0 [616fcc9d4 03-Nov-2025]\n" +
        b"contract fixture prediction\nAnalysis finished successfully\n" + b"-" * 32 + root_bytes,
        1,
    )
    for logical_id in ("selected_sites_incremental", "selected_sites_overlap_flare"):
        selected_document = selected_sites_document(logical_id, plan["root_seed"])
        payload = MOD.canonical_json(selected_document)
        FIXTURE_PAYLOADS[(plan["plan_id"], logical_id)] = payload
        assets[logical_id] = descriptor(
            logical_id, AMENDMENT["manifest_members"][logical_id]["schema"],
            f"{plan['output_prefix']}{logical_id}.fixture", payload,
            len(selected_document["rows"]),
        )
    target_document = target_rare_document(plan["root_seed"])
    target_payload = MOD.canonical_json(target_document)
    FIXTURE_PAYLOADS[(plan["plan_id"], "target_rare_incremental")] = target_payload
    assets["target_rare_incremental"] = descriptor(
        "target_rare_incremental",
        AMENDMENT["manifest_members"]["target_rare_incremental"]["schema"],
        f"{plan['output_prefix']}target_rare_incremental.fixture", target_payload,
        len(target_document["rows"]),
    )
    flare = AMENDMENT["flare_contract"]
    run_payload = MOD.canonical_json(flare_run_document(plan, assets))
    FIXTURE_PAYLOADS[(plan["plan_id"], "flare_run_manifest")] = run_payload
    assets["flare_run_manifest"] = descriptor(
        "flare_run_manifest", AMENDMENT["manifest_members"]["flare_run_manifest"]["schema"],
        f"{plan['output_prefix']}flare_run_manifest.fixture", run_payload,
    )
    audit_payload = MOD.canonical_json(flare_audit_document(plan, assets))
    FIXTURE_PAYLOADS[(plan["plan_id"], "flare_audit")] = audit_payload
    assets["flare_audit"] = descriptor(
        "flare_audit", AMENDMENT["manifest_members"]["flare_audit"]["schema"],
        f"{plan['output_prefix']}flare_audit.fixture", audit_payload,
    )
    flare_receipt = {
        "stage": "M33_FLARE_TRUTH_BLIND",
        "status": "PASS",
        "plan_id": plan["plan_id"],
        "input_logical_ids": list(flare["input_logical_ids"]),
        "input_descriptor_sha256": {
            logical_id: MOD.canonical_json_sha256(assets[logical_id])
            for logical_id in flare["input_logical_ids"]
        },
        "output_descriptor_sha256": {
            logical_id: MOD.canonical_json_sha256(assets[logical_id])
            for logical_id in flare["output_logical_ids"]
        },
        "flare_version": flare["version"],
        "flare_reported_build": flare["reported_build"],
        "flare_jar_sha256": flare["jar_sha256"],
        "container_digest": flare["container_digest"],
        "parameters": copy.deepcopy(flare["parameters"]),
        "ancestry_order": list(flare["ancestry_order"]),
        "simulation_engine_commit": plan["engine"]["git_commit"],
        "orchestrator_commit": plan["orchestrator"]["git_commit"],
        "generator_source_auth_sha256": plan["generator_source_auth_sha256"],
        "run_manifest_sha256": assets["flare_run_manifest"]["sha256_raw"],
        "interface_sha256": MOD.flare_interface_sha256(assets, AMENDMENT),
        "prediction_sha256": assets["flare_anc"]["sha256_raw"],
        "audit_payload_sha256": assets["flare_audit"]["sha256_raw"],
        "truth_mounted": False,
        "truth_argument_available": False,
        "sealed_before_truth_mount": True,
        "sealed_at_utc": "2026-08-21T20:04:00Z",
    }
    receipt_payload = MOD.canonical_json(flare_receipt)
    FIXTURE_PAYLOADS[(plan["plan_id"], "flare_receipt")] = receipt_payload
    assets["flare_receipt"] = descriptor(
        "flare_receipt", AMENDMENT["manifest_members"]["flare_receipt"]["schema"],
        f"{plan['output_prefix']}flare_receipt.fixture", receipt_payload,
    )
    manifest = {
        "schema_version": "1.0.0",
        "stage": "M33_FINAL_MANIFEST",
        "mode": plan["mode"],
        "root_seed": plan["root_seed"],
        "plan_id": plan["plan_id"],
        "asset_set_id": plan["asset_set_id"],
        "output_prefix": plan["output_prefix"],
        "plan_manifest_sha256": MOD.canonical_json_sha256(plan),
        "assets": assets,
        "semantic_fingerprints": {
            "normalized_full_tree_sha256": digest(f"full-tree-{ordinal}"),
            "normalized_genealogy_sha256": digest(f"genealogy-{ordinal}"),
            "root_independent_source_haplotype_sha256": [
                digest(f"source-haplotype-{ordinal}-{index}")
                for index in range(
                    AMENDMENT["semantic_fingerprints"]
                    ["required_source_haplotype_fingerprint_count"]
                )
            ],
        },
        "predict_bundle": list(AMENDMENT["bundles"]["predict_bundle"]),
        "flare_input_bundle": list(AMENDMENT["bundles"]["flare_input_bundle"]),
        "rare_enabled_model_bundle": list(AMENDMENT["bundles"]["rare_enabled_model_bundle"]),
        "private_truth_bundle": list(AMENDMENT["bundles"]["private_truth_bundle"]),
        "flare_receipt": flare_receipt,
        "final_manifest_sha256": "0" * 64,
    }
    manifest["final_manifest_sha256"] = MOD.canonical_json_sha256({
        key: value for key, value in manifest.items() if key != "final_manifest_sha256"
    })
    return manifest


def asset_payload(plan, logical_id, manifest=None):
    if logical_id == "generator_source_auth":
        return SOURCE_AUTH_BYTES
    if logical_id == "genetic_map":
        return MAP_BYTES
    if (plan["plan_id"], logical_id) in FIXTURE_PAYLOADS:
        return FIXTURE_PAYLOADS[(plan["plan_id"], logical_id)]
    return f"{plan['root_seed']}|{logical_id}".encode()


def observed_payload(descriptor_value, payload):
    return {
        "payload": payload,
        "generation": descriptor_value["gcs_generation"],
        "crc32c": MOD.crc32c_base64(payload),
    }


def rebind_descriptor(descriptor_value, payload, record_count=None):
    rebound = copy.deepcopy(descriptor_value)
    rebound["size_bytes"] = len(payload)
    rebound["sha256_raw"] = hashlib.sha256(payload).hexdigest()
    rebound["crc32c"] = MOD.crc32c_base64(payload)
    if record_count is not None:
        rebound["record_count"] = record_count
    return rebound


def make_ready(plan, manifest):
    plan_payload = MOD.canonical_json(plan)
    plan_descriptor = descriptor(
        "plan_manifest", AMENDMENT["publication_envelopes"]["plan_manifest"]["schema"],
        f"{plan['output_prefix']}plan_manifest.json", plan_payload,
    )
    payload = MOD.canonical_json(manifest)
    final_descriptor = descriptor(
        "final_manifest", AMENDMENT["publication_envelopes"]["final_manifest"]["schema"],
        f"{plan['output_prefix']}final_manifest.json", payload,
    )
    write_ids = AMENDMENT["publication"]["root_objects_before_final_manifest"] + ["final_manifest"]
    ready = {
        "schema_version": "1.0.0",
        "stage": "M33_READY",
        "status": "READY",
        "plan_id": plan["plan_id"],
        "plan_manifest_descriptor": plan_descriptor,
        "final_manifest_descriptor": final_descriptor,
        "final_manifest_reopened_and_verified": True,
        "all_prior_descriptors_reopened_and_verified": True,
        "created_with_if_generation_match": 0,
        "publication_log": [
            {"logical_id": logical_id, "if_generation_match": 0,
             "observed_generation": (
                 plan_descriptor if logical_id == "plan_manifest" else
                 final_descriptor if logical_id == "final_manifest" else
                 manifest["assets"][logical_id]
             )["gcs_generation"]}
            for logical_id in write_ids
        ],
    }
    observations = {
        logical_id: observed_payload(
            descriptor_value, asset_payload(plan, logical_id, manifest)
        )
        for logical_id, descriptor_value in manifest["assets"].items()
    }
    observations["plan_manifest"] = observed_payload(plan_descriptor, plan_payload)
    observations["final_manifest"] = observed_payload(final_descriptor, payload)
    return ready, observations


def reseal_final_manifest(manifest):
    manifest["final_manifest_sha256"] = MOD.canonical_json_sha256({
        key: value for key, value in manifest.items() if key != "final_manifest_sha256"
    })


def reseal_flare_chain(plan, manifest):
    changed = {}
    run_payload = MOD.canonical_json(flare_run_document(plan, manifest["assets"]))
    manifest["assets"]["flare_run_manifest"] = rebind_descriptor(
        manifest["assets"]["flare_run_manifest"], run_payload
    )
    changed["flare_run_manifest"] = run_payload
    audit_payload = MOD.canonical_json(flare_audit_document(plan, manifest["assets"]))
    manifest["assets"]["flare_audit"] = rebind_descriptor(
        manifest["assets"]["flare_audit"], audit_payload
    )
    changed["flare_audit"] = audit_payload
    flare = AMENDMENT["flare_contract"]
    receipt = manifest["flare_receipt"]
    receipt["input_descriptor_sha256"] = {
        logical_id: MOD.canonical_json_sha256(manifest["assets"][logical_id])
        for logical_id in flare["input_logical_ids"]
    }
    receipt["output_descriptor_sha256"] = {
        logical_id: MOD.canonical_json_sha256(manifest["assets"][logical_id])
        for logical_id in flare["output_logical_ids"]
    }
    receipt["run_manifest_sha256"] = manifest["assets"]["flare_run_manifest"]["sha256_raw"]
    receipt["interface_sha256"] = MOD.flare_interface_sha256(manifest["assets"], AMENDMENT)
    receipt["prediction_sha256"] = manifest["assets"]["flare_anc"]["sha256_raw"]
    receipt["audit_payload_sha256"] = manifest["assets"]["flare_audit"]["sha256_raw"]
    receipt_payload = MOD.canonical_json(receipt)
    manifest["assets"]["flare_receipt"] = rebind_descriptor(
        manifest["assets"]["flare_receipt"], receipt_payload
    )
    changed["flare_receipt"] = receipt_payload
    reseal_final_manifest(manifest)
    return changed


def make_person(root, role, ancestry, index):
    person_id = f"m33:{root}:{role.lower()}:{index:04d}"
    haps = [
        {"haplotype_id": f"{person_id}:h{homologue}",
         "canonical_sha256": digest(f"{root}|{role}|{ancestry}|{index}|{homologue}")}
        for homologue in (0, 1)
    ]
    source_sha = None if role == "TARGET" else digest(
        ancestry + "|" + "|".join(sorted(row["canonical_sha256"] for row in haps))
    )
    return {"person_id": person_id, "source_person_sha256": source_sha,
            "ancestry": ancestry, "haplotypes": haps}


def make_roles(root):
    specification = {
        "FREQ": (("AFR", 50), ("EUR", 100), ("ASIA", 150)),
        "REF_LAI": (("AFR", 30), ("EUR", 30), ("ASIA", 30)),
        "DONOR": (("AFR", 256), ("EUR", 256), ("ASIA", 256)),
    }
    roles = {}
    for role, groups in specification.items():
        rows = []
        index = 0
        for ancestry, count in groups:
            for _ in range(count):
                rows.append(make_person(root, role, ancestry, index))
                index += 1
        roles[role] = rows
    roles["TARGET"] = [make_person(root, "TARGET", "MOSAIC", index) for index in range(30)]
    return roles


class M33ExecutionContractTests(unittest.TestCase):
    def test_amendment_preserves_science_and_blocks_real_execution(self):
        self.assertEqual(MOD.sha256_file(ROOT / "conf/m33_asset_execution_amendment.json"),
                         MOD.AMENDMENT_SHA256)
        self.assertFalse(AMENDMENT["execution_authorization"]["real_asset_read"])
        self.assertFalse(AMENDMENT["execution_authorization"]["asset_generation"])
        self.assertFalse(AMENDMENT["execution_authorization"]["forward"])
        self.assertFalse(AMENDMENT["execution_authorization"]["training"])
        self.assertEqual(AMENDMENT["immutable_inputs"]["pre4_contract_sha256"],
                         MOD.BASE_CONTRACT_SHA256)

    def test_plan_id_is_prior_to_outputs_and_roots_fail_closed(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        MOD.validate_plan(plan, AMENDMENT, ASSET_CONTRACT)
        broken = copy.deepcopy(plan)
        broken["input_descriptors"]["generator_source_auth"]["output_sha256"] = digest("future")
        with self.assertRaisesRegex(ValueError, "output descriptor"):
            MOD.derive_plan_id(
                plan["root_seed"], plan["generator_source_auth_sha256"],
                broken["input_descriptors"], plan["rng_seedsequence"], MOD.AMENDMENT_SHA256,
            )
        with self.assertRaisesRegex(ValueError, "unregistered"):
            MOD.derive_plan_id(20260817, digest("auth"), {}, {}, MOD.AMENDMENT_SHA256)

    def test_engine_and_orchestrator_are_distinct(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        plan["orchestrator"]["git_commit"] = MOD.ENGINE_COMMIT
        with self.assertRaisesRegex(ValueError, "conflated"):
            MOD.validate_plan(plan, AMENDMENT, ASSET_CONTRACT)
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        manifest = make_manifest(plan, 0)
        manifest["assets"]["generator_source_auth"]["gcs_generation"] = "1720000000000002"
        with self.assertRaisesRegex(ValueError, "differs from the plan"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        manifest = make_manifest(plan, 0)
        manifest["assets"]["genetic_map"]["record_count"] += 1
        with self.assertRaisesRegex(ValueError, "differs from the plan"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)

    def test_opened_bytes_not_self_reported_hashes(self):
        payload = b"fixture-bytes\n"
        item = descriptor(
            "fixture", "fixture-v1",
            "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/fixture", payload,
        )
        MOD.verify_observed_bytes(item, payload, item["gcs_generation"], MOD.crc32c_base64(payload))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            # Keep the byte count unchanged: this specifically exercises the
            # independently recomputed SHA-256 rather than the size guard.
            mutated = b"fixture-byteZ\n"
            self.assertEqual(len(mutated), len(payload))
            MOD.verify_observed_bytes(item, mutated, item["gcs_generation"],
                                      MOD.crc32c_base64(mutated))
        with self.assertRaisesRegex(ValueError, "generation"):
            MOD.verify_observed_bytes(item, payload, "1720000000000002",
                                      MOD.crc32c_base64(payload))

    def test_roles_are_complete_diploid_and_disjoint(self):
        root = MOD.DEVELOPMENT_ROOTS[0]
        roles = make_roles(root)
        result = MOD.validate_complete_diploid_roles(roles, root, ASSET_CONTRACT)
        self.assertEqual(len(result["TARGET"]), 60)
        roles["TARGET"][0]["haplotypes"].pop()
        with self.assertRaisesRegex(ValueError, "not diploid"):
            MOD.validate_complete_diploid_roles(roles, root, ASSET_CONTRACT)

    def test_freq_selector_is_exhaustive_and_partitioned_from_flare(self):
        people = {f"p{i}" for i in range(300)}
        genotypes = {person: [0, 0] for person in people}
        first_two = sorted(people)[:2]
        for person in first_two:
            genotypes[person] = [0, 1]
        one_carrier = {person: [0, 0] for person in people}
        one_carrier[sorted(people)[0]] = [1, 1]
        ref_minor = {person: [1, 1] for person in people}
        for person in first_two:
            ref_minor[person] = [0, 1]
        variants = [
            {"CHROM": "chr22", "POS": 100, "REF": "A", "ALT": "G", "genotypes": genotypes},
            {"CHROM": "chr22", "POS": 200, "REF": "C", "ALT": "T", "genotypes": one_carrier},
            {"CHROM": "chr22", "POS": 300, "REF": "G", "ALT": "A", "genotypes": ref_minor},
        ]
        selected = MOD.recompute_freq_selected_sites(variants, people)
        self.assertEqual([(row["POS"], row["minor_allele_index"]) for row in selected],
                         [(100, 1), (300, 0)])
        MOD.validate_selected_sites_exhaustive(variants, people, selected)
        flare = {("chr22", 100, "A", "G")}
        incremental, overlap = MOD.partition_selected_sites(selected, flare)
        MOD.validate_selected_site_partition(selected, incremental, overlap, flare)
        self.assertEqual([row["POS"] for row in incremental], [300])
        MOD.validate_rare_enabled_inputs(AMENDMENT["bundles"]["rare_enabled_model_bundle"])
        with self.assertRaisesRegex(ValueError, "exact minimal"):
            MOD.validate_rare_enabled_inputs([
                "selected_sites_all", "target_rare_incremental", "flare_anc", "flare_anc_tbi",
                "genetic_map",
            ])
        with self.assertRaisesRegex(ValueError, "incremental"):
            MOD.validate_selected_site_partition(selected, selected, overlap, flare)
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        manifest = make_manifest(plan, 0)
        _, observations = make_ready(plan, manifest)
        summary = MOD.validate_target_rare_incremental_observations(
            manifest["assets"], observations
        )
        self.assertEqual(len(summary["target_haplotype_ids"]), 60)
        self.assertEqual(len(summary["rows"]), 60 * len(fixture_sites(plan["root_seed"])[0]))
        for mutation in ("missing_product", "overlap", "empty_incremental"):
            assets = copy.deepcopy(manifest["assets"])
            changed_observations = copy.deepcopy(observations)
            if mutation == "empty_incremental":
                document = selected_sites_document("selected_sites_incremental", plan["root_seed"])
                document["rows"] = []
                logical_id = "selected_sites_incremental"
            else:
                document = target_rare_document(plan["root_seed"])
                logical_id = "target_rare_incremental"
                if mutation == "missing_product":
                    document["rows"].pop()
                else:
                    overlap_site = fixture_sites(plan["root_seed"])[1][0]
                    document["rows"][0].update({
                        key: overlap_site[key] for key in ("CHROM", "POS", "REF", "ALT")
                    })
            payload = MOD.canonical_json(document)
            assets[logical_id] = rebind_descriptor(
                assets[logical_id], payload, len(document["rows"])
            )
            changed_observations[logical_id] = observed_payload(assets[logical_id], payload)
            message = {
                "missing_product": "exact TARGET-haplotype",
                "overlap": "nonincremental or overlapping",
                "empty_incremental": "channel is empty",
            }[mutation]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                MOD.validate_target_rare_incremental_observations(assets, changed_observations)

    def test_tree_hashes_separate_mutations_from_genealogy(self):
        tables = {
            "sequence_length": 300.0, "time_units": "generations",
            "nodes": [[1, 2], [0, 0]], "edges": [[100, 300, 0, 1], [0, 100, 0, 1]],
            "individuals": [[1], [0]], "populations": [["EUR"], ["AFR"]],
            "sites": [[200, "C"], [100, "A"]], "mutations": [[1, "T"], [0, "G"]],
            "provenance": ["volatile"], "file_uuid": "volatile",
        }
        first = MOD.normalized_tree_hashes(tables)
        changed = copy.deepcopy(tables)
        changed["mutations"] = [[1, "A"], [0, "G"]]
        second = MOD.normalized_tree_hashes(changed)
        self.assertNotEqual(first["normalized_full_tree_sha256"],
                            second["normalized_full_tree_sha256"])
        self.assertEqual(first["normalized_genealogy_sha256"],
                         second["normalized_genealogy_sha256"])
        volatile = copy.deepcopy(tables)
        volatile["file_uuid"] = "other"
        volatile["provenance"] = ["other"]
        self.assertEqual(first, MOD.normalized_tree_hashes(volatile))
        permuted = copy.deepcopy(tables)
        for table in ("nodes", "edges", "individuals", "populations", "sites", "mutations"):
            permuted[table].reverse()
        self.assertEqual(first, MOD.normalized_tree_hashes(permuted))
        changed_domain = copy.deepcopy(tables)
        changed_domain["sequence_length"] = 301.0
        self.assertNotEqual(first, MOD.normalized_tree_hashes(changed_domain))
        missing_units = copy.deepcopy(tables)
        del missing_units["time_units"]
        with self.assertRaisesRegex(ValueError, "lacks"):
            MOD.normalized_tree_hashes(missing_units)

    def test_truth_is_complete_merge_of_valid_donor_events(self):
        targets = {f"t{i}" for i in range(60)}
        donor_ancestry = {f"d{i}": "AFR" for i in range(60)}
        events = [
            {"target_haplotype_id": f"t{i}", "start_bp": 100, "end_bp": 300,
             "donor_haplotype_id": f"d{i}", "ancestry": "AFR"}
            for i in range(60)
        ]
        truth = MOD.truth_from_mosaic_events(events, targets, donor_ancestry, 100, 300)
        MOD.validate_truth_equals_events(events, truth, targets, donor_ancestry, 100, 300)
        broken = copy.deepcopy(events)
        broken[0]["start_bp"] = 101
        with self.assertRaisesRegex(ValueError, "starts after"):
            MOD.truth_from_mosaic_events(broken, targets, donor_ancestry, 100, 300)
        broken = events[:-1]
        with self.assertRaisesRegex(ValueError, "inventory"):
            MOD.truth_from_mosaic_events(broken, targets, donor_ancestry, 100, 300)
        replacement = copy.deepcopy(events)
        replacement[1]["donor_haplotype_id"] = replacement[0]["donor_haplotype_id"]
        with self.assertRaisesRegex(ValueError, "sampled with replacement"):
            MOD.truth_from_mosaic_events(replacement, targets, donor_ancestry, 100, 300)

    def test_donor_to_target_genotypes_require_complete_product(self):
        targets = {f"t{i}" for i in range(60)}
        events = [
            {"target_haplotype_id": f"t{i}", "start_bp": 100, "end_bp": 300,
             "donor_haplotype_id": f"d{i}", "ancestry": "AFR"}
            for i in range(60)
        ]
        loci = {("chr22", 150, "A", "G"), ("chr22", 250, "C", "T")}
        donor = {(f"d{i}", locus): (i + locus[1]) % 2 for i in range(60) for locus in loci}
        target = {(f"t{i}", locus): donor[(f"d{i}", locus)] for i in range(60) for locus in loci}
        MOD.validate_donor_to_target_genotypes(events, donor, target, targets, loci)
        target.pop(next(iter(target)))
        with self.assertRaisesRegex(ValueError, "complete"):
            MOD.validate_donor_to_target_genotypes(events, donor, target, targets, loci)

    def test_flare_tensor_has_sample_locus_parity_and_simplex(self):
        targets = {f"t{i}" for i in range(60)}
        grid = [("chr22", 150, "A", "G"), ("chr22", 250, "C", "T")]
        rows = [
            {"target_haplotype_id": target, "CHROM": locus[0], "POS": locus[1],
             "REF": locus[2], "ALT": locus[3], "probabilities": [0.2, 0.3, 0.5]}
            for target in sorted(targets) for locus in grid
        ]
        MOD.validate_flare_probability_tensor(rows, targets, grid)
        broken = copy.deepcopy(rows)
        broken[0]["probabilities"] = [0.2, 0.3, 0.6]
        with self.assertRaisesRegex(ValueError, "simplex"):
            MOD.validate_flare_probability_tensor(broken, targets, grid)
        with self.assertRaisesRegex(ValueError, "parity"):
            MOD.validate_flare_probability_tensor(rows[:-1], targets, grid)

    def test_three_root_manifest_and_ready_bundle_passes(self):
        plans = [make_plan(root) for root in MOD.DEVELOPMENT_ROOTS]
        manifests = [make_manifest(plan, index) for index, plan in enumerate(plans)]
        pairs = [make_ready(plan, manifest) for plan, manifest in zip(plans, manifests)]
        auth_observation = pairs[0][1]["generator_source_auth"]
        self.assertEqual(
            MOD.validate_generator_source_auth_observation(
                plans[0]["input_descriptors"]["generator_source_auth"], auth_observation,
                ORCHESTRATOR_COMMIT, ASSET_CONTRACT,
            ),
            SOURCE_AUTH_OBJECT,
        )
        receipt = MOD.validate_root_bundle(
            plans, manifests, [row[0] for row in pairs], [row[1] for row in pairs],
            AMENDMENT, ASSET_CONTRACT,
        )
        self.assertEqual(receipt["status"], MOD.PASS_STATUS)
        self.assertFalse(receipt["real_asset_read"])
        bad_observation = copy.deepcopy(auth_observation)
        bad_observation["generation"] = "1720000000000002"
        with self.assertRaisesRegex(ValueError, "generation"):
            MOD.validate_generator_source_auth_observation(
                plans[0]["input_descriptors"]["generator_source_auth"], bad_observation,
                ORCHESTRATOR_COMMIT, ASSET_CONTRACT,
            )
        corrupt_auth = copy.deepcopy(auth_observation)
        corrupt_auth["payload"] = corrupt_auth["payload"][:-1] + b"X"
        corrupt_auth["crc32c"] = MOD.crc32c_base64(corrupt_auth["payload"])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MOD.validate_generator_source_auth_observation(
                plans[0]["input_descriptors"]["generator_source_auth"], corrupt_auth,
                ORCHESTRATOR_COMMIT, ASSET_CONTRACT,
            )
        invalid_plans = copy.deepcopy(plans)
        invalid_plans[0]["creation_precondition"] = "overwrite"
        missing_source = copy.deepcopy(pairs[0][1])
        del missing_source["generator_source_auth"]
        with self.assertRaisesRegex(ValueError, "source-auth observation absent"):
            MOD.validate_root_bundle(
                invalid_plans, manifests, [row[0] for row in pairs],
                [missing_source, pairs[1][1], pairs[2][1]], AMENDMENT, ASSET_CONTRACT,
            )

    def test_cross_root_tree_genealogy_and_source_hash_reuse_are_rejected(self):
        plans = [make_plan(root) for root in MOD.DEVELOPMENT_ROOTS]
        manifests = [make_manifest(plan, index) for index, plan in enumerate(plans)]
        manifests[1]["semantic_fingerprints"]["normalized_full_tree_sha256"] = (
            manifests[0]["semantic_fingerprints"]["normalized_full_tree_sha256"]
        )
        reseal_final_manifest(manifests[1])
        pairs = [make_ready(plan, manifest) for plan, manifest in zip(plans, manifests)]
        with self.assertRaisesRegex(ValueError, "full tree reused"):
            MOD.validate_root_bundle(
                plans, manifests, [row[0] for row in pairs], [row[1] for row in pairs],
                AMENDMENT, ASSET_CONTRACT,
            )
        manifests = [make_manifest(plan, index) for index, plan in enumerate(plans)]
        manifests[2]["semantic_fingerprints"]["root_independent_source_haplotype_sha256"][0] = (
            manifests[0]["semantic_fingerprints"]["root_independent_source_haplotype_sha256"][0]
        )
        reseal_final_manifest(manifests[2])
        pairs = [make_ready(plan, manifest) for plan, manifest in zip(plans, manifests)]
        with self.assertRaisesRegex(ValueError, "source haplotype reused"):
            MOD.validate_root_bundle(
                plans, manifests, [row[0] for row in pairs], [row[1] for row in pairs],
                AMENDMENT, ASSET_CONTRACT,
            )
        manifests = [make_manifest(plan, index) for index, plan in enumerate(plans)]
        for key in ("size_bytes", "sha256_raw", "crc32c"):
            manifests[1]["assets"]["roles"][key] = manifests[0]["assets"]["roles"][key]
        reseal_final_manifest(manifests[1])
        pairs = [make_ready(plan, manifest) for plan, manifest in zip(plans, manifests)]
        pairs[1][1]["roles"] = observed_payload(
            manifests[1]["assets"]["roles"], asset_payload(plans[0], "roles")
        )
        with self.assertRaisesRegex(ValueError, "non-global person ID|root-specific bytes reused"):
            MOD.validate_root_bundle(
                plans, manifests, [row[0] for row in pairs], [row[1] for row in pairs],
                AMENDMENT, ASSET_CONTRACT,
            )

    def test_final_manifest_cannot_contain_self_or_ready_and_ready_is_last(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        manifest = make_manifest(plan, 0)
        manifest["assets"]["final_manifest"] = copy.deepcopy(manifest["assets"]["generator_manifest"])
        with self.assertRaisesRegex(ValueError, "inventory"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        manifest = make_manifest(plan, 0)
        ready, observation = make_ready(plan, manifest)
        MOD.validate_reopened_flare_run_and_audit(
            plan, manifest["assets"], observation, manifest["flare_receipt"], AMENDMENT
        )
        missing = copy.deepcopy(observation)
        del missing["flare_audit"]
        with self.assertRaisesRegex(ValueError, "lacks an independent observation"):
            MOD.validate_ready(ready, plan, manifest, missing, AMENDMENT, ASSET_CONTRACT)
        corrupt = copy.deepcopy(observation)
        corrupt["flare_audit"]["payload"] = asset_payload(plan, "flare_audit")[:-1] + b"X"
        corrupt["flare_audit"]["crc32c"] = MOD.crc32c_base64(
            corrupt["flare_audit"]["payload"]
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MOD.validate_ready(ready, plan, manifest, corrupt, AMENDMENT, ASSET_CONTRACT)
        corrupt_receipt = copy.deepcopy(observation)
        corrupt_receipt["flare_receipt"]["payload"] = (
            corrupt_receipt["flare_receipt"]["payload"][:-1] + b"X"
        )
        corrupt_receipt["flare_receipt"]["crc32c"] = MOD.crc32c_base64(
            corrupt_receipt["flare_receipt"]["payload"]
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MOD.validate_ready(
                ready, plan, manifest, corrupt_receipt, AMENDMENT, ASSET_CONTRACT
            )
        for logical_id, label in (
            ("flare_run_manifest", "run manifest"), ("flare_audit", "audit")
        ):
            assets = copy.deepcopy(manifest["assets"])
            arbitrary = MOD.canonical_json({"arbitrary": True})
            assets[logical_id] = rebind_descriptor(assets[logical_id], arbitrary)
            changed_observations = copy.deepcopy(observation)
            changed_observations[logical_id] = observation_fn = observed_payload(
                assets[logical_id], arbitrary
            )
            self.assertEqual(observation_fn["payload"], arbitrary)
            with self.subTest(logical_id=logical_id), self.assertRaisesRegex(
                ValueError, f"FLARE {label} key inventory drift"
            ):
                MOD.validate_reopened_flare_run_and_audit(
                    plan, assets, changed_observations, manifest["flare_receipt"], AMENDMENT
                )
        ready["publication_log"][-1], ready["publication_log"][-2] = (
            ready["publication_log"][-2], ready["publication_log"][-1]
        )
        with self.assertRaisesRegex(ValueError, "before READY"):
            MOD.validate_ready(
                ready, plan, manifest, observation, AMENDMENT, ASSET_CONTRACT
            )
        ready, observation = make_ready(plan, manifest)
        ready["publication_log"][0]["observed_generation"] = "1720000000000002"
        with self.assertRaisesRegex(ValueError, "generation differs"):
            MOD.validate_ready(
                ready, plan, manifest, observation, AMENDMENT, ASSET_CONTRACT
            )
        duplicate_uri = make_manifest(plan, 0)
        duplicate_uri["assets"]["roles"]["gcs_uri"] = (
            duplicate_uri["assets"]["tree_sequence"]["gcs_uri"]
        )
        with self.assertRaisesRegex(ValueError, "reuse the same URI"):
            MOD.validate_final_manifest(duplicate_uri, plan, AMENDMENT)
        wrong_count = make_manifest(plan, 0)
        wrong_count["semantic_fingerprints"]["root_independent_source_haplotype_sha256"].pop()
        with self.assertRaisesRegex(ValueError, "2316"):
            MOD.validate_final_manifest(wrong_count, plan, AMENDMENT)

    def test_truth_cannot_enter_predict_or_flare_receipt(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        manifest = make_manifest(plan, 0)
        manifest["predict_bundle"] = manifest["predict_bundle"] + ["truth"]
        with self.assertRaisesRegex(ValueError, "predict bundle"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        manifest = make_manifest(plan, 0)
        manifest["flare_receipt"]["truth_mounted"] = True
        with self.assertRaisesRegex(ValueError, "truth was addressable"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        for field, value, message in (
            ("input_descriptor_sha256", {}, "descriptor-hash inventory"),
            ("output_descriptor_sha256", {}, "output descriptor-hash inventory"),
            ("flare_version", "0.5.2", "version/build"),
            ("flare_jar_sha256", digest("wrong-jar"), "JAR drift"),
            ("container_digest", "sha256:" + "0" * 64, "container drift"),
            ("parameters", {}, "parameter drift"),
            ("ancestry_order", ["EUR", "AFR", "ASIA"], "ancestry-order"),
            ("interface_sha256", digest("wrong-interface"), "interface hash"),
            ("prediction_sha256", digest("wrong-prediction"), "prediction hash"),
            ("audit_payload_sha256", digest("wrong-audit"), "audit-payload"),
        ):
            manifest = make_manifest(plan, 0)
            manifest["flare_receipt"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        manifest = make_manifest(plan, 0)
        self.assertEqual(
            set(manifest["flare_receipt"]["output_descriptor_sha256"]),
            set(AMENDMENT["flare_contract"]["output_logical_ids"]),
        )
        substitute = b"arbitrary-valid-flare-global-output"
        manifest["assets"]["flare_global"] = rebind_descriptor(
            manifest["assets"]["flare_global"], substitute
        )
        with self.assertRaisesRegex(ValueError, "exact output descriptor: flare_global"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        manifest = make_manifest(plan, 0)
        manifest["flare_input_bundle"] = manifest["flare_input_bundle"][:-1]
        with self.assertRaisesRegex(ValueError, "FLARE input bundle"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)
        manifest = make_manifest(plan, 0)
        manifest["rare_enabled_model_bundle"] = [
            "selected_sites_all", "target_rare_incremental", "flare_anc", "flare_anc_tbi",
            "genetic_map"
        ]
        with self.assertRaisesRegex(ValueError, "rare-enabled model bundle"):
            MOD.validate_final_manifest(manifest, plan, AMENDMENT)

    def test_ready_rejects_resigned_arbitrary_mosaic_truth_and_provenance(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        for logical_id in ("mosaic_events", "truth", "donor_to_target_provenance"):
            manifest = make_manifest(plan, 0)
            arbitrary = MOD.canonical_json({
                "schema_version": "1.0.0", "stage": "ARBITRARY_RESIGNED",
                "status": "PASS", "logical_id": logical_id,
            })
            manifest["assets"][logical_id] = rebind_descriptor(
                manifest["assets"][logical_id], arbitrary, 1
            )
            reseal_final_manifest(manifest)
            ready, observations = make_ready(plan, manifest)
            observations[logical_id] = observed_payload(
                manifest["assets"][logical_id], arbitrary
            )
            with self.subTest(logical_id=logical_id), self.assertRaisesRegex(
                ValueError, "key inventory drift"
            ):
                MOD.validate_ready(
                    ready, plan, manifest, observations, AMENDMENT, ASSET_CONTRACT
                )

    def test_ready_binds_minor_presence_to_reconstructed_target_allele(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        manifest = make_manifest(plan, 0)
        ready, observations = make_ready(plan, manifest)
        MOD.validate_ready(
            ready, plan, manifest, observations, AMENDMENT, ASSET_CONTRACT
        )
        base = target_rare_document(plan["root_seed"])
        provenance = mosaic_fixture_documents(plan["root_seed"])[
            "donor_to_target_provenance"
        ]
        target_alleles = {
            (row["haplotype_id"], row["POS"]): row["allele"]
            for row in provenance["target_alleles"]
        }
        ref_minor = next(row for row in base["rows"] if row["POS"] == 300)
        self.assertEqual(
            ref_minor["minor_allele_presence"],
            1 - target_alleles[(ref_minor["target_haplotype_id"], ref_minor["POS"])],
        )
        for label, position in (("ALT_minor", 100), ("REF_minor", 300)):
            document = copy.deepcopy(base)
            row = next(item for item in document["rows"] if item["POS"] == position)
            row["minor_allele_presence"] = 1 - row["minor_allele_presence"]
            payload = MOD.canonical_json(document)
            changed = make_manifest(plan, 0)
            changed["assets"]["target_rare_incremental"] = rebind_descriptor(
                changed["assets"]["target_rare_incremental"], payload,
                len(document["rows"]),
            )
            reseal_final_manifest(changed)
            changed_ready, changed_observations = make_ready(plan, changed)
            changed_observations["target_rare_incremental"] = observed_payload(
                changed["assets"]["target_rare_incremental"], payload
            )
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "minor-allele presence differs"
            ):
                MOD.validate_ready(
                    changed_ready, plan, changed, changed_observations,
                    AMENDMENT, ASSET_CONTRACT,
                )

    def test_ready_rejects_coherently_resigned_arbitrary_flare_outputs(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        cases = {
            "flare_anc": b'{"arbitrary":true}',
            "flare_global": b'{"arbitrary":true}',
            "flare_anc_tbi": b"arbitrary-index-without-magic",
            "flare_model": b"arbitrary-model" * 8,
            "flare_log": b"arbitrary-log" * 10,
        }
        for logical_id, arbitrary in cases.items():
            manifest = make_manifest(plan, 0)
            manifest["assets"][logical_id] = rebind_descriptor(
                manifest["assets"][logical_id], arbitrary, 1
            )
            changed = {logical_id: arbitrary, **reseal_flare_chain(plan, manifest)}
            ready, observations = make_ready(plan, manifest)
            for changed_id, payload in changed.items():
                observations[changed_id] = observed_payload(
                    manifest["assets"][changed_id], payload
                )
            with self.subTest(logical_id=logical_id), self.assertRaises(ValueError):
                MOD.validate_ready(
                    ready, plan, manifest, observations, AMENDMENT, ASSET_CONTRACT
                )

    def test_full_ready_rejects_bool_nan_and_incoherent_rare_metrics(self):
        plan = make_plan(MOD.DEVELOPMENT_ROOTS[0])
        manifest = make_manifest(plan, 0)
        ready, observations = make_ready(plan, manifest)
        ready["created_with_if_generation_match"] = False
        with self.assertRaisesRegex(ValueError, "append-only"):
            MOD.validate_ready(
                ready, plan, manifest, observations, AMENDMENT, ASSET_CONTRACT
            )
        ready, observations = make_ready(plan, manifest)
        ready["publication_log"][0]["if_generation_match"] = False
        with self.assertRaisesRegex(ValueError, "append-only"):
            MOD.validate_ready(
                ready, plan, manifest, observations, AMENDMENT, ASSET_CONTRACT
            )
        for mutation, expected in (("nan", "non-finite JSON"), ("maf", "minor MAF")):
            manifest = make_manifest(plan, 0)
            document = selected_sites_document("selected_sites_incremental", plan["root_seed"])
            document["rows"][0]["minor_maf"] = (
                float("nan") if mutation == "nan" else 0.02
            )
            payload = MOD.canonical_json(document)
            manifest["assets"]["selected_sites_incremental"] = rebind_descriptor(
                manifest["assets"]["selected_sites_incremental"], payload,
                len(document["rows"]),
            )
            reseal_final_manifest(manifest)
            ready, observations = make_ready(plan, manifest)
            observations["selected_sites_incremental"] = observed_payload(
                manifest["assets"]["selected_sites_incremental"], payload
            )
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, expected):
                MOD.validate_ready(
                    ready, plan, manifest, observations, AMENDMENT, ASSET_CONTRACT
                )

    def test_cli_rejects_real_asset_flag_before_any_asset_path_is_accepted(self):
        completed = subprocess.run(
            [
                sys.executable, str(ROOT / "bin/m33_asset_execution_contract.py"),
                "--asset-contract", str(ROOT / "conf/m33_asset_manifest_contract.json"),
                "--amendment", str(ROOT / "conf/m33_asset_execution_amendment.json"),
                "--allow-real-assets",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("real asset access is blocked", completed.stderr)


if __name__ == "__main__":
    unittest.main()
