#!/usr/bin/env python3
"""Freeze one analytical SOURCE_VALID opening for an authenticated REF catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from m27f_validation_contract import build_validation_plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--claim-registry-dir", type=Path, required=True)
    parser.add_argument("--claim-key", required=True)
    parser.add_argument("--claim-uri", required=True)
    parser.add_argument("--gcloud", default="gcloud")
    parser.add_argument("--claim-py", type=Path, required=True)
    parser.add_argument("--validation-contract-py", type=Path, required=True)
    parser.add_argument("--valid-audit-py", type=Path, required=True)
    parser.add_argument("--ref-audit-py", type=Path, required=True)
    parser.add_argument("--m27e-py", type=Path, required=True)
    parser.add_argument("--bridge-py", type=Path, required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--projection-public", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--ref-support-private", type=Path, required=True)
    parser.add_argument("--ref-primary-catalog", type=Path, required=True)
    parser.add_argument("--ref-public", type=Path, required=True)
    parser.add_argument("--ref-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def require_manifest_hash(
    manifest: dict[str, object], section: str, name: str, expected: str
) -> None:
    values = manifest.get(section)
    if not isinstance(values, dict) or values.get(name) != expected:
        raise ValueError(
            f"STOP_PROVENANCE: manifest {section}.{name} does not authenticate"
        )


def publish_gcs_claim(candidate: Path, claim_uri: str, gcloud: str) -> None:
    if not claim_uri.startswith("gs://"):
        raise ValueError("STOP_VALIDATION_OPENING: claim URI must use gs://")
    upload = subprocess.run(
        [
            gcloud,
            "storage",
            "cp",
            "--if-generation-match=0",
            str(candidate),
            claim_uri,
        ],
        text=True,
        capture_output=True,
    )
    if upload.returncode == 0:
        return
    existing = candidate.with_name(f".{candidate.name}.remote-existing")
    download = subprocess.run(
        [gcloud, "storage", "cp", claim_uri, str(existing)],
        text=True,
        capture_output=True,
    )
    if download.returncode != 0:
        raise ValueError(
            "STOP_VALIDATION_OPENING: atomic GCS claim failed and existing claim "
            f"could not be authenticated: {upload.stderr[:500]}"
        )
    try:
        if existing.read_bytes() != candidate.read_bytes():
            raise ValueError(
                "STOP_VALIDATION_REOPENING: a different canonical GCS claim already exists"
            )
    finally:
        existing.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.run_id.strip():
        raise ValueError("STOP_VALIDATION_OPENING: empty run id")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", args.claim_key):
        raise ValueError("STOP_VALIDATION_OPENING: invalid claim key")
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    projection = json.loads(args.projection_public.read_text(encoding="utf-8"))
    projection_manifest = json.loads(
        args.projection_manifest.read_text(encoding="utf-8")
    )
    ref_public = json.loads(args.ref_public.read_text(encoding="utf-8"))
    ref_manifest = json.loads(args.ref_manifest.read_text(encoding="utf-8"))

    support_contract = prereg.get("support_contract", {})
    if (
        args.claim_key != support_contract.get("validation_claim_key")
        or args.claim_uri != support_contract.get("validation_claim_uri")
        or args.container_image
        != prereg.get("projection_contract", {}).get("container")
        or args.container_digest != args.container_image.split("@")[-1]
    ):
        raise ValueError("STOP_VALIDATION_OPENING: claim identity or container differs")

    if (
        prereg.get("stage") != "M27F_REF_VALID_SUPPORT_AUDIT"
        or prereg.get("version") != 2
        or projection.get("stage") != "M27F_REF_VALID_MECHANICAL_PROJECTION"
        or projection.get("decision") != "GO_REF_SUPPORT_AUDIT"
        or any(value != "PASS" for value in projection.get("gates", {}).values())
        or projection.get("source_test_projection_created") is not False
        or projection.get("source_test_samples_in_projected_outputs") != 0
        or any(
            int(row.get("n_records_with_nonempty_info", -1)) != 0
            for row in projection.get("projections", {}).values()
        )
        or projection_manifest.get("stage")
        != "M27F_REF_VALID_MECHANICAL_PROJECTION"
        or ref_public.get("stage") != "M27F_REF_SUPPORT_SELECTION"
        or ref_manifest.get("stage") != "M27F_REF_SUPPORT_SELECTION"
    ):
        raise ValueError("STOP_VALIDATION_OPENING: upstream stage or gate differs")

    support_hash = sha256_file(args.ref_support_private)
    catalog_hash = sha256_file(args.ref_primary_catalog)
    public_hash = sha256_file(args.ref_public)
    prereg_hash = sha256_file(args.preregistration)
    projection_hash = sha256_file(args.projection_public)
    projection_manifest_hash = sha256_file(args.projection_manifest)
    if (
        ref_public.get("private_ref_support_sha256") != support_hash
        or ref_public.get("private_primary_catalog_sha256") != catalog_hash
    ):
        raise ValueError("STOP_VALIDATION_OPENING: REF public receipt differs")

    require_manifest_hash(
        ref_manifest, "sha256", args.ref_support_private.name, support_hash
    )
    require_manifest_hash(
        ref_manifest, "sha256", args.ref_primary_catalog.name, catalog_hash
    )
    require_manifest_hash(ref_manifest, "sha256", args.ref_public.name, public_hash)
    require_manifest_hash(
        ref_manifest, "inputs", args.preregistration.name, prereg_hash
    )
    require_manifest_hash(
        projection_manifest, "sha256", args.projection_public.name, projection_hash
    )
    projection_files = {
        "DISCOVERY_CORE": "m27f_discovery_core.chr22.private.bcf",
        "REF_TRAIN": "m27f_ref.chr22.private.bcf",
        "SOURCE_VALID": "m27f_valid.chr22.private.bcf",
    }
    for role, filename in projection_files.items():
        require_manifest_hash(
            projection_manifest,
            "sha256",
            filename,
            projection["projections"][role]["bcf_sha256"],
        )
    require_manifest_hash(
        ref_manifest, "inputs", args.projection_public.name, projection_hash
    )
    require_manifest_hash(
        ref_manifest,
        "inputs",
        args.projection_manifest.name,
        projection_manifest_hash,
    )

    authorized = ref_public.get("decision") == "GO_VALID_SUPPORT_AUDIT"
    if ref_public.get("decision") not in {
        "GO_VALID_SUPPORT_AUDIT",
        "STOP_REF_NO_TRANSFERABLE_SUPPORT",
        "INCONCLUSIVE_REF_CALLABILITY",
    }:
        raise ValueError("STOP_VALIDATION_OPENING: unexpected REF decision")
    validation_plan, validation_plan_hash = build_validation_plan(
        {
            "claim": args.claim_py,
            "validation_contract": args.validation_contract_py,
            "valid_audit": args.valid_audit_py,
            "ref_audit": args.ref_audit_py,
            "m27e_audit": args.m27e_py,
            "bridge": args.bridge_py,
        },
        args.container_image,
        args.container_digest,
        prereg,
    )

    receipt = {
        "stage": "M27F_VALIDATION_OPENING",
        "decision": (
            "VALIDATION_OPENING_FROZEN"
            if authorized
            else "VALIDATION_NOT_AUTHORIZED"
        ),
        "run_id": args.run_id,
        "registry_claim_key": args.claim_key,
        "registry_claim_uri": args.claim_uri,
        "authorized_analytical_openings": 1 if authorized else 0,
        "resume_same_run_same_hashes_is_same_opening": True,
        "projection_public_sha256": projection_hash,
        "projection_manifest_sha256": projection_manifest_hash,
        "valid_projection_bcf_sha256": projection["projections"]["SOURCE_VALID"][
            "bcf_sha256"
        ],
        "split_private_sha256": prereg["upstream_contract"][
            "m27f_split_private_sha256"
        ],
        "ref_support_private_sha256": support_hash,
        "ref_primary_catalog_sha256": catalog_hash,
        "ref_public_sha256": public_hash,
        "ref_manifest_sha256": sha256_file(args.ref_manifest),
        "preregistration_sha256": prereg_hash,
        "primary_ref_min_atomic_units": int(
            prereg["support_contract"]["primary_ref_min_atomic_units"]
        ),
        "required_valid_atomic_units": int(
            prereg["support_contract"]["required_valid_atomic_units"]
        ),
        "ref_decision": ref_public["decision"],
        "validation_plan": validation_plan,
        "validation_plan_sha256": validation_plan_hash,
        "source_valid_genotypes_read_to_create_receipt": False,
        "source_test_genotypes_opened": False,
        "sample_ids_emitted": False,
        "population_labels_emitted": False,
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not authorized:
        args.out.write_bytes(payload)
        os.chmod(args.out, 0o600)
        return receipt
    candidate = args.out.with_name(f".{args.out.name}.candidate")
    candidate.write_bytes(payload)
    os.chmod(candidate, 0o600)
    publish_gcs_claim(candidate, args.claim_uri, args.gcloud)

    args.claim_registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.claim_registry_dir / f"{args.claim_key}.json"
    try:
        descriptor = os.open(
            registry_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        if registry_path.read_bytes() != payload:
            raise ValueError(
                "STOP_VALIDATION_REOPENING: a different canonical claim already exists"
            )
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    args.out.write_bytes(payload)
    os.chmod(args.out, 0o600)
    candidate.unlink()
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
