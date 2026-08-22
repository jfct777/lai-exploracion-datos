#!/usr/bin/env python3
"""Authenticate the M33 controller and launch only the synthetic Tabix KAT."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import urllib.request
from pathlib import Path
import re


CONTROLLER = "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
SOURCE_FILES = (
    "bin/m33_gcs_append_only.py",
    "bin/m33_batch_postflight.py",
    "bin/m33_infra_kat_contract.py",
    "bin/m33_storage_policy.py",
    "bin/m33_tabix_kat.py",
    "bin/m33_tabix_kat_cloud_runner.py",
    "bin/m33_tabix_kat_controller.py",
    "conf/m33_infra_kat_authorization.json",
    "conf/m33_m0_materializer_contract.json",
    "conf/m33_storage_namespace_policy.json",
    "conf/m33_tabix_kat.config",
    "conf/gcp/m33_batch_submitter_role.yaml",
    "containers/m33-controller/Dockerfile",
    "containers/m33-tabix/Dockerfile",
    "modules/33_TABIX_KAT.nf",
    "workflows/m33_tabix_kat.nf",
)
FORBIDDEN_CREDENTIAL_ENV = {
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
    "GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS",
}
METADATA_EMAIL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/email"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def controller_email() -> str:
    request = urllib.request.Request(METADATA_EMAIL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        require(response.headers.get("Metadata-Flavor") == "Google", "metadata response is unauthenticated")
        return response.read().decode("ascii").strip()


def load_contract_module(path: Path):
    specification = importlib.util.spec_from_file_location("m33_infra_kat_contract", path)
    require(specification is not None and specification.loader is not None, "contract module cannot load")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_auth(repo_root: Path) -> tuple[Path, str]:
    source_auth_path = repo_root / "conf" / "m33_tabix_kat_source_auth.json"
    payload = json.loads(source_auth_path.read_text(encoding="utf-8"))
    require(
        set(payload) == {"schema_version", "stage", "status", "files"},
        "source authorization schema drifted",
    )
    require(payload["schema_version"] == "1.0.0", "source authorization version drifted")
    require(payload["stage"] == "M33_TABIX_KAT_SOURCE_AUTH", "source authorization stage drifted")
    require(payload["status"] == "AUTHORIZED_EXACT_SOURCE_BUNDLE", "source bundle is not authorized")
    files = payload["files"]
    require(isinstance(files, dict) and set(files) == set(SOURCE_FILES), "source file inventory drifted")
    for relative in SOURCE_FILES:
        require(re.fullmatch(r"[0-9a-f]{64}", str(files[relative])) is not None, "invalid source hash")
        require(sha256_file(repo_root / relative) == files[relative], f"source hash drifted: {relative}")
    return source_auth_path, sha256_file(source_auth_path)


def reject_credential_overrides(environment: dict[str, str]) -> None:
    present = sorted(key for key in FORBIDDEN_CREDENTIAL_ENV if environment.get(key))
    require(not present, f"credential override is forbidden: {present}")


def nextflow_version() -> str:
    completed = subprocess.run(
        ["nextflow", "-version"], check=True, capture_output=True, text=True
    )
    for line in completed.stdout.splitlines():
        if line.strip().startswith("version "):
            return line.strip().split()[1]
    raise ValueError("Nextflow version was not reported")


def nf_google_version(nextflow_home: Path) -> str:
    plugins = nextflow_home / "plugins"
    matches = sorted(path.name for path in plugins.glob("nf-google-*") if path.is_dir())
    require(matches == ["nf-google-1.27.3"], f"nf-google plugin inventory drifted: {matches}")
    return "1.27.3"


def write_exclusive(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def nextflow_environment(base: dict[str, str], values: dict[str, str]) -> dict[str, str]:
    environment = {key: value for key, value in base.items() if key not in FORBIDDEN_CREDENTIAL_ENV}
    environment.update(values)
    environment["NXF_OFFLINE"] = "true"
    environment["NXF_SYNTAX_PARSER"] = "v1"
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--controller-image", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    require(RUN_ID_RE.fullmatch(args.run_id) is not None, "invalid M33 run ID")
    require(
        re.fullmatch(
            r"us-central1-docker\.pkg\.dev/uspbr-242713/dnabr-lai/m33-controller@sha256:[0-9a-f]{64}",
            args.controller_image,
        ) is not None,
        "controller image must use the immutable M33 controller repository digest",
    )
    repo_root = Path(__file__).resolve().parents[1]
    authorization_path = repo_root / "conf" / "m33_infra_kat_authorization.json"
    storage_policy_path = repo_root / "conf" / "m33_storage_namespace_policy.json"
    m0_contract_path = repo_root / "conf" / "m33_m0_materializer_contract.json"
    contract_validator_path = repo_root / "bin" / "m33_infra_kat_contract.py"
    workflow_path = repo_root / "workflows" / "m33_tabix_kat.nf"
    config_path = repo_root / "conf" / "m33_tabix_kat.config"
    reject_credential_overrides(dict(os.environ))
    source_auth_path, source_auth_sha = validate_source_auth(repo_root)

    observed_controller = controller_email()
    require(observed_controller == CONTROLLER, "launcher is not using the M33 controller service account")
    module = load_contract_module(contract_validator_path)
    authorization = module.load_json(authorization_path)
    module.validate_authorization(
        authorization,
        storage_policy=storage_policy_path,
        m0_contract=m0_contract_path,
        require_published_digest=True,
    )
    expected_image = (
        f"{authorization['runtime']['oci_repository']}@{authorization['runtime']['oci_digest']}"
    )
    require(args.runtime_image == expected_image, "runtime image differs from authorization")
    observed_nextflow = nextflow_version()
    require(observed_nextflow == authorization["runtime"]["nextflow_version"], "Nextflow version drifted")
    observed_nf_google = nf_google_version(Path(os.environ.get("NXF_HOME", "/.nextflow")))
    require(observed_nf_google == authorization["runtime"]["nf_google_version"], "nf-google version drifted")

    receipt = {
        "stage": "M33_TABIX_KAT_CONTROLLER",
        "status": "PASS_CONTROLLER_IDENTITY_AND_AUTHORIZATION",
        "controller_service_account": observed_controller,
        "authorization_sha256": module.sha256_file(authorization_path),
        "controller_image": args.controller_image,
        "runtime_image": args.runtime_image,
        "nextflow_version": observed_nextflow,
        "nf_google_version": observed_nf_google,
        "run_id": args.run_id,
        "work_prefix": (
            "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/"
            f"{args.run_id}/"
        ),
        "source_auth_sha256": source_auth_sha,
        "real_asset_read": False,
    }
    write_exclusive(args.receipt, receipt)
    environment = nextflow_environment(dict(os.environ), {
        "DNABR_M33_INFRA_KAT": "1",
        "DNABR_RUN_ID": args.run_id,
        "DNABR_M33_TABIX_IMAGE": args.runtime_image,
        "DNABR_M33_CONTROLLER_RECEIPT": str(args.receipt.resolve()),
        "DNABR_M33_AUTHORIZATION": str(authorization_path),
        "DNABR_M33_SOURCE_AUTH": str(source_auth_path),
    })
    command = [
        "nextflow", "-C", str(config_path), "run", str(workflow_path),
    ]
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        return completed.returncode

    publisher = load_contract_module(repo_root / "bin" / "m33_gcs_append_only.py")
    postflight_module = load_contract_module(repo_root / "bin" / "m33_batch_postflight.py")
    token = publisher.access_token()
    gcs = publisher.AppendOnlyGCS(token)
    kat_object = (
        "frank/lai-exploracion-datos/runs/"
        f"{args.run_id}/m33_tabix_kat.receipt.json"
    )
    kat_record = gcs.record_for_existing("teams-usp", kat_object)
    kat_receipt = json.loads(gcs.reopen_record(kat_record))
    batch_api = postflight_module.BatchAPI(token)
    job_summaries = postflight_module.stable_inventory(
        batch_api.list_jobs,
        run_id=args.run_id,
    )
    jobs = [batch_api.get_job(job["name"]) for job in job_summaries]
    postflight = postflight_module.make_postflight(
        jobs=jobs,
        run_id=args.run_id,
        runtime_image=args.runtime_image,
        controller_image=args.controller_image,
        kat_receipt=kat_receipt,
    )
    final = publisher.finalize_candidate(
        run_id=args.run_id,
        storage_policy=storage_policy_path,
        storage_validator=repo_root / "bin" / "m33_storage_policy.py",
        postflight=postflight,
    )
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
