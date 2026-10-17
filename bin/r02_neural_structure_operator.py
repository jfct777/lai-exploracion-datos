#!/usr/bin/env python3
"""VM operator for the bounded HUMAN structure instrumentation.

It verifies the exact VM/data-disk identity and labels, runs the resumable
Nextflow graph under a 10-hour compute/staging ceiling, publishes only compact
run products create-only, verifies GCS metadata, and powers the VM off.  Any
failure also powers off while retaining the separately attached data disk.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import tarfile


TRAINING_REPODIGEST = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
    "m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99"
)
TRAINING_IMAGE_ID = "sha256:574bf28f8456852fcbfa3a2f15eaedfe6bf3833f3c6dd252ab905fd585aaa32a"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.digest()


def describe(kind, name, project, zone):
    return json.loads(subprocess.check_output(
        ["gcloud", "compute", kind, "describe", name, "--project", project,
         "--zone", zone, "--format=json"], text=True))


def verify_resources(identity):
    vm = describe("instances", identity["vm"]["name"], identity["project"], identity["zone"])
    boot = describe("disks", identity["boot_disk"]["name"], identity["project"], identity["zone"])
    disk = describe("disks", identity["data_disk"]["name"], identity["project"], identity["zone"])
    require(str(vm["id"]) == identity["vm"]["id"] and
            str(boot["id"]) == identity["boot_disk"]["id"] and
            str(disk["id"]) == identity["data_disk"]["id"],
            "live GCP numeric resource identity differs")
    require(vm.get("labels", {}).get("team") == "frank" and
            boot.get("labels", {}).get("team") == "frank" and
            disk.get("labels", {}).get("team") == "frank",
            "VM, boot disk and data disk must each have team=frank")
    require(disk.get("sizeGb") == "300" and disk.get("status") == "READY",
            "data disk size/status differs")
    require(all(any(name in item["source"] for item in vm.get("disks", []))
                for name in (identity["boot_disk"]["name"], identity["data_disk"]["name"])),
            "exact boot/data disks are not attached to exact VM")


def publish(path, uri):
    size = path.stat().st_size
    sha256 = sha(path).hex()
    md5 = base64.b64encode(sha(path, "md5")).decode()
    result = subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
                             "--content-md5=" + md5, str(path), uri],
                            text=True, capture_output=True)
    reused = result.returncode != 0
    require(not reused or "HTTPError 412:" in (result.stderr or ""),
            "create-only result publication failed")
    remote = json.loads(subprocess.check_output(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"], text=True))
    require(int(remote["size"]) == size and
            remote.get("md5_hash", remote.get("md5Hash")) == md5,
            "published result metadata differs")
    return {"uri": uri, "generation": str(remote["generation"]),
            "bytes": size, "sha256": sha256, "md5_base64": md5,
            "operation": "CREATE_ONLY_VERIFIED"}


def verify_training_image(image, repo):
    require(image == TRAINING_REPODIGEST,
            "training container must be the frozen Torch RepoDigest")
    inspected = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", image], text=True))[0]
    require(inspected.get("Id") == TRAINING_IMAGE_ID,
            "training container local image ID differs")
    require(TRAINING_REPODIGEST in inspected.get("RepoDigests", []),
            "training container RepoDigest differs")
    deps = (Path(repo) / ".claude/runs/r02-corrections-20261006j/"
            "parallel-integral-20261009au/neural-structure/deployment/python-deps")
    require((deps / "scipy-1.17.0.dist-info").is_dir()
            and (deps / "scikit_learn-1.8.0.dist-info").is_dir(),
            "frozen offline scientific Python dependency tree is absent")
    probe = subprocess.check_output(
        ["docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "4g",
         "-v", f"{repo}:{repo}:ro", "-e", f"PYTHONPATH={deps}",
         "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
         "-e", "MKL_NUM_THREADS=1", "-e", "NUMEXPR_NUM_THREADS=1",
         "--entrypoint", "python3",
         image, "-c",
         "import json,numpy,scipy,sklearn,torch; from scipy import sparse; from sklearn.linear_model import Ridge; "
         "assert sparse.csr_matrix([[1]]).nnz==1 and Ridge().alpha==1.0; "
         "print(json.dumps({'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},sort_keys=True))"],
        text=True)
    versions = json.loads(probe)
    require(versions == {"numpy": "2.4.6", "scipy": "1.17.0",
                         "sklearn": "1.8.0", "torch": "2.12.1+cpu"},
            "frozen training runtime dependency versions differ")
    return {"repo_digest": image, "image_id": TRAINING_IMAGE_ID, **versions}


def retain_run_evidence(run_root, controller_cwd, frozen_manifest_path,
                        frozen_manifest_sha256, runtime):
    """Classify and preserve only useful continuation/reproduction products."""
    provenance = run_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    nextflow_log = controller_cwd / ".nextflow.log"
    if nextflow_log.is_file():
        (provenance / "nextflow.log").write_bytes(nextflow_log.read_bytes())
    bootstrap_log = Path("/var/log/r02-neural-structure-bootstrap.log")
    if bootstrap_log.is_file():
        # The bootstrap log can grow after this point; this named prefix records
        # exactly what was available before publication.
        (provenance / "bootstrap.log.prefix").write_bytes(bootstrap_log.read_bytes())
    manifest = json.loads(Path(frozen_manifest_path).read_text())
    require(sha(frozen_manifest_path).hex() == frozen_manifest_sha256,
            "frozen deployment manifest SHA256 differs before retention")
    restored = manifest.get("large_gcs_inputs")
    require(isinstance(restored, dict)
            and set(restored) == {"stores", "target_objects"}
            and len(restored["stores"]) == 22
            and len(restored["target_objects"]) == 5,
            "frozen restored-input inventory differs")
    retention = {
        "schema": "r02_neural_structure_retention_v1",
        "status": "USEFUL_PRODUCTS_COMPLETE",
        "scope": "BOUNDED_ENGINEERING_INSTRUMENTATION_ONLY",
        "pending_work": [],
        "unclassified_paths": [],
        "useful_run_tree": "run",
        "useful_products": [
            "final admission, data/model configuration and completion gate",
            "three arm measurements, training receipts and final outputs",
            "durable checkpoints including the forced RARE step-1 recovery evidence",
            "Nextflow log and available bootstrap-log prefix under run/provenance",
        ],
        "restored_inputs": restored,
        "restored_input_policy": (
            "FULL22 stores and five target objects are verified restorations of the "
            "generation-pinned GCS sources in the frozen manifest; the archive references "
            "those sources and does not duplicate the 195.6 GB payload"
        ),
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "code_policy": "exact code/configuration is in the separately published frozen deployment bundle",
        "runtime": runtime,
        "excluded": {
            "operating_system": "regenerable bootstrap image; no continuation value",
            "restored_store_bytes": "equivalent generation-pinned GCS objects",
            "restored_target_bytes": "equivalent generation-pinned GCS objects",
            "nextflow_work_cache": "not required after all three final two-update measurements and checkpoints are retained",
            "temporary_files": "regenerable and not scientific products",
        },
        "biological_evaluation": False,
    }
    (run_root / "RETENTION.json").write_text(
        json.dumps(retention, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("identity", "identity-sha256", "repo", "packet", "request-index",
                "prior-full22-gate", "prior-full22-gate-sha256", "run-root",
                "work-dir", "gcloud-config", "publication-uri", "target-catalogue",
                "target-catalogue-sha256", "target-root", "training-container",
                "frozen-manifest", "frozen-manifest-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    identity_path = Path(args.identity)
    require(sha(identity_path).hex() == args.identity_sha256, "identity SHA256 differs")
    identity = json.loads(identity_path.read_text())
    require(identity.get("schema") == "r02_neural_structure_live_identity_v1",
            "live identity schema differs")
    os.environ["CLOUDSDK_CONFIG"] = args.gcloud_config
    os.environ["NXF_OFFLINE"] = "true"
    verify_resources(identity)
    repo, run_root, work = Path(args.repo), Path(args.run_root), Path(args.work_dir)
    target_root = Path(args.target_root)
    command = ["timeout", "--signal=TERM", "--kill-after=180s", "36000s",
               "nextflow", "-C", "/dev/null", "run",
               str(repo / "workflows/r02_neural_structure_instrumentation.nf"),
               "-with-trace", str(run_root / "provenance/trace.txt"),
               "-with-report", str(run_root / "provenance/report.html"),
               "-with-timeline", str(run_root / "provenance/timeline.html"),
               "-work-dir", str(work), "-resume", "--packet", args.packet,
               "--store_request_index", args.request_index, "--repo", str(repo),
               "--run_root", str(run_root), "--gcloud_config", args.gcloud_config,
               "--prior_full22_gate", args.prior_full22_gate,
               "--prior_full22_gate_sha256", args.prior_full22_gate_sha256,
               "--training_container", args.training_container,
               "--target_root", str(target_root)]
    outcome = "FAILED_PRESERVE_DISK"
    controller = None
    try:
        run_root.parent.mkdir(parents=True, exist_ok=True)
        (run_root / "provenance").mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        if not (target_root / "trainer-target.json").is_file():
            require(not target_root.exists(), "partial target stage requires inspection")
            target_root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["python3", str(repo / "bin/r02_neural_target_stage.py"),
                            "--catalogue", args.target_catalogue,
                            "--catalogue-sha256", args.target_catalogue_sha256,
                            "--view-id", "chr01_part01__BLOCK_AND_LD_MARGIN",
                            "--out", str(target_root)], check=True)
        subprocess.run(["docker", "pull", args.training_container], check=True)
        runtime = verify_training_image(args.training_container, repo)
        version = subprocess.check_output(["nextflow", "-version"], text=True)
        require("version 26.04.6" in version, "exact bootstrapped Nextflow 26.04.6 required")
        controller = subprocess.Popen(command, cwd=run_root.parent, start_new_session=True)
        returncode = controller.wait()
        require(returncode == 0, "Nextflow failed or exceeded the 10-hour execution ceiling")
        gate = json.loads((run_root / "INSTRUMENTATION_COMPLETE.json").read_text())
        require(gate.get("status") == "PASS_ENGINEERING_ONLY"
                and gate.get("evaluate_opened") is False
                and len(gate.get("measurements", [])) == 3,
                "final three-arm gate differs")
        retain_run_evidence(run_root, run_root.parent, args.frozen_manifest,
                            args.frozen_manifest_sha256, runtime)
        archive = run_root.parent / "neural-structure-instrumentation-20261009au.tar.gz"
        if not archive.exists():
            partial = archive.with_suffix(archive.suffix + ".partial")
            if partial.exists():
                partial.unlink()  # operator-owned incomplete publication artifact
            with tarfile.open(partial, "x:gz") as output:
                output.add(run_root, arcname="run", recursive=True)
            partial.replace(archive)
        publication = publish(archive, args.publication_uri.rstrip("/") + "/run.tar.gz")
        receipt = {"schema": "r02_neural_structure_publication_v1", "status": "VERIFIED",
                   "result": publication, "gate_sha256": sha(run_root / "INSTRUMENTATION_COMPLETE.json").hex(),
                   "identity_sha256": args.identity_sha256, "evaluate_opened": False}
        receipt_path = run_root.parent / "PUBLICATION.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        publish(receipt_path, args.publication_uri.rstrip("/") + "/PUBLICATION.json")
        outcome = "COMPLETE_VERIFIED_SAFE_TO_REVIEW_RETIREMENT"
    finally:
        if controller is not None and controller.poll() is None:
            os.killpg(controller.pid, signal.SIGTERM)
            try:
                controller.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(controller.pid, signal.SIGKILL); controller.wait()
        run_root.parent.mkdir(parents=True, exist_ok=True)
        status = run_root.parent / "OPERATOR_EXIT.json"
        status.write_text(json.dumps({"schema": "r02_neural_structure_operator_exit_v1",
                                      "status": outcome,
                                      "data_disk_retained": True},
                                     sort_keys=True, indent=2) + "\n")
        subprocess.run(["shutdown", "-h", "now"], check=False)
    return 0 if outcome.startswith("COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
