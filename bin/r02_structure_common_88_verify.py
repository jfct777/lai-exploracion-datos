#!/usr/bin/env python3
"""Verify the complete 88-view target preparation without opening EVALUATE."""
import argparse, hashlib, json
from pathlib import Path

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def load(path): return json.loads(Path(path).read_text())

def verify(root, output):
    root = Path(root).resolve(); output = Path(output).resolve()
    require(root.is_dir() and output.parent == root and not output.exists(), "New runtime completion receipt required")
    all_requests = load(root / "materialized-88/batch.json")
    admitted = load(root / "config/admissions/batch.json")
    require(all_requests.get("view_count") == 88 and all_requests.get("evaluate_allowed") is False
            and all_requests.get("penalty_present") is False, "Exact penalty-free 88 request materialization required")
    require(admitted.get("view_count") == 87 and admitted.get("evaluate_allowed") is False
            and admitted.get("status") == "EXACT_PENALTY_FREE_TARGET_REQUESTS_ADMITTED",
            "Exact 87-new admitted batch required")
    materialized = {row["id"]: row["request_sha256"] for row in all_requests["views"]}
    frozen = {path.stem: sha(path) for path in (root / "config/requests").glob("*.json")}
    require(len(materialized) == len(frozen) == 88 and materialized == frozen,
            "Staged 88 requests differ from the reviewed frozen allowlist")
    products = []
    for row in admitted["views"]:
        view = row["id"]
        bridge_dir = root / "results/bridges" / ("common_" + view)
        neural_dir = root / "results/neural-targets" / ("neural_" + view)
        bridge_manifest = bridge_dir / "manifest.json"; neural_manifest = neural_dir / "manifest.json"
        bridge = load(bridge_manifest); neural = load(neural_manifest)
        require(bridge.get("schema") == "r02_k03_fit_select_arrays_v1"
                and bridge.get("stage") == "FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS"
                and bridge.get("block_id") + "__" + bridge.get("exclusion_scope") == view
                and bridge.get("evaluate_targets_computed") is False
                and bridge.get("alpha_selection_performed") is False
                and bridge.get("penalty_present") is False,
                "Bridge is not the exact penalty-free FIT/SELECT product: " + view)
        require(neural.get("schema") == "r02_k03_neural_target_adapter_result_v1"
                and neural.get("status") == "FIT_SELECT_TARGETS_COMPLETE_NOT_TRAINING_ADMISSION"
                and neural.get("evaluate_targets_computed") is False
                and neural.get("training_allowed") is False
                and set(neural.get("targets", {})) == {"FIT", "SELECT"},
                "Neural adapter output differs: " + view)
        for directory, manifest, kind in ((bridge_dir, bridge, "bridge"), (neural_dir, neural, "neural")):
            files = []
            for path in sorted(directory.iterdir()):
                require(path.is_file() and not path.is_symlink(), "Unexpected output entry: " + str(path))
                files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
            products.append({"view_id": view, "kind": kind, "directory": str(directory), "files": files,
                             "manifest_sha256": sha(directory / "manifest.json")})
    existing = all_requests.get("existing_reused") or load(root / "config/REQUEST_BATCH.json").get("existing_reused")
    require(existing.get("view_id") == "chr01_part01__WHOLE_CHROMOSOME"
            and existing["neural_target_archive"]["generation"] == "1791422735032792"
            and existing["neural_target_archive"]["sha256"] == "cb45ac0579f242b01d3e3caba20fe162f6670c16a9f1c9296020b36e38e78474",
            "Existing 88th neural target identity differs")
    result = {"schema": "r02_common_88_preparation_complete_v1",
              "status": "COMMON_88_PREPARATION_COMPLETE_NO_EVALUATE_NO_TRAINING",
              "materialized_requests": 88, "new_bridge_products": 87,
              "new_neural_target_products": 87, "reused_neural_target_products": 1,
              "existing_reused": existing, "products": products,
              "evaluate_targets_computed": False, "training_allowed": False,
              "model_fits_run": False}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(json.dumps(verify(args.runtime_root, args.output), sort_keys=True))
