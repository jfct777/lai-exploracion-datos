#!/usr/bin/env python3
"""Bind the 12 authenticated LAI materials to reciprocal truth-blind requests."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requirements", type=Path, required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--genotype-root", type=Path, required=True)
    p.add_argument("--grid-chr1", type=Path, required=True)
    p.add_argument("--grid-chr22", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    requirements = json.loads(a.requirements.read_text())
    if requirements.get("schema") != "r02_lai_context_requirements_v1" or len(requirements.get("cases", [])) != 12:
        raise ValueError("the authenticated 12-case requirements manifest is required")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    tasks = []
    for case in requirements["cases"]:
        case_id, chrom = case["case_id"], case["chromosome"]
        material = Path(case["material_path"])
        manifest = material / "manifest.json"
        if sha(manifest) != case["material_manifest_sha256"]:
            raise ValueError(f"material changed: {case_id}")
        source = a.genotype_root / "inputs" / case_id
        truth = source / "truth.private.tsv.gz"
        grid = a.grid_chr1 if chrom == "1" else a.grid_chr22
        for direction in ("FOLD_A_TO_FOLD_B", "FOLD_B_TO_FOLD_A"):
            roles = a.prepared_root / direction / f"{case_id}.target-roles.tsv"
            paths = {"query_grid": grid, "query_grid_receipt": Path(str(grid) + ".json"),
                     "persons": material / "persons.tsv", "roles": roles, "truth": truth}
            if not all(path.is_file() for path in paths.values()):
                raise ValueError(f"missing integrated input for {case_id} {direction}")
            request = {"schema": "r02_lai_truthblind_context_request_v2", "case_id": case_id,
                       "direction": direction, "material_manifest_sha256": case["material_manifest_sha256"]}
            for key, path in paths.items():
                request[key + "_path"] = str(path.resolve())
                request[key + "_path_sha256"] = sha(path)
            task_id = f"{case_id}.{direction}"
            request_path = a.output_dir / f"{task_id}.request.json"
            request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
            tasks.append({"task_id": task_id, "request_path": str(request_path.resolve()),
                          "request_sha256": sha(request_path)})
    batch = {"schema": "r02_lai_truthblind_context_batch_v2", "status": "ALLOWED_PREPARATION_ONLY",
             "case_count": 12, "reciprocal_task_count": len(tasks), "tasks": tasks,
             "allowed_for": ["QUERY_OVERLAY_PREPARATION", "SOFTWARE_VALIDATION"],
             "forbidden_for": ["SCIENTIFIC_TRAINING", "MODEL_SELECTION", "BIOLOGICAL_CONCLUSION"]}
    (a.output_dir / "batch-manifest.json").write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
