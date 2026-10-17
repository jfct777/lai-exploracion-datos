#!/usr/bin/env python3
"""Build authenticated Structure and LAI input catalogues without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


RUN = Path(".claude/runs/r02-corrections-20261006j")
AUTOSOMES = tuple(str(value) for value in range(1, 23))
DIRECTIONS = ("FOLD_A_TO_FOLD_B", "FOLD_B_TO_FOLD_A")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def structure_targets(root: Path) -> dict:
    completion_path = root / RUN / "common-territories-20261008ak/cloud/recovery-v2/watcher/completed-from-gcs.json"
    completion = read(completion_path)
    require(completion.get("status") == "COMMON_88_PREPARATION_COMPLETE_NO_EVALUATE_NO_TRAINING"
            and completion.get("new_neural_targets") == 87 and completion.get("views") == 88,
            "common88 completion differs")
    grouped = {}
    for item in completion["files"]:
        match = re.search(r"/products/neural-targets/([^/]+)/([^/]+)$", item.get("uri", ""))
        if match:
            grouped.setdefault(match.group(1), {})[match.group(2)] = {
                key: item[key] for key in ("uri", "generation", "bytes", "sha256", "md5_base64")
            }
    expected = {"manifest.json", "FIT.target.private.npz", "FIT.target.receipt.json",
                "SELECT.target.private.npz", "SELECT.target.receipt.json"}
    require(len(grouped) == 87 and all(set(files) == expected for files in grouped.values()),
            "87-target authenticated object inventory differs")
    targets = [{"view_id": view.removeprefix("neural_"), "source": "common88_direct_objects",
                "objects": files, "roles": ["FIT", "SELECT"], "evaluate": False}
               for view, files in sorted(grouped.items())]

    old_manifest = root / RUN / "parallel-launch-20261008ac/structure_adapter/human/published/neural-targets/manifest.json"
    old = read(old_manifest)
    require(old.get("schema") == "r02_k03_neural_target_adapter_result_v1" and
            set(old.get("targets", {})) == {"FIT", "SELECT"} and
            old.get("evaluate_targets_computed") is False, "reused target manifest differs")
    publication = read(root / RUN / "parallel-launch-20261008ac/structure_adapter/human/PUBLICATION.json")
    require(publication.get("all_verified") is True and len(publication["objects"]) == 2,
            "reused target publication differs")
    targets.append({
        "view_id": "chr01_part01__WHOLE_CHROMOSOME",
        "source": "verified_archive_member",
        "manifest": file_ref(old_manifest),
        "archive": {key: publication["objects"][0][key]
                    for key in ("uri", "generation", "bytes", "sha256", "md5_base64")},
        "roles": ["FIT", "SELECT"], "evaluate": False,
    })
    require(len({target["view_id"] for target in targets}) == 88, "target views are not unique")
    return {
        "schema": "r02_structure_88_target_catalogue_v1",
        "status": "AUTHENTICATED_88_FIT_SELECT_TARGETS_NO_TRAINING",
        "views": 88, "new_direct_views": 87, "reused_archive_views": 1,
        "training_allowed": False, "evaluate_targets_present": False,
        "consumer": "bin/r02_context_train.py",
        "completion_source": file_ref(completion_path), "targets": targets,
    }


def structure_stores(root: Path, stage_root: Path, publication_prefix: str | None) -> dict:
    integration_path = root / RUN / "neural-human-integration-20261007v/REAL_INPUT_CATALOGUE_REQUEST.json"
    integration = read(integration_path)
    descriptors = {str(item["chromosome"]): item for item in integration["store_manifests"]}
    require(set(descriptors) == set(AUTOSOMES), "22 manifest descriptors required")
    staging_path = root / RUN / "k04-completion-20261007v/package/event-staging.json"
    staging = read(staging_path)
    staged = {str(item["chromosome"]): item for item in staging["chromosomes"]}
    archives = {chrom: item["archive"] for chrom, item in staged.items()}
    require(set(archives) == set(AUTOSOMES), "22 pinned archives required")
    reserve_bytes = 10 * 2**30
    requests = []
    total_chunks = total_extracted = total_compressed = 0
    max_payload = None
    for chrom in AUTOSOMES:
        descriptor = descriptors[chrom]
        manifest_path = Path(descriptor["manifest_path"])
        require(sha(manifest_path) == descriptor["manifest_sha256"], "manifest copy differs")
        manifest = read(manifest_path)
        require(str(manifest["chromosome"]) == chrom and manifest["n_people"] == 2619,
                "event store identity differs")
        chunks = [item for channel in ("common", "rare") for item in manifest["channels"][channel]]
        extracted = sum(item["bytes"] for item in chunks)
        archive = archives[chrom]
        payload = extracted + archive["bytes"]
        total_chunks += len(chunks); total_extracted += extracted; total_compressed += archive["bytes"]
        if max_payload is None or payload > max_payload["payload_bytes"]:
            max_payload = {"chromosome": chrom, "payload_bytes": payload,
                           "extracted_bytes": extracted, "archive_bytes": archive["bytes"]}
        requests.append({
            "schema": "r02_neural_store_validation_request_v1", "chromosome": chrom,
            "archive": {key: archive[key] for key in ("uri", "generation", "bytes", "sha256")},
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": descriptor["manifest_sha256"],
            "stage_root": str(stage_root.resolve()),
            "minimum_free_bytes": payload + reserve_bytes,
            "reserve_free_bytes": reserve_bytes,
            "publish_uri": (publication_prefix.rstrip("/") + f"/chr{int(chrom):02d}.validation.json")
                           if publication_prefix else None,
        })
    return {
        "schema": "r02_structure_store_validation_campaign_v1",
        "status": "READY_FOR_SEQUENTIAL_INPUT_VALIDATION_NOT_LAUNCHED",
        "chromosomes": 22, "people": 2619, "chunks": total_chunks,
        "compressed_bytes": total_compressed, "extracted_bytes": total_extracted,
        "maximum_simultaneous_payload": max_payload,
        "post_task_free_space_reserve_bytes": reserve_bytes,
        "execution": {"max_forks": 1, "cpus": 1, "memory_bytes": 3 * 2**30,
                      "time_limit": "REQUIRED_AT_DEPLOYMENT_FROM_MEASURED_THROUGHPUT",
                      "release_after": "PUBLISHED_RECEIPT_SIZE_MD5_GENERATION_VERIFIED",
                      "preserve_on_failure": True},
        "consumer_validation": "bin/r02_event_mmap_reader.py:MappedEventChromosome",
        "future_training_consumer": "bin/r02_context_train.py",
        "training_allowed": False, "evaluate_opened": False,
        "source": {"integration": file_ref(integration_path), "archives": file_ref(staging_path)},
        "requests": requests,
    }


def _flare_candidates(root: Path) -> dict[str, list[dict]]:
    candidates = {}
    base = root / RUN / "lai-full-readout-20261007l"
    for path in sorted(base.glob("*/chr*-gen*/wrapper.json")):
        wrapper = read(path)
        case = wrapper.get("case_id")
        generation = int(wrapper.get("generation", path.parent.name.split("gen")[-1]))
        # The exact wrapper and its source receipt authenticate the archived FLARE products.
        candidates.setdefault(case, []).append({
            "initial_generation": generation, "wrapper": file_ref(path),
            "wrapper_source": file_ref(path.with_name("wrapper-source.json")),
            "archive_recovery": wrapper.get("legacy_block"),
        })
    require(len(candidates) == 12 and all({x["initial_generation"] for x in values} == {5, 10, 16}
                                         for values in candidates.values()),
            "FLARE 12-case x 3-generation inventory differs")
    return candidates


def lai_catalogue(root: Path) -> dict:
    integration = root / RUN / "parallel-next-20261008an/lai/integration"
    batch_path = integration / "requests/batch-manifest.json"
    batch = read(batch_path)
    require(batch.get("case_count") == 12 and batch.get("reciprocal_task_count") == 24,
            "truth-independent task grid differs")
    published = read(integration / "PUBLICATION_COMPLETE.json")
    require(published.get("status") == "COMPLETE" and published.get("published_objects") == 123
            and len(published.get("objects", [])) == 123,
            "truthblind publication differs")
    published_by_uri = {item["uri"]: item for item in published["objects"]}
    published_by_path = {str(Path(item["path"]).resolve()): item for item in published["objects"]}

    source_batch_path = root / RUN / "lai-channel-inventory-20261008al/BATCH.json"
    source_batch = read(source_batch_path)
    reference = {}
    for job in source_batch["jobs"]:
        if job["role"] == "REF_TRAIN":
            reference[str(job["chromosome"])] = job["vcf"]
    require(set(reference) == set(AUTOSOMES), "22 REF_TRAIN detail sources required")

    summaries_complete_path = root / RUN / "parallel-next-20261008an/lai-summary-run/runtime/complete.json"
    summaries_complete = read(summaries_complete_path)
    summary_files = {item["name"]: item for item in summaries_complete["files"]
                     if item["name"].endswith(".tsv")}
    require(len(summary_files) == 132 and summaries_complete.get("truth_read") is False and
            summaries_complete.get("training_allowed") is False, "132 summaries differ")
    flare = _flare_candidates(root)
    tasks = []
    seen = set()
    for entry in batch["tasks"]:
        request_path = Path(entry["request_path"])
        require(sha(request_path) == entry["request_sha256"], "LAI request changed")
        request = read(request_path)
        task_id = entry["task_id"]
        case, direction = task_id.rsplit(".", 1)
        require(direction in DIRECTIONS and request["case_id"] == case and
                request["direction"] == direction, "LAI task identity differs")
        chrom = re.search(r"\.chr([0-9]+)\.", case).group(1)
        split = re.search(r"\.(FOLD_[AB])\.r[123]$", case).group(1)
        validated = read(root / RUN / f"parallel-next-20261008an/lai/prepared/{direction}/validated-split.json")
        assignment = [x for x in validated["assignments"] if x["case_id"] == case]
        require(len(assignment) == 1 and assignment[0]["role"] in ("FIT", "SELECT"),
                "FIT/SELECT role missing")
        roles_path = Path(request["roles_path"])
        require(sha(roles_path) == request["roles_path_sha256"], "role table changed")
        result_dir = integration / "nextflow-output" / task_id
        receipt_path = result_dir / "receipt.json"
        receipt = read(receipt_path)
        require(receipt["request_sha256"] == entry["request_sha256"] and
                receipt["truth_read_for_query_geometry"] is False and
                receipt["roles"] == [assignment[0]["role"]], "query receipt role/truth differs")
        overlay_path = result_dir / "query-overlay.npz"
        require(sha(overlay_path) == receipt["output_sha256"], "query overlay changed")
        overlay_pub = published_by_path.get(str(overlay_path.resolve()))
        receipt_pub = published_by_path.get(str(receipt_path.resolve()))
        require(overlay_pub is not None and receipt_pub is not None, "query product not published")
        event_path = root / RUN / f"parallel-next-20261008an/lai/prepared/{case}.event-candidates.tsv"
        require(event_path.is_file(), "rare event candidate table missing")
        material = root / RUN / f"lai-genotype-integration-20261007v/results/{case}/material/manifest.json"
        require(material.is_file() and sha(material) == request["material_manifest_sha256"],
                "target genotype material differs")
        common_summary = summary_files[f"chr{chrom}.REFERENCE.candidate-counts.tsv"]
        spectrum = summary_files[f"chr{chrom}.REFERENCE.minor-count-spectrum.tsv"]
        tasks.append({
            "task_id": task_id, "case_id": case, "direction": direction,
            "development_role": assignment[0]["role"], "chromosome": chrom,
            "donor_fold": split, "evaluate": False,
            "target_genotypes": file_ref(material),
            "channels": {
                "RARE_EVENT_CANDIDATES": file_ref(event_path),
                "QUERY": {"product": overlay_pub, "receipt": receipt_pub,
                          "model_facing_digest": receipt["predictor_arrays_sha256"],
                          "supervision_digest": receipt["target_arrays_sha256"],
                          "truth_used_for_query_geometry": False},
                "REFERENCE_SUMMARY": {"candidate_counts": common_summary,
                                      "minor_count_spectrum": spectrum,
                                      "status": "SUPPORT_COUNTS_NOT_CHANNEL_ADMISSION"},
                "REFERENCE_DETAIL": {**reference[chrom], "role": "REF_TRAIN",
                                     "people": 753, "materialization": "PENDING_VALIDATED_ADAPTER_RUN"},
                "COMMON": {"status": "CANDIDATE_SUPPORT_AVAILABLE_SELECTION_NOT_ADMITTED",
                           "reason": "baseline-minus-event is not a justified common-locus definition"},
            },
            "flare_comparator_candidates": flare[case],
            "flare_selection": None,
            "roles": file_ref(roles_path),
            "training_allowed": False,
        })
        seen.add((case, direction))
    require(len(tasks) == 24 and len(seen) == 24 and
            {direction for _, direction in seen} == set(DIRECTIONS), "reciprocal grid differs")
    return {
        "schema": "r02_lai_neural_input_catalogue_v1",
        "status": "AUTHENTICATED_24_RECIPROCAL_INPUT_TASKS_TRAINING_BLOCKED",
        "cases": 12, "tasks": 24, "directions": list(DIRECTIONS),
        "roles_present": ["FIT", "SELECT"], "evaluate_present": False,
        "query_products_published": 24, "reference_detail_sources": 22,
        "reference_summary_products": 132, "flare_candidates": 36,
        "training_allowed": False,
        "blocking_decisions": [
            "common-locus admission for the neural COMMON channel",
            "FLARE comparator selection rule across generations 5, 10 and 16",
            "scientifically justified neural configuration and selection rule",
            "extension from 12 existing cases to the predeclared complete 396-cell LAI universe",
        ],
        "truth_policy": "query geometry is finalized before truth; target arrays are supervision only",
        "sources": {"truthblind_batch": file_ref(batch_path),
                    "truthblind_publication": file_ref(integration / "PUBLICATION_COMPLETE.json"),
                    "reference_inventory": file_ref(source_batch_path),
                    "summary_completion": file_ref(summaries_complete_path)},
        "tasks_detail": tasks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--validation-publication-prefix")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(); output = args.output_dir.resolve()
    targets = structure_targets(root)
    stores = structure_stores(root, args.stage_root, args.validation_publication_prefix)
    lai = lai_catalogue(root)
    write(output / "structure-targets.json", targets)
    write(output / "structure-store-validation.json", stores)
    request_dir = output / "store-requests"
    manifest_dir = output / "store-manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    request_index = []
    for request in stores["requests"]:
        label = f"chr{int(request['chromosome']):02d}.json"
        write(request_dir / label, request)
        shutil.copyfile(request["manifest_path"], manifest_dir / label)
        request_index.append({"chromosome": request["chromosome"],
                              "request": file_ref(request_dir / label),
                              "manifest": file_ref(manifest_dir / label)})
    write(output / "STORE_REQUESTS.json", {"schema": "r02_neural_store_request_index_v1",
          "requests": request_index})
    write(output / "lai-inputs.json", lai)
    deployment = output / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    producer = root / ".claude/runs/r02-genomic-event-real-20261005j/package/frozen/bin"
    current_sources = [root / "workflows/r02_neural_input_validation.nf",
                       root / "bin/r02_neural_store_validate.py",
                       root / "bin/r02_neural_target_stage.py",
                       root / "bin/r02_event_mmap_reader.py",
                       root / "bin/r02_lai_reference_detail.py",
                       root / "bin/r02_neural_input_catalogue.py"]
    # These seven modules define EventChromosome.code_hashes().  The stores
    # authenticate the exact producer snapshot, so using the repository HEAD
    # here would make a valid historical store fail its schema/code check.
    producer_sources = [producer / name for name in (
        "r02_genomic_event_inputs.py",
        "r02_structure_materialize.py",
        "r02_structure_increment.py",
        "r02_structure_input_plan.py",
        "r02_genomic_pair_evidence.py",
        "r02_structure_regions.py",
        "r02_genomic_event_cache.py",
    )]
    sources = current_sources + producer_sources
    for source in sources:
        shutil.copyfile(source, deployment / source.name)
    write(deployment / "SOURCES.json", {"schema": "r02_neural_input_deployment_sources_v1",
          "sources": [{**file_ref(deployment / source.name),
                       "source_path": str(source.relative_to(root))}
                      for source in sources]})
    write(output / "CATALOGUE.json", {
        "schema": "r02_neural_inputs_catalogue_v1",
        "status": "PREPARED_NO_TRAINING_NO_STAGING_LAUNCHED",
        "structure": {"targets": 88, "stores": 22, "people": 2619},
        "lai": {"cases": 12, "reciprocal_tasks": 24, "evaluate": False},
        "training_allowed": False,
        "files": {name: file_ref(output / name) for name in
                  ("structure-targets.json", "structure-store-validation.json", "lai-inputs.json")},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
