#!/usr/bin/env python3
"""Authenticate completed chromosome checkpoints and prepare an all-autosome contract.

This is a read-only GCS consumer, not an evaluator or a Nextflow launcher.
Inventory mode downloads only JSON provenance and verifies remote generations.
Stage mode additionally authenticates compressed bytes on local storage and can
emit the existing Nextflow input contract, but only for all 22 autosomes.
Scientific row validation remains mandatory inside the existing reducer.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import r02_biological_checkpoint as checkpoint
import r02_publish_evidence as publication

evaluation = checkpoint.evaluation
pipeline = publication.pipeline
require = checkpoint.require
AUTOSOMES = tuple(map(str, range(1, 23)))
PREFIX = "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/checkpoints/R02_20260930/"
SCHEMA = "r02_biological_aggregate_preparation_v1"
MAX_JSON_BYTES = 4 * 1024**2


class MissingObject(FileNotFoundError):
    """Only an explicit object-not-found, never an authentication failure."""


class GCS:
    def __init__(self):
        self._identity = None

    def identity(self):
        if self._identity is not None:
            return dict(self._identity)
        result = {}
        for field in ("account", "project"):
            value = os.environ.get("CLOUDSDK_CORE_" + field.upper())
            if not value:
                value = subprocess.check_output(["gcloud", "config", "get-value", "core/" + field],
                                                text=True, timeout=30).strip()
            require(value and value != "(unset)", "Explicit effective GCP account/project required")
            result[field] = value
        self._identity = result
        return result

    def flags(self):
        return ["--" + key + "=" + value for key, value in self.identity().items()]

    def metadata(self, uri):
        result = subprocess.run(["gcloud", "storage", "objects", "describe", uri,
                                 "--format=json", *self.flags()], capture_output=True, text=True, timeout=120)
        if result.returncode:
            if re.search(r"(?:HTTPError 404:|NotFoundException: 404|not found: 404\.|does not exist\.)", result.stderr):
                raise MissingObject(uri)
            raise RuntimeError("GCS metadata request failed; not treated as a missing checkpoint: " + result.stderr)
        return json.loads(result.stdout)

    def download(self, uri, generation, destination):
        subprocess.run(["gcloud", "storage", "cp", uri + "#" + generation, str(destination), *self.flags()],
                       check=True, capture_output=True, text=True, timeout=7200)


def safe_uri(uri):
    require(isinstance(uri, str) and uri.startswith(PREFIX), "Object outside checkpoint prefix")
    publication.safe_relative(uri.removeprefix(PREFIX))
    return uri


def operational_hashes():
    """Seal every loaded project Python helper, excluding third-party packages."""
    root = Path(__file__).resolve().parent
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.parent == root and path.suffix == ".py":
                paths.add(path)
    return {path.name: checkpoint.sha256(path) for path in sorted(paths)}


def metadata_record(uri, metadata):
    safe_uri(uri)
    size, generation = metadata.get("size"), metadata.get("generation")
    require(not isinstance(size, bool) and re.fullmatch(r"[0-9]+", str(size)), "Invalid remote size")
    require(not isinstance(generation, bool) and re.fullmatch(r"[1-9][0-9]*", str(generation)),
            "Missing remote generation")
    md5 = metadata.get("md5_hash", metadata.get("md5Hash"))
    try:
        valid_md5 = isinstance(md5, str) and len(base64.b64decode(md5, validate=True)) == 16
    except ValueError:
        valid_md5 = False
    require(valid_md5, "Missing or invalid remote MD5")
    return dict(uri=uri, generation=str(generation), bytes=int(size), md5_base64=md5)


def recheck_remote(client, record):
    current = metadata_record(record["uri"], client.metadata(record["uri"]))
    require(all(current[k] == record[k] for k in current), "Remote object generation/size/MD5 changed")


def fetch(client, record, target, *, min_free_bytes=0, json_only=False):
    """Generation-pinned, create-only download; interrupted objects are not reused."""
    target = Path(target)
    require(target.parent.is_dir() and not target.is_symlink(), "Unsafe staging target")
    recheck_remote(client, record)
    if json_only:
        require(record["bytes"] <= MAX_JSON_BYTES, "Oversized JSON provenance object")
    if not target.exists():
        require(shutil.disk_usage(target.parent).free >= record["bytes"] + min_free_bytes,
                "Staging would violate free-space reserve")
        part = target.with_name(target.name + ".partial")
        require(not os.path.lexists(part), "Partial staging object exists; explicit recovery required")
        # Exclusive creation prevents a pre-existing symlink from becoming a download destination.
        with part.open("xb"):
            pass
        client.download(record["uri"], record["generation"], part)
        digest = pipeline.publication_digests(part)
        require(all(digest[k] == record[k] for k in ("bytes", "md5_base64")) and
                ("sha256" not in record or digest["sha256"] == record["sha256"]),
                "Downloaded object hash/size mismatch")
        os.link(part, target)
        part.unlink()
    target = publication.safe_local(target)
    digest = pipeline.publication_digests(target)
    require(all(digest[k] == record[k] for k in ("bytes", "md5_base64")) and
            ("sha256" not in record or digest["sha256"] == record["sha256"]),
            "Existing staged object differs")
    recheck_remote(client, record)
    return dict(record, sha256=digest["sha256"], path=str(target))


def validate_index(index):
    require(index.get("schema") == "r02_all_autosome_checkpoint_source_index_v1", "Unsupported source index")
    require(index.get("no_local_accumulated_bp_threshold") is True and
            index.get("biological_result") is False, "Wrong checkpoint evidence scope")
    require(type(index.get("n_samples")) is int and index["n_samples"] >= 2 and
            type(index.get("n_configurations")) is int and index["n_configurations"] > 0,
            "Invalid cohort or configuration count")
    publication.validate_sha(index.get("sample_ids_sha256"))
    require(index.get("scientific_code_sha256") == checkpoint.code_hashes(), "Scientific code differs from index")
    seen, prefixes = [], set()
    for campaign in index.get("campaigns", []):
        prefix = safe_uri(campaign["prefix"])
        require(prefix not in prefixes, "Duplicate campaign prefix")
        prefixes.add(prefix)
        require(campaign.get("completion_relative") == "completed.json", "Unexpected completion marker")
        chromosomes = campaign.get("chromosomes", [])
        require(chromosomes and all(c in AUTOSOMES for c in chromosomes), "Invalid campaign chromosomes")
        seen.extend(chromosomes)
    require(len(seen) == 22 and set(seen) == set(AUTOSOMES), "Source index must cover 22 unique autosomes")


def authenticate_settings(settings, index):
    require(settings.get("schema") == "r02_biological_aggregate_settings_v1", "Unsupported settings")
    p = settings["scientific_parameters"]
    require(p["chromosomes"] == ",".join(AUTOSOMES) and
            p["expected_samples"] == index["n_samples"] and
            p["expected_configurations"] == index["n_configurations"] and
            p["sample_ids_sha256"] == index["sample_ids_sha256"], "Settings/index scope mismatch")
    thresholds = tuple(map(float, p["kinship_thresholds"].split(",")))
    edges = tuple(map(int, p["edge_thresholds_bp"].split(",")))
    require(thresholds and len(set(thresholds)) == len(thresholds) and
            all(math.isfinite(t) and 0 <= t <= .5 for t in thresholds), "Invalid kinship thresholds")
    require(0 in edges and len(set(edges)) == len(edges) and all(t >= 0 for t in edges),
            "Invalid accumulated-bp thresholds")
    files = [(p["sample_ids"], p["sample_ids_sha256"]), (p["pcrelate_file"], p["kinship_sha256"]),
             (p["study"]["path"], p["study"]["sha256"])]
    for path, digest in files:
        publication.validate_sha(digest)
        require(checkpoint.sha256(publication.safe_local(path)) == digest, "Auxiliary input SHA256 mismatch")
    samples = evaluation.diagnostics.sweep.samples_from(p["sample_ids"], p["expected_samples"])
    study = evaluation.read_study(p["study"]["path"], samples, p["kinship_sha256"], thresholds)
    require(study["provenance"]["persons_sha256"] == p["study"]["persons_sha256"], "Study persons mismatch")
    consumer = settings["consumer"]
    source = publication.safe_local(consumer["source_bin"], directory=True)
    expected_code = index["scientific_code_sha256"]
    require(set(consumer["source_sha256"]) == set(expected_code) | {"preprocess_storage_guard.py"},
            "Consumer source inventory mismatch")
    for name, digest in consumer["source_sha256"].items():
        require(checkpoint.sha256(publication.safe_local(source / publication.safe_relative(name))) ==
                publication.validate_sha(digest), "Consumer frozen source hash mismatch")
        if name in expected_code:
            require(digest == expected_code[name], "Consumer source differs from checkpoint science")
    workflow = consumer["workflow"]
    require(Path(workflow["path"]).name == "r02_biological_checkpoint_aggregate.nf" and
            checkpoint.sha256(publication.safe_local(workflow["path"])) == publication.validate_sha(workflow["sha256"]),
            "Consumer workflow hash mismatch")
    return p


def receipt_files(item, prefix):
    chrom = item["chrom"]
    expected = {"checkpoint.json", f"chr{chrom}.segments.jsonl.gz", f"chr{chrom}.ids.gz"}
    files = item.get("files", [])
    require(len(files) == 3 and {f["name"] for f in files} == expected, "Incomplete or duplicate file inventory")
    folder = prefix + "/chr" + chrom.zfill(2) + "/"
    for record in files:
        require(record["uri"] == folder + record["name"], "Checkpoint object URI/chromosome mismatch")
        require(type(record.get("bytes")) is int and record["bytes"] >= 0, "Invalid receipt size")
        publication.validate_sha(record.get("sha256"))
        metadata_record(record["uri"], dict(size=record["bytes"], generation=record.get("generation"),
                                           md5Hash=record.get("md5_base64")))
    manifest = next(r for r in files if r["name"] == "checkpoint.json")
    require(manifest["sha256"] == item["checkpoint_sha256"], "Checkpoint digest differs from receipt")
    proof = item.get("publication", {})
    require(proof.get("schema") == "r02_technical_checkpoint_publication_v1" and
            proof.get("status") == "PUBLISHED_VERIFIED_PRIVATE_CHECKPOINT" and proof.get("chrom") == chrom and
            proof.get("checkpoint_sha256") == manifest["sha256"] and proof.get("files") == files and
            proof.get("biological_result") is False, "Invalid per-chromosome publication receipt")
    return files


def validate_checkpoint_header(record, chrom, index, settings):
    """Validate provenance headers; the reducer separately validates every scientific row."""
    require(record.get("schema") == checkpoint.SCHEMA and record.get("status") == checkpoint.STATUS and
            record.get("chrom") == chrom, "Invalid checkpoint status/chromosome")
    require(record.get("n_samples") == index["n_samples"] and
            record.get("sample_ids_sha256") == index["sample_ids_sha256"] and
            record.get("n_configurations") == index["n_configurations"] and
            record.get("code_sha256") == index["scientific_code_sha256"], "Checkpoint cohort/configuration/code mismatch")
    require(record.get("sort_fields") == list(checkpoint.SORT_FIELDS) and
            record.get("no_local_accumulated_bp_threshold") is True and record.get("biological_result") is False,
            "Checkpoint sort/evidence scope mismatch")
    expected = {"segments": f"chr{chrom}.segments.jsonl.gz", "identifiers": f"chr{chrom}.ids.gz"}
    require(record.get("streams") == expected and set(record.get("outputs", {})) == set(expected.values()),
            "Checkpoint stream inventory mismatch")
    registry = record["registry"]
    require(registry["chromosomes"] == [chrom] and len(registry["sources"]) == 1 and
            registry["sources"][0]["chrom"] == chrom, "Checkpoint source registry mismatch")
    require(registry["identity"] == settings["expected_identity"], "Checkpoint allele/origin/missingness identity mismatch")
    require(registry["configs"] == settings["expected_configurations_ledger"], "Checkpoint configuration ledger mismatch")
    converted = ({k: str(v) for k, v in r.items()} for r in registry["configs"].values())
    require(evaluation.configuration_records(converted, index["n_configurations"]) == registry["configs"],
            "Invalid configuration records")
    require(registry["ibd_chromosomes"] in ([], [chrom]), "Invalid IBD coverage registry")
    source = registry["sources"][0]
    publication.validate_sha(source["sha256"])
    require(type(source["n_chains"]) is int and source["n_chains"] >= 0, "Invalid source chain count")


def prepare(index_path, expected_index_sha256, settings_path, expected_settings_sha256,
            output_dir, *, mode="inventory", min_free_gib=12, max_stage_gib=16, client=None):
    require(mode in {"inventory", "stage"}, "Invalid mode")
    require(all(math.isfinite(v) and v >= 0 for v in (min_free_gib, max_stage_gib)), "Invalid storage budget")
    output = Path(output_dir).absolute()
    require(not os.path.lexists(output), "Output exists; never overwrite a preparation receipt")
    publication.safe_local(output.parent, directory=True)
    implementation = operational_hashes()
    inputs = [(index_path, expected_index_sha256), (settings_path, expected_settings_sha256)]
    for path, digest in inputs:
        require(checkpoint.sha256(publication.safe_local(path)) == publication.validate_sha(digest), "Preparation input hash mismatch")
    index, settings = checkpoint.read_json(index_path), checkpoint.read_json(settings_path)
    validate_index(index)
    params = authenticate_settings(settings, index)
    identity = settings["expected_identity"]
    require(identity["genome_build"] == params["genome_build"] and
            str(identity["source_rare_contract"]["dnabr_rare_cohort_n_samples"]) == str(params["expected_source_samples"]) and
            identity["source_rare_contract"]["dnabr_rare_cohort_sha256"] == params["source_cohort_sha256"],
            "Expected identity differs from parameters")
    fs = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(output.parent)], text=True).strip()
    require(fs in {"ext4", "xfs"}, "Local block filesystem required, not a bucket mount")
    reserve = int(min_free_gib * 1024**3)
    require(shutil.disk_usage(output.parent).free >= reserve, "Free-space reserve unavailable")
    output.mkdir(mode=0o700)
    client = client or GCS()
    cloud_identity = client.identity()
    all_objects, complete, missing, campaigns = [], [], [], []
    optional_criteria = {}
    for number, campaign in enumerate(index["campaigns"]):
        folder = output / ("campaign%02d" % number)
        folder.mkdir(mode=0o700)
        uri = campaign["prefix"] + "/completed.json"
        try:
            remote = metadata_record(uri, client.metadata(uri))
        except MissingObject:
            missing.extend(campaign["chromosomes"])
            campaigns.append(dict(prefix=campaign["prefix"], status="WAITING_FOR_COMPLETED_MARKER"))
            continue
        for key, field in (("generation", "completion_generation"), ("sha256", "completion_sha256"),
                           ("md5_base64", "completion_md5_base64")):
            if field in campaign:
                if key == "sha256":
                    remote[key] = campaign[field]
                else:
                    require(remote[key] == campaign[field], "Pinned completion marker changed")
        marker = fetch(client, remote, folder / "completed.json", min_free_bytes=reserve, json_only=True)
        completion = checkpoint.read_json(marker["path"])
        require(completion.get("schema") == "r02_checkpoint_campaign_complete_v1" and
                completion.get("status") == campaign["expected_status"] and completion.get("biological_result") is False,
                "Invalid campaign completion status")
        require(sorted(completion.get("chromosomes", []), key=int) == sorted(campaign["chromosomes"], key=int) and
                sorted([r["chrom"] for r in completion.get("checkpoints", [])], key=int) == sorted(campaign["chromosomes"], key=int),
                "Campaign chromosome inventory mismatch")
        if "bundle_sha256" in campaign:
            require(completion.get("launch_bundle_sha256") == campaign["bundle_sha256"], "Launch bundle mismatch")
        all_objects.append(marker)
        for item in completion["checkpoints"]:
            chrom = item["chrom"]
            files = receipt_files(item, campaign["prefix"])
            proof = item["receipt_publication"]
            require(proof["uri"] == campaign["prefix"] + "/receipts/chr" + chrom.zfill(2) + ".json",
                    "Publication receipt URI mismatch")
            receipt = fetch(client, proof, folder / ("chr" + chrom + ".receipt.json"),
                            min_free_bytes=reserve, json_only=True)
            require(checkpoint.read_json(receipt["path"]) == item["publication"], "Remote receipt content mismatch")
            all_objects.append(receipt)
            target = output / ("chr" + chrom.zfill(2))
            target.mkdir(mode=0o700)
            manifest_record = next(f for f in files if f["name"] == "checkpoint.json")
            manifest = fetch(client, manifest_record, target / "checkpoint.json", min_free_bytes=reserve, json_only=True)
            record = checkpoint.read_json(manifest["path"])
            validate_checkpoint_header(record, chrom, index, settings)
            for key, value in record["registry"]["optional_criteria"].items():
                require(key not in optional_criteria or optional_criteria[key] == value, "Incompatible optional evidence criteria")
                optional_criteria[key] = value
            for file in files:
                recheck_remote(client, file)
                if file["name"] != "checkpoint.json":
                    require(all(record["outputs"][file["name"]][k] == file[k] for k in ("bytes", "sha256")),
                            "Stream declaration differs from publication")
                all_objects.append(file)
            complete.append(dict(chrom=chrom, path=str(target / "checkpoint.json"), sha256=manifest["sha256"],
                                 n_chains=record["registry"]["sources"][0]["n_chains"], files=files))
        campaigns.append(dict(prefix=campaign["prefix"], status="REMOTE_PROVENANCE_VERIFIED", completion=marker))
    complete.sort(key=lambda r: int(r["chrom"]))
    require(len({r["chrom"] for r in complete}) == len(complete), "Duplicate checkpoint chromosome")
    total_bytes = sum(f["bytes"] for r in complete for f in r["files"])
    result = dict(schema=SCHEMA, created_utc=pipeline.now(), mode=mode,
                  status="WAITING_FOR_CHECKPOINTS" if missing else "ALL22_REMOTE_PROVENANCE_VERIFIED_NOT_STAGED",
                  completed_chromosomes=[r["chrom"] for r in complete], missing_chromosomes=sorted(missing, key=int),
                  total_checkpoint_bytes=total_bytes, campaigns=campaigns, checkpoints=complete,
                  source_index_sha256=expected_index_sha256, settings_sha256=expected_settings_sha256,
                  scientific_code_sha256=index["scientific_code_sha256"], implementation_sha256=checkpoint.sha256(__file__),
                  operational_code_sha256=implementation, effective_cloud_identity=cloud_identity,
                  operational_parameters=dict(min_free_gib=min_free_gib, max_stage_gib=max_stage_gib,
                                              units="GiB=1024^3 bytes", mode=mode), consumer=settings["consumer"],
                  no_analysis_launched=True, scientific_rows_validated=False, no_biological_validation=True,
                  optional_evidence_criteria=optional_criteria, fs_type=fs, reserve_bytes=reserve)
    if mode == "stage" and not missing:
        require(total_bytes <= max_stage_gib * 1024**3, "Stage inventory exceeds explicit storage budget")
        require(shutil.disk_usage(output).free >= total_bytes + reserve, "Staging inventory would violate reserve")
        staged = []
        for item in complete:
            for file in item["files"]:
                staged.append(fetch(client, file, Path(item["path"]).parent / file["name"], min_free_bytes=reserve))
        result.update(status="ALL22_STAGED_AUTHENTICATED_CONTRACT_PREPARED_NOT_EXECUTED", staged_files=staged)
    # Revalidate all observed remote generations and auxiliary files before emitting any seal.
    for obj in all_objects:
        recheck_remote(client, obj)
    authenticate_settings(settings, index)
    require(checkpoint.code_hashes() == index["scientific_code_sha256"], "Scientific code changed during preparation")
    require(operational_hashes() == implementation, "Operational helper code changed during preparation")
    for path, digest in inputs:
        require(checkpoint.sha256(path) == digest, "Preparation input changed during read")
    if result["status"] == "ALL22_STAGED_AUTHENTICATED_CONTRACT_PREPARED_NOT_EXECUTED":
        contract = {k: params[k] for k in ("expected_samples", "expected_source_samples", "expected_configurations",
                    "genome_build", "source_cohort_sha256", "kinship_sha256", "kinship_thresholds", "edge_thresholds_bp",
                    "sample_ids_sha256")}
        contract.update(schema="r02_biological_checkpoint_aggregate_v1", study=params["study"],
                        checkpoints=[{k: r[k] for k in ("chrom", "path", "sha256")} for r in complete],
                        consumer=settings["consumer"],
                        preparation_scope="authenticated_bytes_and_headers; scientific_rows_checked_by_reducer")
        publication.write_receipt_new(output / "aggregate_contract.json", contract)
        result["aggregate_contract_sha256"] = checkpoint.sha256(output / "aggregate_contract.json")
    publication.write_receipt_new(output / "readiness.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index-path", "expected-index-sha256", "settings-path", "expected-settings-sha256", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("inventory", "stage"), default="inventory")
    parser.add_argument("--min-free-gib", type=float, required=True)
    parser.add_argument("--max-stage-gib", type=float, required=True)
    result = prepare(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("status", "completed_chromosomes", "missing_chromosomes", "total_checkpoint_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
