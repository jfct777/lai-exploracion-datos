#!/usr/bin/env python3
"""Authenticate the published K02/K07 prepared catalogues for all 88 views."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


SCOPES = ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME")
PREFIX = ("gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/"
          "biologico/R02_20260930/r02-structure-assembly-20261005g")


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): h.update(block)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): h.update(block)
    return base64.b64encode(h.digest()).decode()


def load(path): return json.loads(Path(path).read_text())


def build(gcs_list_path, metadata_root):
    listing = load(gcs_list_path)
    objects = {}
    for item in listing:
        if item.get("type") != "cloud_object": continue
        metadata = item["metadata"]; base = item["url"].split("#", 1)[0]
        require(base not in objects, "Duplicate GCS object")
        objects[base] = dict(uri=base, generation=str(metadata["generation"]),
            generation_uri=base + "#" + str(metadata["generation"]),
            bytes=int(metadata["size"]), md5_base64=metadata["md5Hash"],
            crc32c=metadata["crc32c"])
    root = Path(metadata_root).resolve(); views = []; staging = {}
    for scope in SCOPES:
        local = root / scope; remote = f"{PREFIX}/{scope}/published/prepared"
        contract_path = local / "contract.json"; index_path = local / "block_index.json"
        manifest_path = local / "manifest.json"
        contract, index, manifest = map(load, (contract_path, index_path, manifest_path))
        require(contract["exclusion_scope"] == scope and manifest["exclusion_scope"] == scope
                and manifest["stage"] == "COMPLETE_BLOCK_INDEX_NO_TRAINING"
                and manifest["outputs_sha256"]["contract.json"] == sha(contract_path)
                and manifest["outputs_sha256"]["block_index.json"] == sha(index_path)
                and len(index["blocks"]) == 44, "Prepared catalogue metadata differs")
        root_objects = {}
        for name in set(contract["sha256"]) | {"contract.json", "manifest.json"}:
            uri = remote + "/" + name; record = objects.get(uri)
            require(record and (name not in contract["sha256"] or
                    contract["sha256"][name] == sha(local / name)), "Root source differs: " + uri)
            require(md5(local / name) == record["md5_base64"], "Downloaded root MD5 differs")
            record = dict(record, sha256=sha(local / name)); root_objects[name] = record
            staging[uri] = record
        for row in index["blocks"]:
            block = row["block_id"]; shard = local / row["path"]
            shard_contract_path = shard / "contract.json"; shard_manifest_path = shard / "manifest.json"
            shard_contract = load(shard_contract_path); shard_manifest = load(shard_manifest_path)
            require(sha(shard_contract_path) == row["contract_sha256"]
                    and sha(shard_manifest_path) == row["manifest_sha256"]
                    and shard_manifest["block_id"] == block
                    and shard_manifest["exclusion_scope"] == scope
                    and shard_manifest["outputs_sha256"]["contract.json"] == row["contract_sha256"]
                    and shard_manifest["training_allowed"] is False,
                    "Shard metadata differs: " + block + "__" + scope)
            shard_objects = {}
            for name, digest in shard_manifest["outputs_sha256"].items():
                uri = remote + "/" + row["path"] + "/" + name; record = objects.get(uri)
                require(record is not None, "Published shard object missing: " + uri)
                if (shard / name).is_file():
                    require(sha(shard / name) == digest and md5(shard / name) == record["md5_base64"],
                            "Downloaded shard metadata differs: " + uri)
                record = dict(record, sha256=digest); shard_objects[name] = record; staging[uri] = record
            manifest_uri = remote + "/" + row["path"] + "/manifest.json"
            manifest_record = objects.get(manifest_uri)
            require(manifest_record and md5(shard_manifest_path) == manifest_record["md5_base64"],
                    "Published shard manifest differs")
            manifest_record = dict(manifest_record, sha256=row["manifest_sha256"])
            staging[manifest_uri] = manifest_record
            views.append(dict(view_id=block + "__" + scope, block_id=block,
                exclusion_scope=scope, prepared=dict(directory_uri=remote,
                    contract_sha256=sha(contract_path), root_objects=root_objects,
                    shard_path=row["path"], shard_contract_sha256=row["contract_sha256"],
                    shard_manifest_sha256=row["manifest_sha256"],
                    shard_manifest=manifest_record, shard_objects=shard_objects)))
    require(len(views) == 88 and len({x["view_id"] for x in views}) == 88,
            "Exactly 88 prepared views required")
    return dict(schema="r02_common_territory_assembly_sources_v1",
        status="GCS_SOURCES_AUTHENTICATED_NOT_STAGED_NOT_ADMITTED", view_count=88,
        views=sorted(views, key=lambda x:x["view_id"]),
        staging=dict(unique_objects=len(staging), total_bytes=sum(x["bytes"] for x in staging.values()),
                     objects=sorted(staging.values(), key=lambda x:x["uri"])),
        model_fits_run=False, evaluate_targets_computed=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-list", required=True);parser.add_argument("--gcs-list-sha256", required=True)
    parser.add_argument("--metadata-root", required=True);parser.add_argument("--output", required=True)
    args=parser.parse_args();require(sha(args.gcs_list)==args.gcs_list_sha256,"GCS listing pin differs")
    result=build(args.gcs_list,args.metadata_root)
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"views":88,"objects":result["staging"]["unique_objects"],
                      "bytes":result["staging"]["total_bytes"],"sha256":sha(args.output)}))
