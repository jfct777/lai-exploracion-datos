#!/usr/bin/env python3
"""Run all admitted retained-FLARE recoveries sequentially and close the campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import r02_lai_flare_recovery_operator as operator


def build_completion(manifest, evidence, verify_remote=True):
    rows=[]
    for source in manifest["tasks"]:
        source_id=source["source_id"]
        completion=evidence/(source_id+".completion.json")
        publication=evidence/(source_id+".publication.json")
        operator.require(completion.is_file() and publication.is_file(), "source publication absent: "+source_id)
        local=operator.read_json(publication); proof=local.get("completion",{})
        operator.require(local.get("status")=="REMOTE_COMPLETION_VERIFIED" and
                         local.get("active_scratch_removed") is True, "source publication incomplete")
        if verify_remote:
            operator.verify_published_file(completion,proof["uri"],proof["generation"])
        rows.append({"source_id":source_id,"completion":{k:v for k,v in proof.items() if k!="operation"}})
    return {"schema":"r02_lai_flare_recovery_campaign_completion_v1",
        "status":"ALL_RETAINED_FLARE_SENSITIVITY_PROJECTIONS_PUBLISHED_AND_VERIFIED",
        "archive_count":36,"predictor_source_count":108,"projection_count":216,
        "count_correction":"72 counted one unspecified predictor arm; 216 preserves all three arms without selection",
        "staging_concurrency_archives":1,"minimum_free_bytes_reserved":operator.MINIMUM_FREE_AFTER_DOWNLOAD,
        "source_completions":sorted(rows,key=lambda row:row["source_id"]),
        "training_started":False,"selection_performed":False,"evaluate_opened":False,
        "source_gcs_deleted_or_overwritten":False,
        "scientific_limit":"These are derived truth-blind inputs, not FLARE evaluation or model selection."}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovery-manifest",type=Path,required=True)
    p.add_argument("--scratch-root",type=Path,required=True)
    p.add_argument("--output-prefix",required=True)
    p.add_argument("--container",required=True)
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args(argv); repo=a.repo_root.resolve(); scratch=a.scratch_root.resolve()
    manifest=operator.read_json(a.recovery_manifest.resolve())
    operator.require(manifest.get("archive_count")==36 and manifest.get("predictor_source_count")==108 and
                     manifest.get("projection_count")==216,"exact 36/108/216 recovery manifest required")
    active=scratch/"active"; active.mkdir(parents=True,exist_ok=True,mode=0o700)
    limit_path=scratch/"campaign-limit.json"
    if limit_path.exists():
        limit=operator.read_json(limit_path)
        operator.require(limit.get("schema")=="r02_lai_flare_recovery_campaign_limit_v1" and
                         limit.get("duration_seconds")==172800,"campaign limit differs")
    else:
        limit={"schema":"r02_lai_flare_recovery_campaign_limit_v1","duration_seconds":172800,
               "started_epoch":time.time(),"deadline_epoch":time.time()+172800}
        with limit_path.open("x") as stream:json.dump(limit,stream,indent=2,sort_keys=True);stream.write("\n")
    operator.require(60 < limit["deadline_epoch"]-time.time() <= 172800,
                     "48-hour campaign deadline exhausted or invalid")
    active_dirs=[path for path in active.iterdir() if path.is_dir()]
    operator.require(len(active_dirs)<=1,"more than one archive is active; inspect before continuing")
    active_source=active_dirs[0].name if active_dirs else None
    admitted_ids={row["source_id"] for row in manifest["tasks"]}
    operator.require(active_source is None or active_source in admitted_ids,
                     "active directory is not an admitted recovery source")
    for source in manifest["tasks"]:
        source_id=source["source_id"]
        if active_source is not None and source_id!=active_source:
            operator.require((scratch/"receipts"/(source_id+".publication.json")).is_file(),
                             "an earlier unclosed archive prevents starting another source")
            continue
        command=[sys.executable,str(repo/"bin/r02_lai_flare_recovery_operator.py"),
                 "--source-id",source_id,"--recovery-manifest",str(a.recovery_manifest.resolve()),
                 "--scratch-root",str(scratch),"--output-prefix",a.output_prefix,
                 "--container",a.container,"--repo-root",str(repo),
                 "--campaign-deadline-epoch",str(limit["deadline_epoch"])]
        if (active/source_id).exists(): command.append("--resume")
        subprocess.run(command,check=True)
        operator.require(not any(active.iterdir()),"successful source left active archive bytes")
        active_source=None
    evidence=scratch/"receipts"; completion=build_completion(manifest,evidence)
    code_proof=operator.publish_file(Path(__file__),a.output_prefix.rstrip("/")+"/provenance/"+Path(__file__).name)
    completion["campaign_operator"]={k:v for k,v in code_proof.items() if k!="operation"}
    completion_path=evidence/"campaign-completion.json"
    if completion_path.exists(): operator.require(operator.read_json(completion_path)==completion,"campaign completion differs")
    else:
        with completion_path.open("x") as stream:json.dump(completion,stream,indent=2,sort_keys=True);stream.write("\n")
    proof=operator.publish_file(completion_path,a.output_prefix.rstrip("/")+"/campaign-completion.json")
    publication_path=evidence/"campaign-publication.json"
    value={"schema":"r02_lai_flare_recovery_campaign_publication_v1",
           "status":"REMOTE_CAMPAIGN_COMPLETION_VERIFIED",
           "completion":{k:v for k,v in proof.items() if k!="operation"}}
    if publication_path.exists():operator.require(operator.read_json(publication_path)==value,"campaign proof differs")
    else:
        with publication_path.open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write("\n")
    print(json.dumps(value,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
