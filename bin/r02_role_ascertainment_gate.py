#!/usr/bin/env python3
"""Verify declared role scope for selection and allele assignment.

This gate authenticates cohort declarations. It cannot by itself prove that a
producer actually used only those people; consumers must also authenticate the
producer receipt and code. A transductive declaration never satisfies a
consumer that requires FIT independence.
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path

ROLES = ("FIT", "SELECT", "EVALUATE")

def require(value, message):
    if not value: raise ValueError(message)

def canonical_ids(ids): return ("\n".join(ids) + "\n").encode()
def ids_sha256(ids): return hashlib.sha256(canonical_ids(ids)).hexdigest()
def file_sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_ids(path):
    values=[line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    require(values and len(values)==len(set(values)),"sample list is empty or has duplicates")
    return values

def read_roles(path):
    with Path(path).open(newline="") as stream: rows=list(csv.DictReader(stream,delimiter="\t"))
    require(rows and {"sample_id","role"}<=set(rows[0]),"roles TSV lacks sample_id or role")
    require(len({r["sample_id"] for r in rows})==len(rows),"roles TSV has duplicate people")
    require(all(r["role"] in ROLES for r in rows),"roles TSV contains an unknown role")
    return {role:[r["sample_id"] for r in rows if r["role"]==role] for role in ROLES}

def check_source(label,source,roles):
    require(set(source)=={"operation","sample_ids_path","sample_ids_sha256","n_samples"},f"{label} provenance fields differ")
    ids=read_ids(source["sample_ids_path"])
    require(len(ids)==source["n_samples"] and ids_sha256(ids)==source["sample_ids_sha256"],f"{label} sample identity/hash differs")
    known={person:role for role,people in roles.items() for person in people}
    counts={role:sum(known.get(person)==role for person in ids) for role in ROLES}
    return ids,counts,sum(person not in known for person in ids)

def checked_document(value,label,extra_fields):
    require(isinstance(value,dict) and set(value)=={"path","sha256",*extra_fields},f"{label} fields differ")
    path=Path(value["path"])
    require(path.is_file() and file_sha256(path)==value["sha256"],f"{label} file/hash differs")
    return json.loads(path.read_text())

def declared_fit_scope(provenance,roles):
    value=provenance.get("fit_scope")
    require(isinstance(value,dict) and set(value)=={"basis","sample_ids_path","sample_ids_sha256","n_samples","justification"},"FIT scope fields differ")
    ids=read_ids(value["sample_ids_path"])
    require(len(ids)==value["n_samples"] and ids_sha256(ids)==value["sample_ids_sha256"],"FIT scope identity/hash differs")
    require(set(ids)<=set(roles["FIT"]),"FIT scope contains a held-out or unknown person")
    if value["basis"]=="FULL_FIT_ROLE":
        require(set(ids)==set(roles["FIT"]) and value["justification"] is None,"FULL_FIT_ROLE must contain all and only FIT people")
    else:
        require(value["basis"]=="PREREGISTERED_FIT_SUBCOHORT","unknown FIT scope basis")
        doc=checked_document(value["justification"],"FIT subcohort justification",{"recorded_at_utc","authorized_scope"})
        require(doc.get("status")=="AUTHORIZED_BEFORE_ANALYSIS" and doc.get("sample_ids_sha256")==value["sample_ids_sha256"],"FIT subcohort justification does not bind the declared people")
    return ids

def evaluate(provenance,role_path):
    require(provenance.get("schema")=="r02_role_ascertainment_provenance_v1","provenance schema differs")
    require(type(provenance.get("consumer_requires_fit_independence")) is bool,"consumer must state whether FIT independence is required")
    roles=read_roles(role_path); fit=roles["FIT"]
    require(fit and roles["SELECT"] and roles["EVALUATE"],"all three roles must be explicit")
    fit_scope=declared_fit_scope(provenance,roles)
    selection,selection_counts,selection_unknown=check_source("selection",provenance["selection"],roles)
    orientation,orientation_counts,orientation_unknown=check_source("orientation",provenance["orientation"],roles)
    facts={"roles_tsv_sha256":file_sha256(role_path),"fit_people":len(fit),"declared_fit_scope_people":len(fit_scope),
           "declared_fit_scope_sha256":ids_sha256(fit_scope),"select_people":len(roles["SELECT"]),"evaluate_people":len(roles["EVALUATE"]),
           "selection_role_counts":selection_counts,"selection_unassigned_people":selection_unknown,
           "orientation_role_counts":orientation_counts,"orientation_unassigned_people":orientation_unknown}
    scope=provenance.get("analysis_scope")
    if scope=="FIT_INDUCTIVE":
        require(set(selection)==set(fit_scope),"selection cohort is not exactly the declared FIT scope")
        require(set(orientation)==set(fit_scope),"allele assignment cohort is not exactly the declared FIT scope")
        return {"status":"VERIFIED_DECLARED_FIT_ROLE_SCOPE_NOT_EXECUTION_PROOF","fit_independence_proven":False,
                "admission_limit":"Authenticate the producer receipt and code to establish actual use.",**facts}
    require(scope=="EXPLICIT_TRANSDUCTIVE","analysis_scope must be FIT_INDUCTIVE or EXPLICIT_TRANSDUCTIVE")
    protocol=provenance.get("transductive_protocol",{})
    require(set(protocol)=={"status","inference_scope","document","authorization"} and protocol.get("status")=="PREREGISTERED_BEFORE_SELECT_EVALUATE" and protocol.get("inference_scope")=="TRANSDUCTIVE_CONDITIONAL_ON_FIXED_COHORT","explicit transductive protocol is incomplete")
    document=checked_document(protocol["document"],"transductive protocol",{"recorded_at_utc","inference_scope"})
    authorization=checked_document(protocol["authorization"],"transductive authorization",{"recorded_at_utc","authorized_scope"})
    require(document.get("status")=="PREREGISTERED_BEFORE_SELECT_EVALUATE" and authorization.get("status")=="AUTHORIZED_BEFORE_ANALYSIS","transductive files do not prove prior design and authorization")
    require(provenance["consumer_requires_fit_independence"] is False,"consumer requires FIT independence; transductive input is blocked")
    return {"status":"VERIFIED_EXPLICIT_TRANSDUCTIVE_DECLARATION_NOT_FIT_INDEPENDENT","fit_independence_proven":False,**facts}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--provenance",type=Path,required=True);parser.add_argument("--roles",type=Path,required=True);parser.add_argument("--output",type=Path);args=parser.parse_args()
    try: result=evaluate(json.loads(args.provenance.read_text()),args.roles)
    except Exception as error:
        result={"status":"REJECTED_ROLE_ASCERTAINMENT","fit_independence_proven":False,"error":str(error)}
        if args.output: args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps(result,sort_keys=True));raise SystemExit(2)
    if args.output: args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
