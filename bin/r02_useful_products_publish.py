#!/usr/bin/env python3
"""Publish only useful completed products and an independently verifiable inventory.

Does not decide scientific completion, stop resources or delete anything.
The caller supplies a verified completed scope; retirement remains gated by
r02_retirement_gate with original resource IDs and pinned GCS generations.
"""
from pathlib import Path
import hashlib
import os
import stat
import r02_autosome_pipeline as pipeline
import r02_publish_evidence as pub
import r02_retirement_gate as gate
require=pub.require;read=pub.read_unique_json;save=pub.write_receipt_new


def retained_role(relative, path):
    """Classify useful files in either a flat or multi-branch runtime."""
    parts=set(relative.parts)
    if parts.intersection({'results','outputs'}):return 'scientific_product'
    if 'frozen' in parts or path.suffix=='.py':return 'frozen_code'
    if parts.intersection({'contracts','admissions','review','evidence'}) or path.suffix in {'.json','.config'}:return 'parameters'
    if parts.intersection({'inputs','sources'}):return 'scientific_input'
    return 'provenance_log'

def publish_file(path,relative,prefix,receipts):
    path=Path(path); record=dict(name=Path(path).name,path=str(path),uri=prefix+'/'+relative,**pipeline.publication_digests(path))
    # Equal bytes at distinct result URIs are distinct publications.
    uri_key=hashlib.sha256(record['uri'].encode()).hexdigest()
    receipt=receipts/(uri_key+'-'+record['sha256']+'.json')
    legacy=receipts/(record['sha256']+'.json')
    if not receipt.exists() and legacy.exists() and read(legacy).get('uri')==record['uri']:receipt=legacy
    if receipt.exists():
        old=read(receipt); require(all(old[k]==record[k] for k in ('uri','bytes','sha256')),'Destination/content receipt changed')
        pub.verify_remote(old,pinned=True);return old
    pub.upload_one(record);pub.verify_remote(record,pinned=True);save(receipt,record);return record


def capture_external_logs(root, paths):
    """Copy the byte prefix present at capture time, not an ever-changing log."""
    require(isinstance(paths,list) and len(paths)==len(set(paths)), 'Unique declared external logs required')
    snapshots=[]
    for value in paths:
        source=Path(value)
        require(source.is_absolute() and source.resolve()==source and source.is_file() and
                not source.is_symlink() and not source.is_relative_to(root), 'Canonical external log required')
        destination=root/'external-log-snapshots'/source.name
        destination.parent.mkdir(exist_ok=True)
        require(not destination.exists(), 'Log snapshot already exists; inspect partial retention before retry')
        started=pipeline.now()
        with source.open('rb') as inp, destination.open('xb') as out:
            before=os.fstat(inp.fileno());require(stat.S_ISREG(before.st_mode), 'External log is not regular')
            remaining=before.st_size
            while remaining:
                block=inp.read(min(2**20,remaining));require(block, 'External log truncated during capture')
                out.write(block);remaining-=len(block)
            after=os.fstat(inp.fileno())
            require(after.st_size>=before.st_size, 'External log truncated during capture')
        snapshots.append(dict(source_path=str(source),snapshot_path=str(destination),
            captured_bytes=before.st_size,sha256=pipeline.sha(destination),capture_started_utc=started,
            capture_completed_utc=pipeline.now(),source_device=before.st_dev,source_inode=before.st_ino,
            scope='EXACT_PREFIX_PRESENT_AT_CAPTURE_START',
            limitation='Later appended startup/retirement lines are not part of this immutable snapshot.'))
    if snapshots:save(root/'external-log-snapshots'/'capture.json',dict(logs=snapshots))
    return snapshots


def retain_completed(root, launch, plan, product_records, *, completion, scope_description, resume_limitations,
                     additional_preserved_objects=(), completed_work_roots=(), durable_nextflow_roots=()):
    root=Path(root)
    require(scope_description and resume_limitations and completion.get("schema") and completion.get("status"), "Explicit completed scope required")
    # Optional phase directories are explicit; old callers retain their original layout.
    def phase_roots(values):
        require(isinstance(values,(tuple,list)) and len(values)==len(set(values)), 'Unique phase roots required')
        result=[]
        for value in values:
            require(isinstance(value,str) and value and not any(x in value for x in ['\\','\n','\r','\0']), 'Safe phase path required')
            rel=Path(value);path=root/rel
            require(not rel.is_absolute() and '..' not in rel.parts, 'Relative owned phase path required')
            require(str(rel)==value and path.is_relative_to(root) and path!=root
                and path.resolve()==path and path.is_dir(), 'Canonical existing phase root required')
            result.append(path)
        return result
    extra_work=phase_roots(completed_work_roots)
    extra_cache=phase_roots(durable_nextflow_roots)
    outside_logs=capture_external_logs(root,launch.get('external_provenance_logs',[]))
    receipts=root/'object-receipts';receipts.mkdir(exist_ok=True)
    # Existing equal GCS objects are sufficient for parental/source material.
    known={(r['sha256'],r['bytes']):r for r in product_records+read(root/'existing-sources.json')['files']}
    entries=[]
    excluded_top={'receipts','object-receipts'}
    for name in sorted(excluded_top):
        entries.append(dict(path=str(root/name),disposition='EXCLUDE',role='regenerable_cache',reason='Entire subtree: all scoped analyses are complete and verified; future analyses continue from scientific products. No pending task needs this completed cache or duplicate publication receipts.'))
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if rel.parts[0] in excluded_top:continue
        if path.is_dir() and not (path.is_symlink() and any(path.is_relative_to(p) for p in extra_work)):continue
        commands={'.command.sh','.command.run','.command.log','.command.out','.command.err','.command.trace','.exitcode'}
        in_work=rel.parts[0]=='work' or any(path.is_relative_to(p) for p in extra_work)
        in_cache=rel.parts[0]=='.nextflow' or any(path.is_relative_to(p) for p in extra_cache)
        if ((in_work and path.name not in commands) or
            (in_cache and (path.name.lower() in {'lock','.lock'} or path.name.endswith(('.lock','.tmp'))))):
            entries.append(dict(path=str(path),disposition='EXCLUDE',role='regenerable_cache',reason='Completed work product has its verified scientific publication; staged links/temporary files and cache locks need no duplicate copy. Commands, traces and durable Nextflow metadata are preserved separately.'));continue
        require(path.is_file() and not path.is_symlink(),'Unclassified special runtime path')
        if path.name in {'operator.lock'}:
            entries.append(dict(path=str(path),disposition='EXCLUDE',role='regenerable_cache',reason='Runtime lock has no scientific content.'));continue
        digest=pipeline.publication_digests(path);key=(digest['sha256'],digest['bytes'])
        if key in known:record=known[key];pub.verify_remote(record,pinned=True)
        else:record=publish_file(path,'provenance/runtime/'+str(rel),launch['output_prefix'],receipts);known[key]=record
        role='resume_metadata' if in_cache else retained_role(rel,path)
        entries.append(dict(path=str(path),disposition='PRESERVE',role=role,reason='Useful exact analysis result, input, code, parameters or provenance.',sha256=digest['sha256'],bytes=digest['bytes'],object=record))
    # Bootstrap files contain only authenticated transfer copies; no OS image.
    observed_paths={e['path'] for e in entries}
    for record in additional_preserved_objects:
        path=Path(record['path'])
        require(path.is_absolute() and path.is_relative_to(root) and '..' not in path.parts and
                str(path) not in observed_paths and not path.exists(), 'Archived object path is unsafe, duplicated or still local')
        pub.verify_remote(record,pinned=True)
        entries.append(dict(path=str(path),disposition='PRESERVE',role='scientific_product',
            reason='Completed scientific archive verified in GCS before its temporary local copy was removed; exact archive inventory is retained in the corresponding completion receipt.',
            sha256=record['sha256'],bytes=record['bytes'],object=record))
        observed_paths.add(str(path))
    bootstrap=Path(launch['bootstrap_root'])
    for p in sorted(bootstrap.rglob('*')):
        if p.is_file():entries.append(dict(path=str(p),disposition='EXCLUDE',role='temporary_transfer',reason='Verified duplicate bootstrap transfer; runtime and image identities are preserved.'))
    m=dict(schema=gate.SCHEMA,authorization_reference=launch['authorization_reference'],scientific_status='COMPLETED',pending_work=[],resource_plan=plan,equivalent_read_only_sources=launch.get('equivalent_read_only_sources',[]),inventory=dict(roots=[str(root),str(bootstrap)],entries=entries,outside_roots_reviewed=True,unclassified_paths=[],external_provenance_log_snapshots=outside_logs,outside_roots_review='New single-purpose VM; scientific commands write below the two inventoried roots. Declared external startup logs are preserved as immutable prefixes with capture bytes, SHA and UTC; later appended shutdown lines are excluded. OS, Docker image cache and user Nextflow framework contain no unique scientific products.'),resume=dict(mode='SCIENTIFIC_PRODUCTS',limitations=resume_limitations),completed_scope=scope_description)
    gate.validate_manifest(m);p=root/'retirement-manifest.json';save(p,m)
    record=publish_file(p,'retirement-manifest.json',launch['output_prefix'],receipts)
    done=root/'complete.json';save(done,dict(completion,resource_plan=plan,retirement_manifest=record,files=product_records,utc=pipeline.now()))
    final=publish_file(done,'complete.json',launch['output_prefix'],receipts)
    return dict(completion=final,retirement_manifest=record)
