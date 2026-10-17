#!/usr/bin/env python3
"""Exclusive fixed-case claims and restoration of authenticated LAI products.

No resource creation, reassignment, deletion, scientific computation or cache
promotion. An expired owner is never replaced by this API. The coordinator
must separately authenticate the original campaign and resource identities.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time

import r02_publish_evidence as pub
import r02_unique_products_archive as archive_tools

require = pub.require
pipeline = pub.pipeline


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_assignment(assignment):
    require(assignment.get('schema') == 'r02_lai_case_assignment_v1', 'Wrong assignment schema')
    pub.validate_sha(assignment['campaign_sha256'])
    pub.destination_prefix(assignment['claims_prefix'])
    workers = assignment['workers']
    cases = assignment['cases']
    require(set(workers) == set('abcdef') and len(cases) == 12, 'Exactly six workers and twelve cases required')
    require(len({x['id'] for x in workers.values()}) == 6, 'Repeated original VM identity')
    expected = {('FOLD_'+fold, replicate, chromosome) for fold in 'AB'
                for replicate in (1, 2, 3) for chromosome in ('1', '22')}
    observed = set()
    for key, owner in workers.items():
        require(set(owner) == {'project', 'zone', 'name', 'id', 'deadline_epoch'}, 'Incomplete/excess owner fields')
        require(owner['project'] == 'uspbr-242713' and owner['zone'] == 'us-central1-a'
                and re.fullmatch(r'[1-9][0-9]*', owner['id'])
                and re.fullmatch(r'dnabr-r02-lai-[a-z0-9-]+', owner['name'])
                and type(owner['deadline_epoch']) is int and owner['deadline_epoch'] > 0,
                'Invalid original owner or absolute deadline')
        selected = [x for x in cases if x['worker_id'] == key]
        require(len(selected) == 2, 'Each worker must own two cases')
        group = ('FOLD_A' if key < 'd' else 'FOLD_B', (ord(key)-ord('a')) % 3+1)
        require({(x['donor_fold'], x['replicate']) for x in selected} == {group}, 'Worker fold/replicate allocation differs')
    for case in cases:
        key = (case['donor_fold'], case['replicate'], case['chromosome'])
        require(key in expected and key not in observed, 'Repeated or unauthorized case')
        observed.add(key)
        require(case['case_id'] == 'R02L2pilot.PULSE_T10_ANCHOR.chr'+key[2]+'.'+key[0]+'.r'+str(key[1]), 'Case name differs from source coordinates')
        pub.validate_sha(case['source_sha256'])
    require(observed == expected, 'Incomplete case coverage')
    return assignment


def claim_document(assignment, case_id, worker_original_identity, deadline_epoch):
    validate_assignment(assignment)
    selected = [x for x in assignment['cases'] if x['case_id'] == case_id]
    require(len(selected) == 1, 'Case absent/duplicated')
    case = selected[0]
    owner = assignment['workers'][case['worker_id']]
    require(worker_original_identity == {k: owner[k] for k in ('project', 'zone', 'name', 'id')},
            'Worker original identity does not own this case')
    require(type(deadline_epoch) is int and deadline_epoch == owner['deadline_epoch'], 'Deadline cannot be extended')
    return dict(schema='r02_lai_case_claim_v1', assignment_sha256=canonical_sha(assignment),
                campaign_sha256=assignment['campaign_sha256'], source_sha256=case['source_sha256'],
                case_id=case_id, worker_id=case['worker_id'], owner=worker_original_identity,
                deadline_epoch=deadline_epoch, transfer_allowed=False)


def claim_case(assignment, case_id, worker_original_identity, deadline_epoch, *, local_dir,
               provider=pub, now=time.time):
    claim = claim_document(assignment, case_id, worker_original_identity, deadline_epoch)
    require(now() < deadline_epoch, 'Owner deadline has expired; no takeover allowed')
    folder = pub.safe_local(local_dir, directory=True)
    path = folder / (case_id+'.claim.json')
    if path.exists():
        require(pub.read_unique_json(path) == claim, 'Divergent previous local claim')
    else:
        pub.write_receipt_new(path, claim)
    record = dict(name=path.name, path=str(path), uri=assignment['claims_prefix'].rstrip('/')+'/'+case_id+'.json',
                  **pipeline.publication_digests(path))
    # upload_one is generation-match=0. HTTP 412 only reuses identical bytes;
    # permission/auth/network errors fail rather than implying absent ownership.
    provider.upload_one(record)
    provider.verify_remote(record, pinned=True)
    return dict(claim=claim, object=record)


def validate_claim_before_block(handle, assignment, case_id, worker_original_identity, deadline_epoch,
                                *, provider=pub, now=time.time):
    expected = claim_document(assignment, case_id, worker_original_identity, deadline_epoch)
    require(handle['claim'] == expected and now() < deadline_epoch, 'Claim owner/assignment/deadline differs')
    record = handle['object']
    require(record['uri'] == assignment['claims_prefix'].rstrip('/')+'/'+case_id+'.json', 'Claim URI differs')
    require(record['sha256'] == hashlib.sha256((json.dumps(expected, indent=2, allow_nan=False)+'\n').encode()).hexdigest(),
            'Claim object bytes differ from expected owner')
    provider.verify_remote(record, pinned=True)
    return expected


def verify_completed_block(receipt, *, expected_case, expected_generation, expected_source_sha,
                           expected_campaign_sha, expected_assignment_sha, expected_claim,
                           allowed_prefixes, provider=pub):
    require(receipt.get('schema') == 'r02_lai_owned_block_v1'
            and receipt['case_id'] == expected_case and receipt['generation'] == expected_generation
            and expected_generation in (5, 10, 16)
            and receipt['source_sha256'] == expected_source_sha
            and receipt['campaign_sha256'] == expected_campaign_sha
            and receipt['assignment_sha256'] == expected_assignment_sha
            and receipt['claim'] == expected_claim, 'Completed block provenance differs')
    for pin in (expected_source_sha, expected_campaign_sha, expected_assignment_sha):pub.validate_sha(pin)
    legacy = receipt['legacy_block']
    require(legacy['status'] == 'FOUR_ARMS_SCORED_SAME_FINITE_DOMAIN'
            and legacy['generation'] == expected_generation
            and legacy['population_generalization'] is False
            and legacy['rare_identity_causal_effect_identified'] is False,
            'Block is not a completed fixed-domain four-arm comparison')
    require(len(legacy['files']) == 4 and len({x['uri'] for x in legacy['files']}) == 4, 'Model/evaluation archive inventory incomplete')
    prefixes = [pub.destination_prefix(x)+'/' for x in allowed_prefixes]
    for record in legacy['files']:
        require(any(record['uri'].startswith(x) for x in prefixes), 'Block object outside admitted prefixes')
        require(not any(x in record['uri'] for x in ('#', '?', '\n', '\r', '/../')), 'Unsafe block URI')
        pub.validate_sha(record['sha256'])
        provider.verify_remote(record, pinned=True)
    require(sum(x['uri'].endswith('.tar') for x in legacy['files']) == 2
            and sum(x['uri'].endswith('.inventory.json') for x in legacy['files']) == 2,
            'Exactly two archives and inventories required')
    return legacy


def download_pinned(record, destination, *, deadline_epoch, provider=pub, derive_sha256=False):
    """Create one local copy, preserving divergent/partial files on any failure.

    The optional derivation mode authenticates a legacy source by its fixed GCS
    generation, byte count and MD5 when no independently recorded SHA256 exists.
    The derived SHA is a new local digest, never a historical independent pin.
    Callers must record publication_digests(destination) after this returns.
    Existing callers still require all three local digest fields by default.
    """
    destination = Path(destination)
    require(type(derive_sha256) is bool, 'Explicit Boolean SHA derivation policy required')
    require('sha256' in record or derive_sha256, 'Independent SHA256 required unless derivation explicitly admitted')
    if 'sha256' in record:pub.validate_sha(record['sha256'])
    expected={k:record[k] for k in ('bytes','md5_base64')}
    if 'sha256' in record:expected['sha256']=record['sha256']
    def equivalent(path):
        actual=pipeline.publication_digests(path)
        return all(actual[k]==v for k,v in expected.items())
    provider.verify_remote(record, pinned=True)
    if destination.exists():
        require(equivalent(destination), 'Existing download differs')
        return destination
    require(time.time() < deadline_epoch, 'Restoration deadline expired')
    temporary = Path(tempfile.mkdtemp(prefix='r02-lai-download-', dir=destination.parent))
    try:
        partial = temporary/'payload'
        proc = subprocess.Popen(['gcloud', 'storage', 'cp', record['uri']+'#'+record['generation'], str(partial)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        try:
            _, error = proc.communicate(timeout=max(.01, deadline_epoch-time.time()))
            require(proc.returncode == 0, 'Pinned block download failed: '+error.decode(errors='replace')[-1000:])
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid, signal.SIGKILL);proc.wait()
        require(equivalent(partial), 'Downloaded block bytes differ')
        provider.verify_remote(record, pinned=True)
        os.link(partial, destination)
        partial.unlink();temporary.rmdir()
    except BaseException:
        # Preserve any partial transfer for diagnosis. It is never a completion.
        raise
    return destination


def restore_stage(archive, inventory, destination, *, deadline_epoch, disk_reserve_bytes):
    """Rebuild exact paths/aliases in one stage directory, never overwrite.

    Different historical stages keep separate roots: their evolving logs and
    request files are not silently overlaid. Exclusions remain explicit; this
    operation does not claim resumable Nextflow cache or restored material VCF.
    """
    destination = pub.safe_local(destination, directory=True)
    require(type(disk_reserve_bytes) is int and disk_reserve_bytes >= 0, 'Invalid disk reserve')
    entries = inventory['entries']
    require(len({x['path'] for x in entries}) == len(entries), 'Repeated inventory path')
    for row in entries:
        archive_tools.relative_member(row['path'])
        require(row['disposition'] in ('PRESERVE', 'EXCLUDE'), 'Unresolved external reference')
        if row['disposition'] == 'EXCLUDE':require(bool(row.get('reason')), 'Unjustified exclusion')
    archive_tools.verify(archive, inventory)
    selected = [x for x in entries if x['disposition'] == 'PRESERVE']
    missing_bytes = 0
    for row in selected:
        path = destination/row['path']
        require(path.resolve() == path and not any(x.is_symlink() for x in (path, *path.parents)), 'Symlink in restore path')
        if path.exists():
            require(path.is_file() and pipeline.publication_digests(path) == {k: row[k] for k in ('bytes', 'sha256', 'md5_base64')}, 'Existing restored file differs')
        else:missing_bytes += row['bytes']
    require(shutil.disk_usage(destination).free >= missing_bytes+disk_reserve_bytes, 'Insufficient disk for exact aliases and reserve')
    restored = 0
    with tarfile.open(archive, 'r:*') as tar:
        for row in selected:
            require(time.time() < deadline_epoch and shutil.disk_usage(destination).free >= disk_reserve_bytes, 'Restoration deadline/disk reserve reached')
            path = destination/row['path']
            if path.exists():continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.restore-', delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    with tar.extractfile(row['archive_member']) as src:
                        while True:
                            require(time.time() < deadline_epoch, 'Restoration deadline expired')
                            chunk = src.read(2**20)
                            if not chunk:break
                            temp.write(chunk)
                    temp.flush();os.fsync(temp.fileno())
                except BaseException:
                    # Keep the interrupted bytes; never mistake them for a complete product.
                    raise
            require(pipeline.publication_digests(temporary) == {k: row[k] for k in ('bytes', 'sha256', 'md5_base64')}, 'Restored payload differs')
            os.link(temporary, path);temporary.unlink();restored += 1
    return dict(status='EXACT_STAGE_PRODUCT_PATHS_RESTORED', newly_restored=restored,
                selected_paths=len(selected), nextflow_cache_resume_certified=False,
                exclusions=[x for x in entries if x['disposition'] == 'EXCLUDE'])


def restore_completed_block(receipt, destination, *, cache_dir, deadline_epoch, disk_reserve_bytes,
                            provider=pub, downloader=download_pinned, **expectations):
    legacy = verify_completed_block(receipt, provider=provider, **expectations)
    destination = pub.safe_local(destination, directory=True)
    cache_dir = pub.safe_local(cache_dir, directory=True)
    require(shutil.disk_usage(cache_dir).free >= sum(x['bytes'] for x in legacy['files'])+disk_reserve_bytes,
            'Insufficient transfer space plus reserve')
    local = {}
    for record in legacy['files']:
        name = hashlib.sha256(record['uri'].encode()).hexdigest()
        local[record['uri']] = downloader(record, cache_dir/name, deadline_epoch=deadline_epoch, provider=provider)
    results = {}
    for record in legacy['files']:
        if not record['uri'].endswith('.inventory.json'):continue
        inventory = pub.read_unique_json(local[record['uri']])
        require(inventory.get('schema') == 'r02_lai_useful_stage_archive_v1'
                and inventory['case'] == expectations['expected_case']
                and inventory['initial_generation'] == expectations['expected_generation']
                and inventory['source_test_opened'] is False
                and inventory['stage'] in ('models', 'evaluation'), 'Stage archive scope differs')
        stage = inventory['stage']
        require(stage not in results, 'Duplicate block stage')
        archive_uri = record['uri'].removesuffix('.inventory.json')+'.tar'
        require(archive_uri in local, 'Missing corresponding stage archive')
        folder = destination/stage
        folder.mkdir(exist_ok=True)
        results[stage] = restore_stage(local[archive_uri], inventory, folder,
                                      deadline_epoch=deadline_epoch, disk_reserve_bytes=disk_reserve_bytes)
    require(set(results) == {'models', 'evaluation'}, 'Both block stages required')
    for record in legacy['files']:provider.verify_remote(record, pinned=True)
    return dict(status='COMPLETED_BLOCK_PRODUCTS_RESTORED_WITH_STAGE_SEPARATION', stages=results,
                case_id=expectations['expected_case'], generation=expectations['expected_generation'],
                nextflow_cache_resume_certified=False)
