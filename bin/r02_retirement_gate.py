#!/usr/bin/env python3
"""Verify useful thesis products, then optionally retire their exact original IDs.

No upload, restart, stop, snapshot, OS imaging, or GCS deletion is performed.
Default is read-only. --apply requires an immutable reviewed manifest already
published to GCS. Numeric resource IDs are used for every cloud request; there
is deliberately no fallback to resource names when the provider rejects an ID.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication
import r02_resource_labels as labels


SCHEMA = 'r02_useful_products_retirement_v1'
IMPLEMENTATION_VERSION = '1.1.0'
LEGACY_GNOMIX_SCOPE = 'GNOMIX_MSP_FB_LEGACY_V1'
LEGACY_GNOMIX = publication.LEGACY_GNOMIX_SOURCE
KEEP_ROLES = {'scientific_input', 'scientific_product', 'scientific_partial',
              'variant_selection', 'mask', 'denominator', 'frozen_code',
              'parameters', 'provenance_log', 'resume_metadata', 'resume_work'}
EXCLUDE_ROLES = {'operating_system', 'credential', 'regenerable_cache',
                 'temporary_transfer', 'irrelevant_log'}
PREFIX = 'gs://projects-usp/dnaBr-lai/datalake/'
require = publication.require


def validate_record(record, *, allow_legacy_gnomix=False):
    require(isinstance(record, dict), 'Object record required')
    uri = record.get('uri')
    require(isinstance(uri, str) and (uri.startswith(PREFIX) or
            (allow_legacy_gnomix and LEGACY_GNOMIX.fullmatch(uri)))
            and not any(c in uri for c in '\n\r#?'), 'Object outside permitted read-only source/data prefix')
    publication.validate_sha(record.get('sha256'))
    require(type(record.get('bytes')) is int and record['bytes'] >= 0, 'Invalid byte count')
    require(isinstance(record.get('generation'), str) and
            re.fullmatch(r'[1-9][0-9]*', record['generation']), 'Pinned generation required')
    try:
        digest = base64.b64decode(record['md5_base64'], validate=True)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('Valid GCS MD5 required') from exc
    require(len(digest) == 16, 'Invalid MD5 length')


def validate_manifest(manifest):
    require(manifest.get('schema') == SCHEMA, 'Unsupported useful-products manifest')
    require(isinstance(manifest.get('authorization_reference'), str)
            and manifest['authorization_reference'].strip(), 'Recorded authorization required')
    require(manifest.get('scientific_status') == 'COMPLETED', 'Work not completed: retain disk')
    require(manifest.get('pending_work') == [], 'Pending work requires preserving its resources')
    scopes = manifest.get('equivalent_read_only_sources', [])
    require(isinstance(scopes, list) and len(scopes)==len(set(scopes)) and
            set(scopes) <= {LEGACY_GNOMIX_SCOPE}, 'Unknown equivalent read-only source scope')
    plan = manifest['resource_plan']
    labels.validate_plan(plan)
    require(len(plan['resources']) >= 2, 'VM and its original disk IDs required')
    inventory = manifest['inventory']
    require(inventory.get('outside_roots_reviewed') is True and
            inventory.get('unclassified_paths') == [], 'Incomplete inventory: retain disk')
    roots = inventory.get('roots')
    require(isinstance(roots, list) and roots and len(roots) == len(set(roots)) and
            all(isinstance(p, str) and p.startswith('/') and '..' not in Path(p).parts
                and p != '/' for p in roots), 'Explicit reviewed scientific roots required')
    entries = inventory.get('entries')
    require(isinstance(entries, list) and entries, 'Useful-product inventory is empty')
    seen, roles, records, archive_inventories = set(), set(), [], {}
    for entry in entries:
        name = entry.get('path')
        require(isinstance(name, str) and name.startswith('/') and
                '..' not in Path(name).parts and name not in seen, 'Duplicate/unsafe inventory path')
        require(any(Path(name).is_relative_to(Path(r)) for r in roots), 'Entry outside reviewed roots')
        seen.add(name)
        require(isinstance(entry.get('reason'), str) and entry['reason'].strip(), 'Retention reason required')
        role = entry.get('role')
        if entry.get('disposition') == 'PRESERVE':
            require(role in KEEP_ROLES, 'OS, credentials and regenerable caches are not useful products')
            require(not any(p in {'.ssh', 'credentials.db', 'access_tokens.db', 'gcloud'}
                            for p in Path(name).parts), 'Authentication material cannot be retained')
            publication.validate_sha(entry.get('sha256'))
            require(type(entry.get('bytes')) is int and entry['bytes'] >= 0, 'Invalid source byte count')
            validate_record(entry['object'], allow_legacy_gnomix=(
                role == 'scientific_input' and LEGACY_GNOMIX_SCOPE in scopes))
            if 'archive_member' in entry:
                proof = entry['archive_inventory']
                validate_record(proof['object'])
                path = str(publication.safe_local(proof['local_path']))
                if path not in archive_inventories:
                    digests = pipeline.publication_digests(path)
                    require(all(proof['object'][k] == digests[k] for k in digests),
                            'Archive inventory differs from authenticated copy')
                    value = publication.read_unique_json(Path(path))
                    members = value['entries']
                    require(len({m['path'] for m in members}) == len(members), 'Repeated archive member')
                    archive_inventories[path] = digests, value, {m['path']:m for m in members}
                digests, value, members = archive_inventories[path]
                require(all(proof['object'][k] == digests[k] for k in digests), 'Inventory proof changed')
                require(value['archive_sha256'] == entry['object']['sha256'], 'Wrong archive for member')
                member = members.get(entry['archive_member'], {})
                require(member.get('kind') == 'FILE' and member.get('sha256') == entry.get('sha256')
                        and member.get('bytes') == entry.get('bytes'), 'Useful archive member is absent or changed')
                records.append(proof['object'])
            else:
                require(entry.get('sha256') == entry['object']['sha256'] and
                        entry.get('bytes') == entry['object']['bytes'], 'Source/object identity differs')
            roles.add(role)
            records.append(entry['object'])
        else:
            require(entry.get('disposition') == 'EXCLUDE' and role in EXCLUDE_ROLES,
                    'Useful scientific material cannot be excluded as regenerable')
            require('object' not in entry, 'Excluded entries must not demand an unnecessary copy')
    require({'scientific_product', 'frozen_code', 'parameters', 'provenance_log'} <= roles,
            'Results, frozen code, parameters and scientific provenance must be preserved')
    resume = manifest['resume']
    require(resume.get('mode') in {'SCIENTIFIC_PRODUCTS', 'NEXTFLOW_TASKS'}, 'Explicit resume granularity required')
    require(isinstance(resume.get('limitations'), str) and resume['limitations'].strip(),
            'Resume limits must be recorded')
    if resume['mode'] == 'NEXTFLOW_TASKS':
        require({'resume_metadata', 'resume_work'} <= roles and
                resume.get('restore_verified') is True, 'Required Nextflow state is unverified')
    # Same immutable object may cover equivalent copies; differing descriptions
    # of that same object are a contradiction, not a reason to pick one.
    unique = {}
    for record in records:
        key = record['uri']
        signature = {k:record[k] for k in ('bytes', 'sha256', 'md5_base64', 'generation')}
        require(key not in unique or unique[key][0] == signature, 'Conflicting equivalent copies')
        unique[key] = signature, record
    return plan, [value[1] for value in unique.values()]


def cloud(plan, resource, action, runner=None):
    """Only exact numeric-ID GET and DELETE, never mutation by a reused name."""
    require(action in {'describe', 'delete'}, 'Only describe/delete supported')
    require(resource['kind'] in {'instances', 'disks'} and
            re.fullmatch(r'[1-9][0-9]*', resource['id']), 'Exact original numeric ID required')
    command = ['gcloud', 'compute', resource['kind'], action, resource['id'],
               '--project='+plan['project'], '--zone='+plan['zone'],
               '--account='+plan['account'], '--format=json', '--quiet']
    if action == 'delete' and resource['kind'] == 'instances':
        command.append('--keep-disks=all')
    result = (runner or subprocess.run)(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        # Authentication failures, invalid numeric identifiers and other errors
        # are never interpreted as resource absence or followed by name deletion.
        absent = re.search(r'(?i)(?:HTTPError\s+404|code[=: ]+404|was not found|notFound)', result.stderr or '')
        if action == 'describe' and absent:
            return None
        raise RuntimeError('Retirement cloud operation failed; retain remaining resources: '+(result.stderr or '')[-1000:])
    return json.loads(result.stdout) if result.stdout.strip() else None


def inspect(plan, resource, runner=None):
    value = cloud(plan, resource, 'describe', runner)
    if value is None:
        return None
    labels.resource_identity(value, plan['project'], plan['zone'], resource['kind'],
                             resource['name'], resource['id'])
    require(value.get('labels', {}).get('team') == 'frank', 'team=frank required on each exact resource')
    if resource['kind'] == 'instances':
        require(value.get('status') == 'TERMINATED', 'Active VM: retain all resources')
        expected = {r['self_link'] for r in plan['resources'][1:]}
        disks = value.get('disks', [])
        require(all(d.get('type') == 'PERSISTENT' and d.get('autoDelete') is False for d in disks),
                'Only retained persistent disks may be retired')
        require({d.get('source') for d in disks} == expected, 'Attached disk set changed')
    else:
        require(value.get('status') == 'READY', 'Disk not READY')
    return value


def verify_products(manifest_path, expected_sha256, manifest_record, verifier=None):
    digest = pipeline.publication_digests(manifest_path)
    require(digest['sha256'] == publication.validate_sha(expected_sha256), 'Reviewed manifest changed')
    manifest = publication.read_unique_json(manifest_path)
    plan, records = validate_manifest(manifest)
    validate_record(manifest_record)
    require(all(manifest_record[k] == digest[k] for k in digest), 'Published manifest differs from reviewed bytes')
    if verifier is None:
        require(os.environ.get('CLOUDSDK_CORE_ACCOUNT') == plan['account'],
                'Set CLOUDSDK_CORE_ACCOUNT to the recorded account for object verification')
        verifier = lambda record: publication.verify_remote(record, pinned=True)
    selected = [manifest_record, *records]
    if hasattr(verifier, 'verify_all'):
        verifier.verify_all(selected)
    else:
        for record in selected:
            verifier(record)
    require(pipeline.sha(manifest_path) == expected_sha256, 'Manifest changed during verification')
    return manifest, plan, records


def evaluate(manifest_path, expected_sha256, manifest_record, *, verifier=None, runner=None):
    manifest, plan, records = verify_products(manifest_path, expected_sha256, manifest_record, verifier)
    values = [inspect(plan, r, runner) for r in plan['resources']]
    vm = values[0]
    for disk in values[1:]:
        if disk is not None:
            require(disk.get('users', []) == ([vm['selfLink']] if vm else []),
                    'Disk has another user or attachment differs')
    require(vm is None or all(d is not None for d in values[1:]), 'Attached original disk missing')
    return dict(schema=SCHEMA, status='READY_FOR_RETIREMENT_NO_DELETION', utc=pipeline.now(),
                manifest_sha256=expected_sha256, verified_useful_objects=len(records),
                resume=manifest['resume'], resources=[dict(**r, present=v is not None)
                for r,v in zip(plan['resources'], values)], no_gcs_deletion=True)


def retire(manifest_path, expected_sha256, manifest_record, state_dir, *, verifier=None, runner=None):
    """Idempotent exact-ID retirement, with product/resource checks before every delete."""
    state_dir = publication.safe_local(state_dir, directory=True)
    with (state_dir/'retirement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = state_dir/'intent.json'
        if intent.exists():
            require(publication.read_unique_json(intent)['manifest_sha256'] == expected_sha256,
                    'State directory belongs to another manifest')
        final = state_dir/'complete.json'
        if final.exists():
            require(publication.read_unique_json(final)['manifest_sha256'] == expected_sha256,
                    'State directory belongs to another manifest')
        evaluate(manifest_path, expected_sha256, manifest_record, verifier=verifier, runner=runner)
        manifest = publication.read_unique_json(manifest_path)
        plan = manifest['resource_plan']
        if not intent.exists():
            publication.write_receipt_new(intent, dict(utc=pipeline.now(),
                manifest_sha256=expected_sha256, resource_plan=plan,
                no_gcs_deletion=True, numeric_ids_only=True))
        for resource in plan['resources']:
            # Rechecking every object is intentionally conservative; a previous
            # successful VM deletion cannot justify deleting an unprotected disk.
            evaluate(manifest_path, expected_sha256, manifest_record, verifier=verifier, runner=runner)
            current = inspect(plan, resource, runner)
            if current is None:
                continue
            if resource['kind'] == 'disks':
                require(not current.get('users', []), 'Attached disk cannot be deleted')
                require(inspect(plan, plan['resources'][0], runner) is None,
                        'Original VM must be absent before deleting a disk')
            sequence = len(list(state_dir.glob('event-*.json')))
            event = state_dir/f'event-{sequence:05d}.json'
            publication.write_receipt_new(event, dict(utc=pipeline.now(), action='DELETE_EXACT_ORIGINAL_ID',
                manifest_sha256=expected_sha256, resource=resource, before=current))
            operation = cloud(plan, resource, 'delete', runner)
            require(inspect(plan, resource, runner) is None, 'Deletion not verified; retain remaining resources')
            publication.write_receipt_new(state_dir/f'operation-{sequence:05d}.json',
                dict(utc=pipeline.now(), manifest_sha256=expected_sha256,
                     resource=resource, operation=operation, original_absent=True))
        checked = evaluate(manifest_path, expected_sha256, manifest_record, verifier=verifier, runner=runner)
        require(not any(r['present'] for r in checked['resources']), 'Original resource remains')
        result = {**checked, 'status':'ORIGINAL_RESOURCES_RETIRED_USEFUL_PRODUCTS_VERIFIED'}
        if not final.exists():
            publication.write_receipt_new(final, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--manifest-publication-record', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--state-dir', type=Path)
    args = parser.parse_args(argv)
    receipt = publication.safe_local(args.receipt, new=True)
    record = publication.read_unique_json(args.manifest_publication_record)
    call = (args.manifest, args.expected_manifest_sha256, record)
    if args.apply:
        require(args.state_dir is not None, '--apply requires persistent state-dir')
        result = retire(*call, args.state_dir)
    else:
        result = evaluate(*call)
    publication.write_receipt_new(receipt, result)
    print(json.dumps(dict(status=result['status'], receipt=str(receipt))))


if __name__ == '__main__':
    main()
