#!/usr/bin/env python3
"""Execute a certified chromosome job or reuse its exact completed product."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import shutil
import r02_k03_event_alt_chunks as raw
import r02_k03_derive_exclusion as exclusions
import r02_k03_event_view_plan as planning


def reuse(directory, output, identity, people, manifest_sha256):
    directory, output = Path(directory).resolve(), Path(output)
    raw.require(raw.sha256(directory/'chromosome.json') == manifest_sha256,
                'Completed manifest changed since task inventory')
    record = exclusions.authenticated_record(directory, identity, people)
    raw.require(not output.exists(), 'Reuse destination already exists')
    (output/'chunks').mkdir(parents=True)
    for item in record['chunks']:
        source, dest = directory/item['path'], output/item['path']
        try: os.link(source, dest)
        except OSError: shutil.copyfile(source, dest)
        raw.require(raw.sha256(dest)==item['sha256'], 'Reuse copy changed')
    shutil.copyfile(directory/'chromosome.json',output/'chromosome.json')
    raw.write_json(output/'REUSE.json',dict(source=str(directory),manifest_sha256=manifest_sha256,
                   original_decoder_reexecuted=False))
    return record


def produce(request, output, base=None):
    job = raw.load_json(request)
    raw.require(raw.sha256(Path(job['campaign'])) == job['campaign_sha256'], 'Campaign changed')
    campaign, _, _, _, fit, ids = raw._campaign(Path(job['campaign']))
    people = [ids[i] for i in fit]
    identity = raw.chromosome_identity(campaign, job['chromosome'])
    raw.require(identity == job['input_identity'], 'Planned job identity differs')
    prior = job.get('completed')
    if prior:
        return reuse(prior['path'], output, identity, people, prior['manifest_sha256'])
    if base is None:
        raw.require(not identity['excluded_intervals'], 'Only missing full chromosomes may be decoded')
        return raw.convert_chromosome(Path(job['campaign']), job['chromosome'], Path(output))
    if not identity['excluded_intervals']:
        return reuse(base, output, identity, people, raw.sha256(Path(base)/'chromosome.json'))
    return exclusions.derive(base, output, identity, people,
        max_output_bytes=campaign['resources']['max_output_bytes'],
        min_free_bytes=campaign['resources']['min_free_bytes'])


def assemble(plan, products, output, storage_root=None, storage_alias=None):
    records = {Path(p).name:raw.load_json(Path(p)/'chromosome.json') for p in products}
    for record in records.values():
        if 'derivation' in record:
            d = record['derivation']
            raw.require(d['code_sha256'] == raw.sha256(Path(exclusions.__file__)) and
                        d['original_decoder_reexecuted'] is False and
                        d['scientific_models_admitted'] is False and
                        raw.sha256(Path(d['source_manifest'])) == d['source_manifest_sha256'],
                        'Derived product provenance changed')
    # One bind mount avoids EXDEV between Docker's separate task mounts. Without
    # this, a hardlink fallback could duplicate every chromosome across 88 views.
    if storage_root or storage_alias:
        raw.require(storage_root and storage_alias, 'Both storage paths required')
        root, alias = Path(storage_root).resolve(), Path(storage_alias).resolve()
        raw.require(root.stat().st_dev == alias.stat().st_dev and
                    root.stat().st_ino == alias.stat().st_ino, 'Storage alias is a different filesystem')
        def mapped(p):
            return alias / Path(p).resolve().relative_to(root)
        products = [mapped(p) for p in products]
        output = mapped(output)
    result = planning.assemble(plan, products, output)
    raw.write_json(Path(output)/'TECHNICAL_DERIVATIONS.json',dict(
        schema='r02_k03_exclusion_derivations_v1', products=records,
        code_sha256=raw.sha256(Path(__file__)), biological_validation=False,
        explanation='Original producer hashes identify the representation; derivation identifies each filtered product.'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    job=sub.add_parser('produce');job.add_argument('--request',required=True)
    job.add_argument('--output',required=True);job.add_argument('--base')
    a=sub.add_parser('assemble');a.add_argument('--plan',required=True)
    a.add_argument('--product',required=True,action='append');a.add_argument('--output',required=True)
    a.add_argument('--storage-root');a.add_argument('--storage-alias')
    args=p.parse_args()
    if args.command=='produce': produce(args.request,args.output,args.base)
    else: assemble(args.plan,args.product,args.output,args.storage_root,args.storage_alias)


if __name__=='__main__': main()
