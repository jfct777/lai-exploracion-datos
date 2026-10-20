#!/usr/bin/env python3
"""Plan and assemble every frozen K03 view with exact chromosome reuse.

No site selection or new allele orientation is learned here. An unchanged
chromosome is computed once, and never borrowed from a different exclusion.
"""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path

import r02_k03_event_alt_chunks as raw

SCHEMA = 'r02_k03_fixed_source_all_views_v1'


def deployment_sha256():
    """Include transitive local imports, not just the two entrypoint files."""
    root = Path(__file__).resolve().parent
    pending = ['r02_k03_event_view_plan', 'r02_k03_event_alt_chunks']
    found = {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = root / (name + '.py')
        raw.require(path.is_file(), 'Missing local deployment dependency: ' + name)
        found[name] = raw.sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            imported = ([n.name.split('.')[0] for n in node.names]
                        if isinstance(node, ast.Import) else
                        [node.module.split('.')[0]]
                        if isinstance(node, ast.ImportFrom) and node.module else [])
            pending.extend(n for n in imported if n.startswith('r02_'))
    return {name + '.py': digest for name, digest in sorted(found.items())}


def views(blocks_path, source):
    with Path(blocks_path).open() as stream:
        blocks = list(csv.DictReader(stream, delimiter='\t'))
    lengths = {c: d['length_bp'] for c, d in source['sources'].items()}
    for scope in ('BLOCK_AND_LD_MARGIN', 'WHOLE_CHROMOSOME'):
        for region in raw.input_plan.plan_regions(blocks, lengths, scope):
            c, start, end = region['excluded']
            yield {'view_id': scope + ':' + region['block_id'],
                   'excluded_intervals': [{'chromosome': int(c), 'start_bp': start, 'end_bp': end}],
                   'target_locus_policy': 'EXCLUDED_BEFORE_OUTPUT',
                   'source_candidate_rule':
                       'FIXED_SOURCE_COHORT_RARE_CANDIDATES_PRESERVED_NO_INNER_ROLE_FILTER'}


def plan(campaign_path, output):
    campaign_path, output = Path(campaign_path).resolve(), Path(output).resolve()
    campaign, source, _, blocks, _, _ = raw._campaign(campaign_path)
    raw.require(not output.exists(), 'New all-view plan directory required')
    # Resolve metadata references before copying campaign descriptions.
    for key in ('source_contract', 'roster', 'blocks'):
        campaign[key]['path'] = str(raw._pin(campaign[key], key, campaign_path.parent).resolve())
    for row in campaign['chromosomes']:
        for key in ('event_archive', 'event_manifest', 'tail_vcf', 'tail_manifest'):
            p = Path(row[key]['path'])
            row[key]['path'] = str(p if p.is_absolute() else (campaign_path.parent / p).resolve())
    output.mkdir(mode=0o700)
    jobs, entries = {}, []
    for territory in views(blocks, source):
        value = copy.deepcopy(campaign)
        value['territory'] = raw._territory(blocks, source, territory)
        name = territory['view_id'].replace(':', '__')
        path = output / (name + '.json')
        raw.write_json(path, value)
        references = []
        for c in range(1, 23):
            identity = raw.chromosome_identity(value, c)
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            identifier = f'chr{c:02d}_{digest[:16]}'
            if identifier in jobs:
                raw.require(jobs[identifier]['input_identity'] == identity, 'Task identity collision')
            else:
                jobs[identifier] = dict(id=identifier, chromosome=c, campaign=str(path),
                                        campaign_sha256=raw.sha256(path), input_identity=identity)
            references.append(identifier)
        entries.append(dict(id=name, campaign=str(path), campaign_sha256=raw.sha256(path),
                            chromosome_tasks=references))
    raw.require(len(entries) == 88 and len({v['id'] for v in entries}) == 88,
                'All 44 blocks and both exclusion scopes are required')
    value = dict(schema=SCHEMA, views=entries, jobs=list(jobs.values()),
                 logical_chromosome_uses=88*22, unique_chromosome_tasks=len(jobs),
                 implementation_sha256=raw.implementation_sha256(),
                 deployment_sha256=deployment_sha256(),
                 planner_sha256=raw.sha256(Path(__file__)),
                 scientific_models_admitted=False)
    raw.write_json(output / 'plan.json', value)
    return value


def assemble(plan_path, products, output):
    value = raw.load_json(plan_path)
    raw.require(value['schema'] == SCHEMA and value['planner_sha256'] == raw.sha256(Path(__file__))
                and value['implementation_sha256'] == raw.implementation_sha256(),
                'All-view plan or implementation differs')
    lookup = {Path(p).name: Path(p) for p in products}
    raw.require(len(lookup) == len(products) and set(lookup) == {j['id'] for j in value['jobs']},
                'Exactly the planned chromosome products are required')
    out = Path(output)
    raw.require(not out.exists(), 'New complete-view directory required')
    out.mkdir(mode=0o700)
    summaries = []
    for view in value['views']:
        raw.require(raw.sha256(Path(view['campaign'])) == view['campaign_sha256'],
                    'View campaign changed after planning')
        summaries.append(raw.assemble_shared(Path(view['campaign']),
            [lookup[j] for j in view['chromosome_tasks']], out / view['id']))
    raw.require(len(summaries) == 88, 'Incomplete view assembly')
    result = dict(status='ALL_88_FIXED_SOURCE_INPUT_VIEWS_ASSEMBLED', views=88,
                  unique_chromosome_tasks=len(lookup), biological_validation=False,
                  scientific_models_admitted=False, source_minor_redefined=False)
    raw.write_json(out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--campaign', required=True)
    p.add_argument('--output', required=True)
    a = sub.add_parser('assemble')
    a.add_argument('--plan', required=True)
    a.add_argument('--product', action='append', required=True)
    a.add_argument('--output', required=True)
    args = parser.parse_args()
    result = (plan(args.campaign, args.output) if args.command == 'plan'
              else assemble(args.plan, args.product, args.output))
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','views','implementation_sha256')}
                     if args.command == 'plan' else result))


if __name__ == '__main__':
    main()
