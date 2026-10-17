#!/usr/bin/env python3
"""Summarize sealed L1 genotype-support tables; never decode or recount GT.

One streaming task consumes one chromosome and one panel role. ALL and ancestry
strata describe the same people and must never be added together. Counts summed
across loci are person-locus observations/allele copies, not distinct people.
Missing records, unresolved allele representation, missing GT and observed zero
copies remain separate. GT_ONLY and observed phase do not certify quality,
haplotype accuracy, donor independence or suitability for simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

SCHEMA = 'r02_lai_support_review_v1'
STATUS = 'COMPLETE_DESCRIPTIVE_SUPPORT_REVIEW_NOT_PHASE_OR_LAI_VALIDATION'
ROLES = {'REF_TRAIN', 'SOURCE_VALID'}
TABLE = 'soporte_alelos_dnabr.tsv.gz'
BUNDLE_FILES = {'manifest.json', TABLE, 'map_support.json', 'complete.json', 'stage_receipt.json'}
CALLED = ('n_gt_complete', 'n_gt_missing', 'n_gt_non_diploid', 'an_called',
          'ac_counted_called', 'n_carriers_called', 'n_phased_gt_complete',
          'n_phased_carriers_called', 'n_site_filter_failed')
QUALITY = ('n_quality_evaluable', 'an_evaluable', 'ac_counted_evaluable',
           'n_carriers_evaluable', 'n_phased_carriers_evaluable',
           'n_homozygous_carriers_evaluable', 'n_quality_missing', 'n_quality_failed')
REQUIRED = ('chromosome', 'position_bp', 'REF', 'ALT', 'counted_allele', 'panel_role', 'panel_ancestry', 'n_role_members',
            'correspondence_status', 'genotype_evidence_status', 'quality_status',
            'phase_status', 'n_phase_usable_carriers', *CALLED, *QUALITY)
LOCI = ('source_loci', 'no_record_loci', 'unresolved_ref_loci', 'other_incompatible_loci',
        'genotype_read_loci', 'no_complete_gt_loci', 'positive_called_loci',
        'zero_called_all_members_loci', 'zero_called_partial_members_loci',
        'positive_quality_evaluable_loci', 'zero_quality_all_members_loci',
        'zero_quality_partial_members_loci', 'no_quality_evaluable_loci')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def check_space(path, reserve):
    require(shutil.disk_usage(path).free >= reserve * 1024**3, 'Disk reserve insufficient')


def authenticate(bundle, contract):
    """Bind completion/role/provenance to bytes, before and after streaming."""
    require(contract.get('schema') == 'r02_lai_support_review_input_v1', 'Invalid input schema')
    require(contract.get('role') in ROLES, 'Forbidden or unknown panel role')
    require(str(contract.get('chromosome')) in {str(i) for i in range(1, 23)}, 'Invalid autosome')
    require(set(contract.get('sha256', {})) == BUNDLE_FILES, 'Incomplete sealed input inventory')
    for name, digest in contract['sha256'].items():
        require(sha(bundle / name) == digest, 'Input SHA256 mismatch: ' + name)
    manifest, complete, stage = (read(bundle / n) for n in ('manifest.json', 'complete.json', 'stage_receipt.json'))
    require(manifest.get('schema') == 'r02_lai_genotype_support_v1' and
            manifest.get('status') == 'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION',
            'Incomplete or unsupported source manifest')
    require(complete.get('status') == 'COMPLETE_GENOTYPE_AND_MAP_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION'
            and complete.get('publication_manifest_sha256') == contract['sha256']['manifest.json'],
            'Completion is not bound to the publication manifest')
    require(stage.get('schema') == 'r02_lai_phased_stage_v1' and stage.get('stage') == 'support',
            'Missing final support-stage receipt')
    for item in (manifest, complete, stage):
        require(str(item.get('chromosome')) == contract['chromosome'], 'Chromosome differs from contract')
    for item in (complete, stage):
        require(item.get('panel_role') == contract['role'] and item.get('source_test_projected') is False,
                'Role mismatch or TEST source opened')
    require(manifest['role_validation']['role'] == contract['role'] and
            manifest['genotype_contract']['panel_role'] == contract['role'], 'Manifest role mismatch')
    require(manifest.get('contains_sample_identifiers') is False and
            manifest.get('contains_individual_genotypes') is False, 'Unexpected individual records')
    require(manifest.get('donor_phase_support') == 'OBSERVED_PHASE_ONLY_NOT_CERTIFIED', 'Unexpected phase claim')
    require(manifest.get('outputs_sha256') == {n: contract['sha256'][n] for n in (TABLE, 'map_support.json')},
            'Manifest output inventory differs')
    mode = manifest['genotype_contract']['quality']['mode']
    require(mode in ('GT_ONLY', 'DP_GQ'), 'Unknown quality mode')
    strata = manifest['ancestry_strata']
    require('ALL' in strata and len(strata) > 1 and all(type(v) is int and v > 0 for v in strata.values())
            and sum(v for k, v in strata.items() if k != 'ALL') == strata['ALL'], 'Invalid ancestry denominators')
    require(strata['ALL'] == manifest['role_validation']['n_role_members'], 'Role total differs')
    return manifest


def numeric(value, name, *, optional=False):
    if value == 'NA' and optional:
        return None
    require(value.isdecimal(), 'Invalid nonnegative count: ' + name)
    return int(value)


def summarize(table, manifest, reserve_gib=12, timeout_seconds=7200):
    """Stream O(strata) memory, maintaining exact denominators and NA status."""
    started = time.monotonic()
    strata = manifest['ancestry_strata']
    totals = {name: Counter() for name in strata}
    categories = {name: Counter() for name in strata}
    positions = dict.fromkeys(strata, 0)
    locus_hashes = {name: hashlib.sha256() for name in strata}
    mode = manifest['genotype_contract']['quality']['mode']
    role, chrom = manifest['role_validation']['role'], manifest['chromosome']
    row_count = 0
    with gzip.open(table, 'rt') as handle:
        fields = handle.readline().rstrip('\n').split('\t')
        require(len(fields) == len(set(fields)) and set(REQUIRED) <= set(fields), 'Missing or duplicated table columns')
        index = {name: fields.index(name) for name in REQUIRED}
        # Positional field access avoids allocating a 52-key dictionary for each
        # of the many NO_RECORD rows. No per-locus or per-person matrix is kept.
        for line in handle:
            cells = line.rstrip('\n').split('\t')
            require(len(cells) == len(fields), 'Malformed support row')
            ancestry = cells[index['panel_ancestry']]
            require(ancestry in strata, 'Unknown ancestry stratum')
            n = strata[ancestry]
            require(cells[index['panel_role']] == role and cells[index['chromosome']] == chrom
                    and cells[index['n_role_members']] == str(n), 'Row chromosome/role/denominator differs')
            position = numeric(cells[index['position_bp']], 'position')
            require(position > positions[ancestry], 'Duplicate or decreasing locus')
            positions[ancestry] = position
            alleles = [cells[index[k]] for k in ('REF', 'ALT', 'counted_allele')]
            require(alleles[0] in ('A','C','G','T') and alleles[1] in ('A','C','G','T')
                    and alleles[0] != alleles[1] and alleles[2] in alleles[:2], 'Invalid source allele identity')
            locus_hashes[ancestry].update(('\t'.join([str(position), *alleles])+'\n').encode())
            require(cells[index['n_phase_usable_carriers']] == 'NA', 'Observed phase promoted to certified phase')
            c, cats = totals[ancestry], categories[ancestry]
            status = cells[index['correspondence_status']]
            c['source_loci'] += 1
            cats[status] += 1
            evidence = cells[index['genotype_evidence_status']]
            if evidence == 'NO_GENOTYPE_EVIDENCE_IN_THIS_SOURCE':
                require(all(cells[index[key]] == 'NA' for key in (*CALLED, *QUALITY)),
                        'No-record or unresolved allele converted to genotype counts')
                c['no_record_loci' if status == 'NO_RECORD' else 'unresolved_ref_loci'
                  if status.startswith('UNKNOWN_REF_DOSAGE_') else 'other_incompatible_loci'] += 1
            else:
                require(evidence == 'GT_READ' and status in ('EXACT_KEY', 'PANEL_MULTIALLELIC_ALLELE_MAPPED'),
                        'Unexpected genotype evidence/correspondence')
                c['genotype_read_loci'] += 1
                values = {key: numeric(cells[index[key]], key, optional=key == 'ac_counted_called') for key in CALLED}
                ngt, an, ac, carriers = (values[key] for key in ('n_gt_complete', 'an_called', 'ac_counted_called', 'n_carriers_called'))
                require(ngt + values['n_gt_missing'] + values['n_gt_non_diploid'] == n and an == 2 * ngt,
                        'Called genotype denominator inconsistent')
                require(0 <= carriers <= ngt and (ac is None if an == 0 else ac is not None and carriers <= ac <= 2 * carriers),
                        'Called allele/carrier count inconsistent')
                require(values['n_phased_gt_complete'] <= ngt and
                        values['n_phased_carriers_called'] <= min(carriers, values['n_phased_gt_complete']) and
                        values['n_site_filter_failed'] <= ngt, 'Invalid phase/site counts')
                require(cells[index['phase_status']] == 'OBSERVED_PHASE_NOT_VALIDATED', 'Unexpected phase state')
                for key, value in values.items():
                    if value is not None:
                        c[key] += value
                c['no_complete_gt_loci' if an == 0 else 'positive_called_loci' if ac else
                  'zero_called_all_members_loci' if ngt == n else 'zero_called_partial_members_loci'] += 1
                if mode == 'GT_ONLY':
                    require(all(cells[index[key]] == 'NA' for key in QUALITY) and
                            cells[index['quality_status']] == 'NOT_EVALUATED_GT_ONLY', 'GT_ONLY misrepresented as quality-evaluable')
                else:
                    require(cells[index['quality_status']] == 'EVALUATED_UNDER_SUPPLIED_FILTERS', 'Quality state differs')
                    q = {key: numeric(cells[index[key]], key, optional=key == 'ac_counted_evaluable') for key in QUALITY}
                    nq, aq, cq, kq = (q[key] for key in ('n_quality_evaluable', 'an_evaluable', 'ac_counted_evaluable', 'n_carriers_evaluable'))
                    require(nq <= ngt and aq == 2 * nq and kq <= nq and
                            (cq is None if aq == 0 else cq is not None and kq <= cq <= 2 * kq), 'Quality denominator inconsistent')
                    require(q['n_phased_carriers_evaluable'] <= min(kq, values['n_phased_carriers_called'])
                            and q['n_homozygous_carriers_evaluable'] <= kq, 'Invalid quality/phase counts')
                    require(ngt == values['n_site_filter_failed'] + q['n_quality_missing'] + q['n_quality_failed'] + nq,
                            'Quality categories do not partition complete GT')
                    require(cq is None or cq == kq + q['n_homozygous_carriers_evaluable'],
                            'Quality allele/carrier/homozygote counts inconsistent')
                    require(kq <= carriers and (cq is None or cq <= ac)
                            and q['n_homozygous_carriers_evaluable'] <= (ac or 0) - carriers,
                            'Quality carriers/copies are not a subset of called genotypes')
                    for key, value in q.items():
                        if value is not None:
                            c[key] += value
                    c['no_quality_evaluable_loci' if aq == 0 else 'positive_quality_evaluable_loci' if cq else
                      'zero_quality_all_members_loci' if nq == n else 'zero_quality_partial_members_loci'] += 1
            row_count += 1
            if row_count % 100000 == 0:
                check_space(Path(table).parent, reserve_gib)
                require(time.monotonic() - started < timeout_seconds, 'Review time budget exceeded')
    expected = manifest['counts']['source_rows']
    require(all(c['source_loci'] == expected for c in totals.values()) and row_count == manifest['output_rows'],
            'Incomplete source catalogue or ancestry strata')
    require(len({h.hexdigest() for h in locus_hashes.values()}) == 1,
            'Locus/allele sequence differs between ALL and ancestry strata')
    for name, cats in categories.items():
        require(dict(cats) == {key: manifest['counts'][key] for key in cats}, 'Correspondence counts differ from source manifest')
    for key in (*CALLED, *(QUALITY if mode == 'DP_GQ' else ())):
        require(totals['ALL'][key] == sum(v[key] for k, v in totals.items() if k != 'ALL'),
                'ALL total is not equal to ancestry partition: ' + key)
    result = []
    for ancestry, c in totals.items():
        row = dict(chromosome=chrom, panel_role=role, panel_ancestry=ancestry, n_distinct_role_members=strata[ancestry], quality_mode=mode)
        row.update({key: c[key] for key in LOCI})
        row.update({key + '_person_locus_sum': c[key] for key in CALLED})
        row.update({key + '_person_locus_sum': c[key] if mode == 'DP_GQ' else 'NA' for key in QUALITY})
        if mode == 'GT_ONLY':
            for key in LOCI[9:]:
                row[key] = 'NA'
        row['genotype_read_person_locus_opportunities'] = c['genotype_read_loci'] * strata[ancestry]
        row['catalogue_person_locus_opportunities'] = expected * strata[ancestry]
        row['called_gt_fraction_at_read_loci'] = c['n_gt_complete'] / row['genotype_read_person_locus_opportunities'] if row['genotype_read_person_locus_opportunities'] else 'NA'
        row['fraction_catalogue_loci_with_called_carriers'] = c['positive_called_loci'] / expected if expected else 'NA'
        result.append(row)
    return result, categories


def run(contract_path, bundle, outdir, reserve_gib=12, timeout_seconds=7200):
    started = time.monotonic()
    bundle, out = Path(bundle), Path(outdir)
    require(not out.exists(), 'Output exists; use Nextflow resume')
    require(reserve_gib >= 0 and timeout_seconds > 0, 'Invalid resource budget')
    check_space(out.parent, reserve_gib)
    contract = read(contract_path)
    contract_digest = sha(contract_path)
    manifest = authenticate(bundle, contract)
    rows, categories = summarize(bundle / TABLE, manifest, reserve_gib, timeout_seconds)
    mapped = read(bundle / 'map_support.json')
    require(mapped.get('chromosome') == contract['chromosome'] and mapped.get('extrapolation') is False,
            'Unexpected map/chromosome or extrapolation')
    require(sum(mapped['counts'].get('source_sites_' + k, 0) for k in ('inside', 'outside')) == manifest['counts']['source_rows'],
            'Map endpoint counts differ from catalogue')
    authenticate(bundle, contract)
    require(sha(contract_path) == contract_digest, 'Input contract changed')
    out.mkdir(mode=0o700)
    with (out / 'support_summary.tsv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    with (out / 'correspondence_summary.tsv').open('x') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(('chromosome', 'panel_role', 'panel_ancestry', 'correspondence_status', 'loci'))
        for ancestry, counter in categories.items():
            for state, count in sorted(counter.items()):
                writer.writerow((contract['chromosome'], contract['role'], ancestry, state, count))
    definitions = dict(
        unit='One row per chromosome/role/ancestry; no individual IDs or variant rows',
        denominators='ALL overlaps ancestry rows. Never add ALL to its ancestry strata. n_distinct_role_members is reused across chromosomes, not additive.',
        sums='Counts across loci are allele-copy or person-locus sums, not distinct people, independent observations or population frequencies.',
        no_record='No compatible source record is unknown; it is not a missing GT at an observed locus or zero copies.',
        unresolved_ref='A matching site with unknown allele-decomposition provenance cannot certify REF dosage.',
        zero_called='Zero copies among complete called diploid GT; all/partial refers to how many role members have complete GT, not quality validation.',
        quality='GT_ONLY has NA quality-evaluable counts. Raw called GT may include site-filter failures; those are counted separately.',
        phase='Observed VCF phase flags are not measured accuracy or certified donor copiability.',
        map='Map endpoints describe interpolation range only; not ancestry-specific map accuracy.',
        interpretation='Feasibility/descriptive review only. No simulation promotion, ancestry truth, rare-variant utility or LAI validation.',
        map_endpoint_summary=mapped)
    write(out / 'definitions.json', definitions)
    output = dict(schema=SCHEMA, status=STATUS, evidence_level='DESCRIPTIVE_FEASIBILITY',
        chromosome=contract['chromosome'], panel_role=contract['role'], contains_sample_identifiers=False,
        contains_individual_genotypes=False, contains_variant_rows=False, source_test_opened=False,
        simulations_executed=False, phase_validated=False, input_contract_sha256=contract_digest,
        source_manifest_sha256=contract['sha256']['manifest.json'], source_table_sha256=contract['sha256'][TABLE],
        source_sample_members_sha256=manifest['genotype_contract']['panel_sample_members_sha256'],
        source_catalogue_sha256=manifest['input_files']['rare_vcf']['sha256'],
        role_members=manifest['ancestry_strata'], source_loci=manifest['counts']['source_rows'],
        quality_mode=manifest['genotype_contract']['quality']['mode'],
        outputs_sha256={f.name: sha(f) for f in out.iterdir()}, code_sha256=sha(__file__),
        elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write(out / 'manifest.json', output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'bundle', 'outdir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--min-free-gib', type=float, default=12)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    args = parser.parse_args(argv)
    run(args.contract, args.bundle, args.outdir, args.min_free_gib, args.timeout_seconds)


if __name__ == '__main__':
    os.umask(0o077)
    main()
