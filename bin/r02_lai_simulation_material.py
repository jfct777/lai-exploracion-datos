#!/usr/bin/env python3
"""Join sealed L1c roles by exact source allele, without decoding genotypes.

This is simulation-material feasibility, not approval to simulate. Every source
locus and ancestry stratum remains present, including unmeasured records. ALL
overlaps its ancestry strata. Phase flags and GT_ONLY do not validate quality,
haplotype accuracy, independent donors or genetic-map appropriateness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time

import r02_lai_support_review as support

SCHEMA = 'r02_lai_simulation_material_v1'
STATUS = 'COMPLETE_JOINT_MATERIAL_FEASIBILITY_NOT_SIMULATION_READY'
ROLES = ('REF_TRAIN', 'SOURCE_VALID')
KEY = ('chromosome', 'position_bp', 'REF', 'ALT', 'counted_allele')
SOURCE = ('source_build', 'source_rare_allele_index', 'source_rare_ac', 'source_rare_an',
          'source_cohort_n', 'source_cohort_sha256', 'source_catalogue_sha256',
          'original_allele_count', 'source_site_id', 'source_record_id')
COUNTS = ('n_role_members', *support.CALLED, *support.QUALITY, 'n_phase_usable_carriers')
STATE_FIELDS = ('correspondence_status', 'genotype_evidence_status', 'quality_status', 'phase_status')
require, sha, read, write = support.require, support.sha, support.read, support.write


def snapshot(paths):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, sha(p)) for p in paths}


def audit_role_disjointness(roles_path, manifests):
    """Authenticate only role metadata; no genotype or TEST data are opened."""
    digest = sha(roles_path)
    require(all(m['input_files']['roles']['sha256'] == digest for m in manifests.values()), 'Role source hash differs')
    selected = {r: set() for r in ROLES}; strata = {r: Counter() for r in ROLES}
    with Path(roles_path).open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require({'sample_id', 'role', 'ancestry'} <= set(reader.fieldnames or []), 'Role metadata fields missing')
        for row in reader:
            role = row['role']
            if role not in ROLES: continue
            require(row['sample_id'] and row['sample_id'] not in selected[role], 'Duplicate selected role member')
            selected[role].add(row['sample_id']); strata[role][row['ancestry']] += 1
    require(not selected[ROLES[0]] & selected[ROLES[1]], 'Reference/donor individuals overlap')
    hashes = {r: hashlib.sha256(('\n'.join(sorted(selected[r])) + '\n').encode()).hexdigest() for r in ROLES}
    for role in ROLES:
        require(hashes[role] == manifests[role]['genotype_contract']['panel_sample_members_sha256'] and
                len(selected[role]) == manifests[role]['ancestry_strata']['ALL'] and
                dict(strata[role]) == {a:n for a,n in manifests[role]['ancestry_strata'].items() if a != 'ALL'},
                'Selected metadata membership/ancestry differs from L1c')
    require(sha(roles_path) == digest, 'Role metadata changed during audit')
    return dict(schema='r02_lai_role_disjointness_v1', status='EXACT_MEMBERSHIP_DISJOINT_NOT_KINSHIP_INDEPENDENCE',
                roles_sha256=digest, role_members_sha256=hashes, role_counts={r:len(selected[r]) for r in ROLES},
                overlap_individuals=0, source_test_genotypes_opened=False, contains_sample_identifiers=False,
                interpretation='No individual belongs to both roles; relatedness and ancestry representativeness are not certified.')


def authenticate(root, contract):
    require(contract.get('schema') == 'r02_lai_simulation_material_input_v1', 'Invalid material contract')
    require(set(contract.get('roles', {})) == set(ROLES), 'Exactly REF_TRAIN/SOURCE_VALID required; TEST forbidden')
    chrom = contract.get('chromosome')
    require(chrom in {str(i) for i in range(1, 23)}, 'Invalid chromosome')
    manifests, reviews, paths = {}, {}, []
    for role in ROLES:
        item = contract['roles'][role]
        bundle = root / role / 'source'
        source_contract = item['source_contract']
        require(source_contract.get('role') == role and source_contract.get('chromosome') == chrom,
                'Source contract chromosome/role differs')
        manifests[role] = support.authenticate(bundle, source_contract)
        paths += [bundle / n for n in source_contract['sha256']]
        review_dir = root / role / 'review'
        require(set(item['review_sha256']) == {'manifest.json', 'support_summary.tsv',
                 'correspondence_summary.tsv', 'definitions.json'}, 'Incomplete review inventory')
        for name, digest in item['review_sha256'].items():
            require(sha(review_dir / name) == digest, 'Review SHA256 mismatch')
            paths.append(review_dir / name)
        reviewed = read(review_dir / 'manifest.json')
        require(reviewed.get('schema') == support.SCHEMA and reviewed.get('status') == support.STATUS,
                'Source review not complete')
        require(reviewed['chromosome'] == chrom and reviewed['panel_role'] == role and
                reviewed['source_manifest_sha256'] == source_contract['sha256']['manifest.json'] and
                reviewed['source_table_sha256'] == source_contract['sha256'][support.TABLE] and
                reviewed['source_loci'] == manifests[role]['counts']['source_rows'] and
                reviewed['role_members'] == manifests[role]['ancestry_strata'], 'Review/source binding differs')
        require(reviewed['outputs_sha256'] == {n: item['review_sha256'][n]
                for n in item['review_sha256'] if n != 'manifest.json'}, 'Review output inventory differs')
        with (review_dir / 'support_summary.tsv').open() as handle:
            values = list(csv.DictReader(handle, delimiter='\t'))
        require(len(values) == len(manifests[role]['ancestry_strata']) and
                {r['panel_ancestry'] for r in values} == set(manifests[role]['ancestry_strata']),
                'Incomplete reviewed strata')
        reviews[role] = {r['panel_ancestry']: r for r in values}
    ref, donor = (manifests[r] for r in ROLES)
    for name in ('rare_vcf', 'rare_contract', 'roles'):
        require(ref['input_files'][name]['sha256'] == donor['input_files'][name]['sha256'],
                'Different source catalogue, convention or role assignment: ' + name)
    require(ref['counts']['source_rows'] == donor['counts']['source_rows'] and
            ref['source_cohort_n'] == donor['source_cohort_n'] and
            set(ref['ancestry_strata']) == set(donor['ancestry_strata']), 'Source population/strata differ')
    require(ref['genotype_contract']['panel_sample_members_sha256'] !=
            donor['genotype_contract']['panel_sample_members_sha256'], 'Identical role memberships')
    disjoint_path = root / 'role_disjointness.json'
    require(sha(disjoint_path) == contract['role_disjointness_sha256'], 'Disjointness receipt SHA differs')
    disjoint = read(disjoint_path); paths.append(disjoint_path)
    require(disjoint.get('schema') == 'r02_lai_role_disjointness_v1' and
            disjoint.get('status') == 'EXACT_MEMBERSHIP_DISJOINT_NOT_KINSHIP_INDEPENDENCE' and
            disjoint.get('roles_sha256') == ref['input_files']['roles']['sha256'] and
            disjoint.get('overlap_individuals') == 0 and disjoint.get('source_test_genotypes_opened') is False,
            'Role disjointness not established')
    require(disjoint.get('role_members_sha256') == {r:manifests[r]['genotype_contract']['panel_sample_members_sha256'] for r in ROLES}
            and disjoint.get('role_counts') == {r:manifests[r]['ancestry_strata']['ALL'] for r in ROLES},
            'Disjointness receipt membership differs')
    for manifest in manifests.values():
        require(manifest['build_status'] == 'BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE', 'Build not verified')
        require(manifest['source_frequency_unchanged'] is True and
                manifest['source_catalogue_all_records_preserved'] is True, 'Source catalogue altered')
    require(ref['build_evidence']['source'] == donor['build_evidence']['source'], 'Source build evidence differs')
    maps = [read(root / role / 'source/map_support.json') for role in ROLES]
    for mapped in maps:
        require(mapped.get('schema') == 'r02_lai_phased_map_support_v1' and
                mapped.get('chromosome') == chrom and mapped.get('extrapolation') is False,
                'Invalid map contract')
        require(type(mapped['first_bp']) is int and type(mapped['last_bp']) is int and
                mapped['first_bp'] < mapped['last_bp'], 'Invalid map endpoints')
    for field in ('map_sha256', 'map_points', 'first_bp', 'last_bp'):
        require(maps[0][field] == maps[1][field], 'Role map mismatch: ' + field)
    return manifests, reviews, maps[0], paths


def state(row):
    status = row['correspondence_status']
    if row['genotype_evidence_status'] == 'NO_GENOTYPE_EVIDENCE_IN_THIS_SOURCE':
        require(all(row[k] == 'NA' for k in COUNTS if k != 'n_role_members'), 'Unknown record converted to counts')
        return ('NO_RECORD' if status == 'NO_RECORD' else 'COUNTED_ALLELE_UNRESOLVED'
                if status.startswith('UNKNOWN_REF_DOSAGE_') else 'INCOMPATIBLE_RECORD')
    require(row['genotype_evidence_status'] == 'GT_READ' and
            status in ('EXACT_KEY', 'PANEL_MULTIALLELIC_ALLELE_MAPPED'), 'Unsupported genotype evidence')
    n, gt, missing, hap, an = (int(row[k]) for k in
        ('n_role_members', 'n_gt_complete', 'n_gt_missing', 'n_gt_non_diploid', 'an_called'))
    require(gt + missing + hap == n and an == 2 * gt, 'Genotype partition differs')
    require(row['n_phase_usable_carriers'] == 'NA', 'Observed phase promoted to validated phase')
    if not an:
        require(row['ac_counted_called'] == 'NA' and int(row['n_carriers_called']) == 0, 'Zero denominator must remain NA')
        return 'GT_UNKNOWN_ALL'
    ac, carriers = int(row['ac_counted_called']), int(row['n_carriers_called'])
    require(0 <= carriers <= gt and carriers <= ac <= 2 * carriers, 'Invalid allele/carrier counts')
    if ac:
        return 'OBSERVED_COPIES'
    return 'OBSERVED_ZERO_ALL_CALLED' if gt == n else 'OBSERVED_ZERO_PARTIAL_CALLED'


def joint_class(ref_state, donor_state):
    if donor_state != 'OBSERVED_COPIES':
        return 'DONOR_' + donor_state
    return 'DONOR_COPIES_REF_' + ref_state


def candidacy(donor, donor_state, inside):
    if donor_state != 'OBSERVED_COPIES':
        return 'NO_OBSERVED_DONOR_COPY_OR_UNKNOWN'
    if not inside:
        return 'DONOR_COPIES_OUTSIDE_MAP_RANGE'
    if int(donor['n_site_filter_failed']):
        return 'DONOR_COPIES_SITE_FILTER_FAILURE_PRESENT'
    if int(donor['n_phased_carriers_called']) == 0:
        return 'DONOR_COPIES_PHASE_NOT_OBSERVED'
    return 'OBSERVED_PHASED_DONOR_CANDIDATE_REQUIRES_VALIDATION'


def stream(root, manifests, reviews, mapped, out, reserve_gib, timeout_seconds, max_output_bytes):
    started = time.monotonic()
    strata = list(manifests['REF_TRAIN']['ancestry_strata'])
    tallies = {a: Counter() for a in strata}
    marginals = {r: {a: Counter() for a in strata} for r in ROLES}
    positions = dict.fromkeys(strata, 0)
    current_key, seen = None, set()
    fields = (*KEY, *SOURCE, 'panel_ancestry', 'map_range_status', 'joint_observation',
              'material_candidacy', *[r + '_' + k for r in ROLES for k in ('observation_state', *STATE_FIELDS, *COUNTS)])
    rows_count, locus_count = 0, 0
    with gzip.open(root / ROLES[0] / 'source' / support.TABLE, 'rt') as h1, \
         gzip.open(root / ROLES[1] / 'source' / support.TABLE, 'rt') as h2, \
         gzip.open(out / 'joint_material.tsv.gz', 'wt', compresslevel=1) as dest:
        readers = [csv.DictReader(h, delimiter='\t') for h in (h1, h2)]
        required = {*KEY, *SOURCE, 'panel_ancestry', 'panel_role', *STATE_FIELDS, *COUNTS}
        for reader in readers:
            require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)) and
                    required <= set(reader.fieldnames), 'Missing/duplicated support fields')
        writer = csv.writer(dest, delimiter='\t', lineterminator='\n'); writer.writerow(fields)
        for ref, donor in itertools.zip_longest(*readers):
            require(ref is not None and donor is not None, 'Incomplete role catalogue')
            key = tuple(ref[k] for k in KEY)
            require(key == tuple(donor[k] for k in KEY) and
                    all(ref[k] == donor[k] for k in SOURCE), 'Exact source allele/provenance mismatch')
            ancestry = ref['panel_ancestry']
            require(ancestry == donor['panel_ancestry'] and ancestry in strata, 'Ancestry order/identity differs')
            require(ref['panel_role'] == ROLES[0] and donor['panel_role'] == ROLES[1], 'Role changed in table')
            require(ref['chromosome'] == manifests[ROLES[0]]['chromosome'], 'Chromosome changed in table')
            pos = int(ref['position_bp'])
            require(pos > positions[ancestry], 'Repeated or unsorted source locus')
            positions[ancestry] = pos
            if key != current_key:
                require(current_key is None or seen == set(strata), 'Missing ancestry stratum at locus')
                current_key, seen = key, set(); locus_count += 1
            require(ancestry not in seen, 'Duplicate ancestry stratum'); seen.add(ancestry)
            inside = mapped['first_bp'] <= pos <= mapped['last_bp']
            states = [state(row) for row in (ref, donor)]
            classification = joint_class(*states); candidate = candidacy(donor, states[1], inside)
            c = tallies[ancestry]
            c['source_loci'] += 1; c['joint:' + classification] += 1; c['candidate:' + candidate] += 1
            c['map_inside_loci' if inside else 'map_outside_loci'] += 1
            values = [*key, *(ref[k] for k in SOURCE), ancestry, 'INSIDE' if inside else 'OUTSIDE', classification, candidate]
            for role, row, observed in zip(ROLES, (ref, donor), states):
                require(row['n_role_members'] == str(manifests[role]['ancestry_strata'][ancestry]), 'Role denominator changed')
                marginals[role][ancestry][observed] += 1
                values += [observed, *(row[k] for k in (*STATE_FIELDS, *COUNTS))]
            writer.writerow(values); rows_count += 1
            if rows_count % 10000 == 0:
                support.check_space(out, reserve_gib)
                require(time.monotonic() - started <= timeout_seconds, 'Material task deadline exceeded')
                require((out / 'joint_material.tsv.gz').stat().st_size <= max_output_bytes, 'Output budget exceeded')
        require(seen == set(strata), 'Incomplete final ancestry group')
    expected = manifests[ROLES[0]]['counts']['source_rows']
    require(locus_count == expected and rows_count == expected * len(strata), 'Incomplete joint catalogue')
    require(tallies['ALL']['map_inside_loci'] == mapped['counts'].get('source_sites_inside', 0) and
            tallies['ALL']['map_outside_loci'] == mapped['counts'].get('source_sites_outside', 0), 'Map coverage differs')
    support.check_space(out, reserve_gib)
    require(time.monotonic() - started <= timeout_seconds, 'Material task deadline exceeded')
    require((out / 'joint_material.tsv.gz').stat().st_size <= max_output_bytes, 'Output budget exceeded')
    mappings = {'NO_RECORD':'no_record_loci', 'COUNTED_ALLELE_UNRESOLVED':'unresolved_ref_loci',
                'INCOMPATIBLE_RECORD':'other_incompatible_loci', 'GT_UNKNOWN_ALL':'no_complete_gt_loci',
                'OBSERVED_COPIES':'positive_called_loci', 'OBSERVED_ZERO_ALL_CALLED':'zero_called_all_members_loci',
                'OBSERVED_ZERO_PARTIAL_CALLED':'zero_called_partial_members_loci'}
    for role in ROLES:
        for ancestry in strata:
            require(sum(marginals[role][ancestry].values()) == expected, 'Missing role observations')
            for observed, field in mappings.items():
                require(marginals[role][ancestry][observed] == int(reviews[role][ancestry][field]),
                        'Joint marginals disagree with sealed review: ' + role + '/' + ancestry + '/' + field)
    with (out / 'joint_summary.tsv').open('x') as h:
        w = csv.writer(h, delimiter='\t', lineterminator='\n')
        w.writerow(('chromosome','panel_ancestry','category','loci','source_loci_denominator','fraction',
                    'reference_members','donor_members'))
        for ancestry, c in tallies.items():
            for category in sorted(k for k in c if k != 'source_loci'):
                w.writerow((manifests[ROLES[0]]['chromosome'], ancestry, category, c[category], expected,
                            c[category]/expected if expected else 'NA',
                            manifests[ROLES[0]]['ancestry_strata'][ancestry], manifests[ROLES[1]]['ancestry_strata'][ancestry]))
    return rows_count, locus_count


def run(contract_path, bundle, outdir, reserve_gib=12, timeout_seconds=7200, max_output_bytes=2*1024**3):
    started = time.monotonic(); root, out = Path(bundle), Path(outdir)
    require(reserve_gib >= 0 and timeout_seconds > 0 and max_output_bytes > 0, 'Invalid resource limit')
    require(not out.exists(), 'Output exists; use Nextflow resume')
    contract_digest = sha(contract_path)
    contract = read(contract_path)
    manifests, reviews, mapped, paths = authenticate(root, contract)
    before = snapshot([Path(contract_path), *paths])
    require(before[str(Path(contract_path))][2] == contract_digest, 'Input contract changed during authentication')
    require(before[str(root / 'role_disjointness.json')][2] == contract['role_disjointness_sha256'],
            'Disjointness receipt changed before snapshot')
    for role in ROLES:
        for name, digest in contract['roles'][role]['source_contract']['sha256'].items():
            require(before[str(root / role / 'source' / name)][2] == digest, 'Authenticated source changed before snapshot')
        for name, digest in contract['roles'][role]['review_sha256'].items():
            require(before[str(root / role / 'review' / name)][2] == digest, 'Authenticated review changed before snapshot')
    out.mkdir(parents=True, mode=0o700); support.check_space(out, reserve_gib)
    rows, loci = stream(root, manifests, reviews, mapped, out, reserve_gib, timeout_seconds, max_output_bytes)
    require(snapshot([Path(contract_path), *paths]) == before, 'Input changed during material join')
    definitions = dict(
        purpose='Exact joint source-allele feasibility; no genotype recount, simulation, training, TEST or donor reassignment.',
        row='One exact chromosome/position/REF/ALT/counted-allele and ancestry stratum. ALL overlaps ancestry rows.',
        source_frequency='Frozen DNABR source AC/AN/cohort; not redefined in either external role.',
        unknown='NO_RECORD is unmeasured here, not absent in the population. Unknown counted REF is not zero. GT_UNKNOWN_ALL is a measured record with no complete diploid calls.',
        zero='Zero observed copies with AN>0; ALL_CALLED and PARTIAL_CALLED distinguish denominator completeness. Neither proves biological absence.',
        candidate='Observed donor copies within map endpoints, no recorded site-filter failure and some phased carrier are only candidates for later validation; not certified copyable haplotypes.',
        quality='GT_ONLY leaves quality-evaluable fields NA. Raw called counts do not establish quality, even if all people were called.',
        phase='Observed phase is not haplotype accuracy or certified regional phase. No haplotype assignment of unknown-phase rare heterozygotes.',
        map='Authenticated identical map SHA between roles; endpoints only, not ancestry-specific accuracy or extrapolation.',
        independence='Authenticated role metadata establish no individual in both roles. This does not certify unrelatedness, population representativeness or no previous exposure.',
        empty_strata='Current sealed upstream contracts require positive stratum membership. A future unsampled ancestry needs an explicit NO_ROLE_MEMBERS contract, never GT_UNKNOWN_ALL.',
        summary='Every fraction uses source_loci within that chromosome/ancestry. Joint categories partition loci; candidacy categories are a separate partition. Do not add both partitions or ALL to ancestries.',
        readiness='NOT_SIMULATION_READY: quality, regional phase, donor independence/representativeness and simulation design require separate evidence.')
    write(out / 'definitions.json', definitions)
    manifest = dict(schema=SCHEMA, status=STATUS, evidence_level='JOINT_ALLELE_FEASIBILITY',
        chromosome=contract['chromosome'], source_loci=loci, output_rows=rows,
        contains_sample_identifiers=False, contains_individual_genotypes=False, contains_variant_rows=True,
        source_test_opened=False, genotypes_recomputed=False, simulations_executed=False,
        phase_validated=False, simulation_ready=False,
        source_contract_sha256=before[str(Path(contract_path))][2],
        role_disjointness_sha256=contract['role_disjointness_sha256'],
        source_roles={r: dict(role_members=manifests[r]['ancestry_strata'],
           sample_members_sha256=manifests[r]['genotype_contract']['panel_sample_members_sha256'],
           source_manifest_sha256=contract['roles'][r]['source_contract']['sha256']['manifest.json'],
           quality=manifests[r]['genotype_contract']['quality'], review_sha256=contract['roles'][r]['review_sha256']['manifest.json']) for r in ROLES},
        source_catalogue_sha256=manifests[ROLES[0]]['input_files']['rare_vcf']['sha256'],
        map_sha256=mapped['map_sha256'], map_first_bp=mapped['first_bp'], map_last_bp=mapped['last_bp'],
        code_sha256={Path(__file__).name:sha(__file__), 'r02_lai_support_review.py':sha(support.__file__)},
        outputs_sha256={f.name:sha(f) for f in out.iterdir()},
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'bundle', 'outdir'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--min-free-gib', type=float, default=12)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--max-output-bytes', type=int, default=2*1024**3)
    a = parser.parse_args()
    run(a.contract, a.bundle, a.outdir, a.min_free_gib, a.timeout_seconds, a.max_output_bytes)


if __name__ == '__main__':
    os.umask(0o077)
    main()
