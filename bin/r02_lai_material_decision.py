#!/usr/bin/env python3
"""Decide conditional simulation-material support from sealed L1 joint products.

This stage never decodes genotypes, reassigns donors or opens TEST genotypes.
Candidates mean observed, inferred haplotypes under explicit limitations, not
validated phase, quality, population representativeness or permission to simulate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time

import r02_lai_simulation_material as joint
import r02_lai_support_review as support

require, sha, read, write = support.require, support.sha, support.read, support.write
SCHEMA = 'r02_lai_material_decision_v1'
STATUS = 'COMPLETE_MATERIAL_DECISION_CONDITIONAL_NOT_SIMULATION_READY'
AGG_SCHEMA = 'r02_lai_material_decision_aggregate_v1'
AGG_STATUS = 'COMPLETE_AUTOSOMAL_MATERIAL_DECISION_CONDITIONAL_NOT_SIMULATION_READY'
PARTIAL_STATUS = 'COMPLETE_PARTIAL_MATERIAL_DECISION_CONDITIONAL_NOT_SIMULATION_READY'
ROLES = joint.ROLES
FIELDS = (*joint.KEY, *joint.SOURCE, 'panel_ancestry', 'map_range_status', 'joint_observation',
          'material_candidacy', *[r+'_'+k for r in ROLES for k in
          ('observation_state', *joint.STATE_FIELDS, *joint.COUNTS)])
DECISION_FIELDS = ('decision', 'reference_scenario', 'quality_knowledge', 'phase_knowledge')
FILES = {'candidate_alleles.tsv.gz', 'decision_summary.tsv', 'role_support.json', 'definitions.json'}
SUMMARY_FIELDS = ('chromosome', 'panel_ancestry', 'category', 'loci', 'source_loci_denominator',
                  'fraction', 'reference_members', 'donor_members')
SKIP = b'\tDONOR_NO_RECORD\tNO_OBSERVED_DONOR_COPY_OR_UNKNOWN\tNO_RECORD\tNO_RECORD\t'


class Inputs:
    """Authenticate exact bytes before use and again before sealing outputs."""
    def __init__(self): self.entries = {}

    def add(self, path, digest):
        p = Path(path).resolve(strict=True)
        require(re.fullmatch('[0-9a-f]{64}', digest or '') is not None, 'Invalid SHA256')
        require(p.is_file() and sha(p) == digest, 'Input SHA256 mismatch: '+p.name)
        stat = p.stat(); value = (stat.st_size, stat.st_mtime_ns, digest)
        require(str(p) not in self.entries or self.entries[str(p)] == value, 'Conflicting input binding')
        self.entries[str(p)] = value
        return p

    def descriptor(self, item):
        require(set(item) == {'path', 'sha256'}, 'Expected exact path/SHA256 descriptor')
        return self.add(item['path'], item['sha256'])

    def unchanged(self):
        for name, (size, mtime, digest) in self.entries.items():
            p = Path(name); stat = p.stat()
            require((stat.st_size, stat.st_mtime_ns) == (size, mtime) and sha(p) == digest,
                    'Input changed during decision: '+p.name)


def guard(out, started, reserve, timeout, maximum):
    support.check_space(out, reserve)
    require(time.monotonic()-started <= timeout, 'Decision deadline exceeded')
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= maximum,
            'Decision output budget exceeded')


def role_metadata(path, sources):
    proof = joint.audit_role_disjointness(path, sources)
    counts = {r: defaultdict(lambda: {'members': 0, 'units': set(), 'populations': set(),
                                    'missing_unit_members': 0}) for r in ROLES}
    with path.open() as h:
        reader = csv.DictReader(h, delimiter='\t')
        for row in reader:
            if row['role'] not in ROLES: continue
            for ancestry in ('ALL', row['ancestry']):
                c = counts[row['role']][ancestry]; c['members'] += 1
                if row.get('atomic_unit_id'): c['units'].add(row['atomic_unit_id'])
                else: c['missing_unit_members'] += 1
                if row.get('canonical_population'): c['populations'].add(row['canonical_population'])
    result = {}
    for role in ROLES:
        result[role] = {}
        for ancestry, n in sources[role]['ancestry_strata'].items():
            c = counts[role][ancestry]
            require(c['members'] == n, 'Role metadata denominator differs')
            result[role][ancestry] = dict(distinct_members=n, declared_atomic_units=len(c['units']),
                declared_populations=len(c['populations']), members_without_unit=c['missing_unit_members'],
                units_are_certified_independent=False, distinct_carrier_units='NOT_IDENTIFIABLE_FROM_AGGREGATED_COUNTS')
    return dict(disjointness=proof, strata=result,
        dependency_interpretation='Historical declared atomic groups; not effective sample size or certified independent families. Carrier identities are not in L1 aggregates.')


def projected_header(manifest, inputs):
    entry = manifest['input_files']['panel_vcf']
    p = inputs.add(entry['path'], entry['sha256'])
    formats, filters, header_bytes, sample_names = set(), set(), 0, None
    with gzip.open(p, 'rt') as h:
        for line in h:
            header_bytes += len(line)
            require(header_bytes < 16*1024**2, 'Unbounded VCF header')
            if line.startswith('##FORMAT=<ID='): formats.add(line.split('ID=', 1)[1].split(',', 1)[0])
            if line.startswith('##FILTER=<ID='): filters.add(line.split('ID=', 1)[1].split(',', 1)[0])
            if line.startswith('#CHROM\t'):
                sample_names = line.rstrip('\n').split('\t')[9:]; break
            require(line.startswith('#'), 'Missing VCF sample header')
    require(sample_names is not None and len(sample_names) == len(set(sample_names)), 'Invalid VCF samples')
    digest = hashlib.sha256(('\n'.join(sorted(sample_names))+'\n').encode()).hexdigest()
    require(digest == manifest['genotype_contract']['panel_sample_members_sha256'] and
            len(sample_names) == manifest['ancestry_strata']['ALL'], 'Projection is not the authenticated role')
    require('GT' in formats, 'Projected source lacks declared GT')
    return dict(vcf_sha256=entry['sha256'], declared_format_ids=sorted(formats), declared_filter_ids=sorted(filters),
        dp_declared='DP' in formats, gq_declared='GQ' in formats, ft_declared='FT' in formats,
        ps_declared='PS' in formats, pq_declared='PQ' in formats,
        fields_declared_not_measured_quality=True, genotype_rows_decoded=0,
        site_filter_policy=manifest['genotype_contract']['site_filter_policy'],
        unfiltered_records_are_not_certified_pass=True)


def authenticate(cp, digest):
    inputs = Inputs(); cp = inputs.add(cp, digest); contract = read(cp)
    require(contract.get('schema') == 'r02_lai_material_decision_input_v1', 'Invalid decision input schema')
    chrom = contract.get('chromosome')
    require(chrom in {str(i) for i in range(1, 23)}, 'Invalid chromosome')
    mp = inputs.descriptor(contract['joint_manifest']); manifest = read(mp); root = mp.parent
    require(manifest.get('schema') == joint.SCHEMA and manifest.get('status') == joint.STATUS and
            manifest.get('chromosome') == chrom and manifest.get('source_test_opened') is False and
            manifest.get('genotypes_recomputed') is False and manifest.get('simulation_ready') is False,
            'Joint material is not the complete permitted product')
    require(set(manifest['outputs_sha256']) == {'joint_material.tsv.gz', 'joint_summary.tsv', 'definitions.json'},
            'Incomplete joint output inventory')
    for name, value in manifest['outputs_sha256'].items(): inputs.add(root/name, value)
    rp = inputs.descriptor(contract['publication_receipt']); receipt = read(rp)
    require(receipt.get('schema') == 'r02_evidence_publication_v1' and receipt.get('status') == 'PUBLISHED_VERIFIED' and
            receipt.get('manifest_sha256') == contract['joint_manifest']['sha256'] and
            receipt.get('evidence_schema') == joint.SCHEMA and receipt.get('evidence_status') == joint.STATUS,
            'Joint publication receipt differs')
    expected = dict(manifest['outputs_sha256'], **{'manifest.json': contract['joint_manifest']['sha256']})
    require(len(receipt.get('files', [])) == len(expected), 'Incomplete publication receipt')
    seen = set()
    for item in receipt['files']:
        name = item.get('relative_path', item.get('name', ''))
        require(name in expected and name not in seen and item['sha256'] == expected[name] and
                str(item.get('generation', '')).isdigit(), 'Publication file binding differs')
        require(int(item['bytes']) == (root/name).stat().st_size, 'Publication size differs'); seen.add(name)
    source_cp = inputs.descriptor(contract['material_input_contract']); source_contract = read(source_cp)
    require(source_contract.get('schema') == 'r02_lai_simulation_material_input_v1' and
            source_contract.get('chromosome') == chrom and set(source_contract['roles']) == set(ROLES) and
            contract['material_input_contract']['sha256'] == manifest['source_contract_sha256'],
            'Source material contract differs')
    sources, maps = {}, []
    for role in ROLES:
        item = source_contract['roles'][role]; source = source_cp.parent/role/'source'
        require(item['source_contract']['role'] == role and item['source_contract']['chromosome'] == chrom,
                'Source role/chromosome mismatch')
        p = inputs.add(source/'manifest.json', item['source_contract']['sha256']['manifest.json']); m = read(p)
        require(m.get('schema') == 'r02_lai_genotype_support_v1' and
                m.get('status') == 'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION' and
                m['chromosome'] == chrom and m['genotype_contract']['panel_role'] == role and
                m['build_status'] == 'BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE' and
                m['donor_phase_support'] == 'OBSERVED_PHASE_ONLY_NOT_CERTIFIED', 'Invalid source genotype evidence')
        require(manifest['source_roles'][role] == dict(role_members=m['ancestry_strata'],
                sample_members_sha256=m['genotype_contract']['panel_sample_members_sha256'],
                source_manifest_sha256=item['source_contract']['sha256']['manifest.json'],
                quality=m['genotype_contract']['quality'], review_sha256=item['review_sha256']['manifest.json']),
                'Joint/source role binding differs')
        require(m['counts']['source_rows'] == manifest['source_loci'] and
                m['input_files']['rare_vcf']['sha256'] == manifest['source_catalogue_sha256'], 'Source catalogue differs')
        for name, value in item['review_sha256'].items(): inputs.add(source_cp.parent/role/'review'/name, value)
        review = read(source_cp.parent/role/'review/manifest.json')
        require(review.get('schema') == support.SCHEMA and review.get('status') == support.STATUS and
                review['source_manifest_sha256'] == item['source_contract']['sha256']['manifest.json'], 'Source review differs')
        mapped = read(inputs.add(source/'map_support.json', item['source_contract']['sha256']['map_support.json']))
        require(mapped.get('schema') == 'r02_lai_phased_map_support_v1' and mapped.get('extrapolation') is False and
                mapped['chromosome'] == chrom and mapped['map_sha256'] == manifest['map_sha256'] and
                (mapped['first_bp'], mapped['last_bp']) == (manifest['map_first_bp'], manifest['map_last_bp']), 'Map differs')
        sources[role] = m; maps.append(mapped)
    for key in ('source_sites_inside', 'source_sites_outside'):
        require(maps[0]['counts'].get(key, 0) == maps[1]['counts'].get(key, 0), 'Role map coverage differs')
    roles_path = inputs.descriptor(contract['roles'])
    role_info = role_metadata(roles_path, sources)
    proof_path = inputs.add(source_cp.parent/'role_disjointness.json', source_contract['role_disjointness_sha256'])
    proof = read(proof_path)
    for k in ('roles_sha256', 'role_members_sha256', 'role_counts', 'overlap_individuals'):
        require(proof.get(k) == role_info['disjointness'][k], 'Historical disjointness receipt differs')
    headers = {r: projected_header(sources[r], inputs) for r in ROLES}
    return inputs, contract, manifest, root, role_info, headers


def decision(ref, donor, inside):
    """Do not require a carrier in references for every donor scenario."""
    rs, ds = joint.state(ref), joint.state(donor)
    n = int(donor['n_role_members'])
    if n == 0: label = 'NO_DONORS_SAMPLED_IN_STRATUM'
    elif ds != 'OBSERVED_COPIES': label = 'DONOR_'+ds
    elif not inside: label = 'BLOCKED_MAP_OUTSIDE_ENDPOINTS'
    elif int(donor['n_site_filter_failed']): label = 'BLOCKED_RECORDED_SITE_FILTER_FAILURE'
    elif int(donor['n_phased_carriers_called']) == 0: label = 'BLOCKED_COPY_PHASE_NOT_OBSERVED'
    else: label = 'CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'
    scenario = 'NO_REFERENCES_SAMPLED_IN_STRATUM' if int(ref['n_role_members']) == 0 else 'REFERENCE_'+rs
    quality = ('QUALITY_UNAVAILABLE_NO_COMPATIBLE_RECORD' if donor['genotype_evidence_status'] != 'GT_READ'
               else 'RAW_GT_ONLY_QUALITY_NOT_MEASURED' if donor['quality_status'] == 'NOT_EVALUATED_GT_ONLY'
               else 'DECLARED_QUALITY_AUDIT_NOT_GLOBAL_VALIDATION')
    return label, scenario, quality, 'OBSERVED_FLAGS_NOT_PHASE_ACCURACY'


def write_summary(path, chromosome, counts, loci, role_info):
    with path.open('x') as h:
        w = csv.writer(h, delimiter='\t', lineterminator='\n'); w.writerow(SUMMARY_FIELDS)
        for ancestry in sorted(counts):
            for category, number in sorted(counts[ancestry].items()):
                w.writerow((chromosome, ancestry, category, number, loci, number/loci if loci else 'NA',
                    role_info['strata']['REF_TRAIN'][ancestry]['distinct_members'],
                    role_info['strata']['SOURCE_VALID'][ancestry]['distinct_members']))


def stream(root, manifest, role_info, out, limits):
    strata = list(manifest['source_roles']['REF_TRAIN']['role_members'])
    require(set(strata) == set(manifest['source_roles']['SOURCE_VALID']['role_members']), 'Ancestry sets differ')
    source_loci = manifest['source_loci']; counts = {a: Counter() for a in strata}
    old = {a: Counter() for a in strata}; seen = Counter(); skipped = Counter(); rows = selected = 0
    previous = dict.fromkeys(strata, 0); group_key = None; group_strata = set(); last_position = 0
    with gzip.open(root/'joint_material.tsv.gz', 'rb') as h, gzip.open(out/'candidate_alleles.tsv.gz', 'wt', compresslevel=1) as dest:
        require(tuple(h.readline().decode().rstrip('\n').split('\t')) == FIELDS, 'Joint columns changed')
        w = csv.writer(dest, delimiter='\t', lineterminator='\n'); w.writerow((*FIELDS, *DECISION_FIELDS))
        for line in h:
            rows += 1
            prefix = line.split(b'\t', 16)
            require(len(prefix) == 17 and prefix[0].decode() == manifest['chromosome'], 'Malformed row/chromosome')
            ancestry = prefix[15].decode(); key = tuple(prefix[:5])
            require(ancestry in strata, 'Unknown ancestry')
            if key != group_key:
                require(group_key is None or group_strata == set(strata), 'Allele/ancestry group incomplete')
                position = int(prefix[1]); require(position > last_position, 'Repeated/unsorted source allele')
                group_key, group_strata, last_position = key, set(), position
            require(ancestry not in group_strata, 'Duplicate ancestry at source allele'); group_strata.add(ancestry)
            # The sealed dense producer already validated both roles per exact allele.
            # Only jointly unmeasured rows are omitted from the sparse output, not from denominators.
            if SKIP in line:
                skipped[ancestry] += 1
            else:
                values = line.decode().rstrip('\n').split('\t')
                require(len(values) == len(FIELDS), 'Malformed joint row')
                row = dict(zip(FIELDS, values)); ancestry = row['panel_ancestry']
                require(ancestry in strata and row['chromosome'] == manifest['chromosome'], 'Row scope differs')
                pos = int(row['position_bp']); require(pos > previous[ancestry], 'Repeated or unsorted informative locus')
                previous[ancestry] = pos; seen[ancestry] += 1
                require(row['source_catalogue_sha256'] == manifest['source_catalogue_sha256'], 'Row catalogue differs')
                inside = manifest['map_first_bp'] <= pos <= manifest['map_last_bp']
                require(row['map_range_status'] == ('INSIDE' if inside else 'OUTSIDE'), 'Row map differs')
                ref, donor = ({k: row[r+'_'+k] for k in (*joint.STATE_FIELDS, *joint.COUNTS)} for r in ROLES)
                for role, data in zip(ROLES, (ref, donor)):
                    require(int(data['n_role_members']) == manifest['source_roles'][role]['role_members'][ancestry], 'Role denominator differs')
                    require(joint.state(data) == row[role+'_observation_state'], 'Stored observation differs')
                expected_joint = joint.joint_class(joint.state(ref), joint.state(donor))
                expected_candidate = joint.candidacy(donor, joint.state(donor), inside)
                require(row['joint_observation'] == expected_joint and row['material_candidacy'] == expected_candidate,
                        'Stored joint classification differs')
                old[ancestry]['joint:'+expected_joint] += 1; old[ancestry]['candidate:'+expected_candidate] += 1
                result = decision(ref, donor, inside)
                counts[ancestry]['decision:'+result[0]] += 1
                if result[0] == 'CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE':
                    counts[ancestry]['candidate_reference:'+result[1]] += 1
                    if int(donor['n_gt_missing']) or int(donor['n_gt_non_diploid']):
                        counts[ancestry]['candidate_with_incomplete_donor_calls'] += 1
                    if int(donor['n_phased_carriers_called']) < int(donor['n_carriers_called']):
                        counts[ancestry]['candidate_with_some_unphased_carriers'] += 1
                w.writerow((*values, *result)); selected += 1
            if rows % 10000 == 0: guard(out, *limits)
    require(group_strata == set(strata), 'Incomplete final ancestry group')
    require(rows == manifest['output_rows'] == source_loci*len(strata), 'Joint row denominator differs')
    for ancestry in strata:
        require(seen[ancestry]+skipped[ancestry] == source_loci, 'Incomplete ancestry catalogue')
        old[ancestry]['joint:DONOR_NO_RECORD'] += skipped[ancestry]
        old[ancestry]['candidate:NO_OBSERVED_DONOR_COPY_OR_UNKNOWN'] += skipped[ancestry]
        label = 'NO_DONORS_SAMPLED_IN_STRATUM' if not role_info['strata']['SOURCE_VALID'][ancestry]['distinct_members'] else 'DONOR_NO_RECORD'
        counts[ancestry]['decision:'+label] += skipped[ancestry]
        counts[ancestry]['informative_sparse_rows'] = seen[ancestry]
        counts[ancestry]['jointly_no_record_omitted_rows'] = skipped[ancestry]
    with (root/'joint_summary.tsv').open() as h:
        found = set()
        for row in csv.DictReader(h, delimiter='\t'):
            a, category = row['panel_ancestry'], row['category']
            require(a in strata and (a, category) not in found and row['chromosome'] == manifest['chromosome'] and
                    int(row['source_loci_denominator']) == source_loci, 'Invalid source summary')
            found.add((a, category))
            if category.startswith(('joint:', 'candidate:')):
                require(old[a][category] == int(row['loci']), 'Decision stream disagrees with sealed summary')
                old[a].pop(category)
        require(all(not any(c.values()) for c in old.values()), 'Missing source summary category')
    write_summary(out/'decision_summary.tsv', manifest['chromosome'], counts, source_loci, role_info)
    return selected, counts


def definitions():
    return dict(
        candidate='At least one observed phased donor carrier, map endpoint coverage and no recorded site-filter failure. Conditional use of inferred haplotypes; not phase accuracy or QC certification.',
        sparse_catalogue='All informative allele/ancestry rows retained, including unresolved allele, zero observed and missing GT. Rows with NO_RECORD in BOTH roles omitted only from this sparse file; their counts remain, and the full authenticated joint table is the source of every omitted position.',
        reference='Reference copies are reported, not required universally. Donor-positive/reference-zero defines an unrepresented-reference scenario, not failure or evidence of ancestral specificity.',
        quality='GT_ONLY means no genotype-quality filtering/validation was measured. PASS_OR_UNFILTERED accepts missing site FILTER; not all sites are verified PASS. Declared DP/GQ/FT availability is not measured genotype quality.',
        phase='Observed | and counts are inferred-phase evidence only. No arbitrary assignment of an unphased rare heterozygote to a haplotype. A candidate with some phased carriers does not permit copying its unphased carriers.',
        map='Inside authenticated endpoint range only; no extrapolation and no ancestry-specific map-accuracy claim.',
        zero='Zero with AN>0 is observed absence among called role members, never population absence. Partial-call zero, all-GT-unknown and NO_RECORD remain separate.',
        independence='Disjoint role individuals checked from private metadata. Declared atomic groups are not certified independent donors. Per-allele carrier-unit support cannot be recovered from aggregate counts.',
        scope='SOURCE_VALID is historically exposed development material, not unseen confirmatory TEST. No SOURCE_TEST genotypes opened.',
        summary='decision:* partitions source loci per ancestry. Other categories overlap that partition. ALL overlaps ancestry strata; never add ALL to ancestries. Distinct role members are not summed over chromosomes.',
        next_conditions=['Choose and document a simulation design with donor-unit separation using private carrier/haplotype identity; aggregated allele counts cannot enforce it.',
                         'Specify missing-genotype and phase-uncertainty handling for inferred haplotypes; phase uncertainty can be studied conditionally, not assumed absent.',
                         'Record a QC policy supported by actual available fields and external evidence; do not invent DP/GQ thresholds for GT_ONLY data.',
                         'Use only positions within the authenticated map and preserve reference-unrepresented scenarios; never extrapolate.'],
        conclusion='Nonzero candidates establish material feasibility under stated assumptions, not simulated or validated LAI performance. Zero candidates in a stratum establish only lack of observed usable support under these rules.')


def run(contract_path, contract_sha256, outdir, reserve=12, timeout=7200, maximum=128*1024**2):
    started = time.monotonic(); out = Path(outdir)
    require(reserve >= 0 and timeout > 0 and maximum > 0, 'Invalid resource limits')
    require(not out.exists(), 'Output exists; use Nextflow resume')
    inputs, contract, manifest, root, role_info, headers = authenticate(contract_path, contract_sha256)
    out.mkdir(parents=True, mode=0o700); guard(out, started, reserve, timeout, maximum)
    selected, counts = stream(root, manifest, role_info, out, (started, reserve, timeout, maximum))
    role_info['headers'] = headers; write(out/'role_support.json', role_info); write(out/'definitions.json', definitions())
    inputs.unchanged(); guard(out, started, reserve, timeout, maximum)
    result = dict(schema=SCHEMA, status=STATUS, chromosome=contract['chromosome'], source_loci=manifest['source_loci'],
        source_rows=manifest['output_rows'], sparse_rows=selected,
        conditional_candidate_loci={a: c['decision:CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'] for a,c in counts.items()},
        role_members={r: manifest['source_roles'][r]['role_members'] for r in ROLES},
        role_members_sha256={r: manifest['source_roles'][r]['sample_members_sha256'] for r in ROLES},
        roles_sha256=contract['roles']['sha256'], source_catalogue_sha256=manifest['source_catalogue_sha256'],
        source_contract_sha256=contract_sha256, joint_manifest_sha256=contract['joint_manifest']['sha256'],
        map_sha256=manifest['map_sha256'], map_first_bp=manifest['map_first_bp'], map_last_bp=manifest['map_last_bp'],
        quality_modes={r: manifest['source_roles'][r]['quality']['mode'] for r in ROLES},
        contains_sample_identifiers=False, contains_individual_genotypes=False, contains_variant_rows=True,
        source_test_opened=False, simulations_executed=False, genotypes_recomputed=False,
        simulation_ready=False, phase_validated=False, effective_independent_donors='UNKNOWN',
        public_distribution_allowed=False, publication_verification='Authenticated prior PUBLISHED_VERIFIED receipt; no new remote claim',
        code_sha256={Path(p).name:sha(p) for p in (__file__,joint.__file__,support.__file__)},
        inputs_sha256={p:v[2] for p,v in inputs.entries.items()},
        outputs_sha256={n:sha(out/n) for n in sorted(FILES)}, elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'manifest.json', result); guard(out, started, reserve, timeout, maximum)
    return result


def aggregate(campaign_path, campaign_sha256, manifest_paths, manifest_hashes, outdir, reserve=12, timeout=7200, maximum=128*1024**2):
    started = time.monotonic(); out = Path(outdir); inputs = Inputs()
    cp = inputs.add(campaign_path, campaign_sha256); campaign = read(cp)
    require(campaign.get('schema') == 'r02_lai_material_decision_campaign_v1', 'Invalid campaign schema')
    expected = campaign['expected_chromosomes']
    require(expected and len(expected) == len(set(expected)) and set(expected) <= {str(i) for i in range(1,23)}, 'Invalid expected chromosomes')
    jobs = campaign['jobs']; require(len(jobs) == len(expected) and {j['chromosome'] for j in jobs} == set(expected), 'Incomplete campaign jobs')
    contracts = {j['chromosome']:j['contract']['sha256'] for j in jobs}
    for j in jobs: inputs.descriptor(j['contract'])
    completion = read(inputs.descriptor(campaign['completion']))
    require(completion.get('status') == 'COMPLETE_22_JOINT_CHROMOSOMES_NOT_SIMULATION_READY', 'Upstream autosomes not complete')
    products = completion['products']
    require(len(products) == 22 and {p['chromosome'] for p in products} == {str(i) for i in range(1,23)}, 'Incomplete upstream 22 products')
    upstream = {p['chromosome']:p['manifest_sha256'] for p in products}
    require(len(manifest_paths) == len(manifest_hashes) == len(expected), 'Incomplete decision receipts')
    require(not out.exists(), 'Output exists; use Nextflow resume')
    manifests, totals, rows, knowledge, catalogue_index = {}, defaultdict(Counter), [], [], []
    shared, role_info = None, None
    for path, digest in zip(manifest_paths, manifest_hashes):
        mp = inputs.add(path, digest); m = read(mp); chrom = m.get('chromosome')
        require(m.get('schema') == SCHEMA and m.get('status') == STATUS and chrom in expected and chrom not in manifests,
                'Duplicate/incomplete decision chromosome')
        require(m['source_contract_sha256'] == contracts[chrom] and m['joint_manifest_sha256'] == upstream[chrom], 'Decision source chain differs')
        require(set(m['outputs_sha256']) == FILES and m['simulation_ready'] is False and m['source_test_opened'] is False,
                'Invalid decision output contract')
        for name, value in m['outputs_sha256'].items(): inputs.add(mp.parent/name, value)
        signature = (m['roles_sha256'], m['role_members_sha256'], m['role_members'], m['quality_modes'], m['code_sha256'])
        require(shared is None or signature == shared, 'Different cohorts/roles/code/quality across chromosomes')
        shared = signature; manifests[chrom] = m
        ri = read(mp.parent/'role_support.json')
        catalogue_index.append(dict(chromosome=chrom, candidate_table=str((mp.parent/'candidate_alleles.tsv.gz').resolve()),
            sha256=m['outputs_sha256']['candidate_alleles.tsv.gz'], sparse_rows=m['sparse_rows'],
            source_loci=m['source_loci'], joint_manifest_sha256=m['joint_manifest_sha256']))
        for role in ROLES:
            header=ri['headers'][role]
            knowledge.append(dict(chromosome=chrom, panel_role=role, quality_mode=m['quality_modes'][role],
                declared_format_ids=','.join(header['declared_format_ids']), dp_declared=header['dp_declared'],
                gq_declared=header['gq_declared'], ft_declared=header['ft_declared'], ps_declared=header['ps_declared'],
                pq_declared=header['pq_declared'], site_filter_policy=header['site_filter_policy'],
                phase_accuracy='NOT_MEASURED', map_first_bp=m['map_first_bp'], map_last_bp=m['map_last_bp'],
                map_sha256=m['map_sha256'], projection_vcf_sha256=header['vcf_sha256']))
        base = {k:v for k,v in ri.items() if k != 'headers'}
        require(role_info is None or base == role_info, 'Role dependency metadata differ across chromosomes'); role_info = base
        with (mp.parent/'decision_summary.tsv').open() as h:
            reader = csv.DictReader(h, delimiter='\t'); require(tuple(reader.fieldnames) == SUMMARY_FIELDS, 'Summary columns differ')
            seen = set(); local = defaultdict(Counter)
            for row in reader:
                a, category = row['panel_ancestry'], row['category']
                require((a, category) not in seen and row['chromosome'] == chrom and
                        int(row['source_loci_denominator']) == m['source_loci'], 'Summary denominator/scope differs')
                require(a in m['role_members']['REF_TRAIN'] and
                        int(row['reference_members']) == m['role_members']['REF_TRAIN'][a] and
                        int(row['donor_members']) == m['role_members']['SOURCE_VALID'][a], 'Summary role denominator differs')
                number = int(row['loci']); require(0 <= number <= m['source_loci'], 'Invalid summary count')
                require(abs(float(row['fraction'])-number/m['source_loci']) <= 1e-12, 'Summary fraction differs')
                local[a][category] += number; totals[a][category] += number; seen.add((a,category)); rows.append(row)
            require(set(local) == set(m['role_members']['REF_TRAIN']), 'Incomplete summary ancestries')
            for a,c in local.items():
                require(sum(v for k,v in c.items() if k.startswith('decision:')) == m['source_loci'] and
                        c['decision:CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'] == m['conditional_candidate_loci'][a],
                        'Decision partition or candidate count differs')
    out.mkdir(parents=True, mode=0o700); guard(out, started, reserve, timeout, maximum)
    loci = sum(m['source_loci'] for m in manifests.values())
    write_summary(out/'decision_summary.tsv', 'AUTOSOMES' if len(expected)==22 else 'SELECTED_CHROMOSOMES', totals, loci, role_info)
    with (out/'chromosome_summary.tsv').open('x') as h:
        w=csv.DictWriter(h, fieldnames=SUMMARY_FIELDS, delimiter='\t', lineterminator='\n'); w.writeheader(); w.writerows(rows)
    for name, table in (('source_knowledge.tsv',knowledge),('candidate_catalogue_index.tsv',catalogue_index)):
        with (out/name).open('x') as h:
            w=csv.DictWriter(h,fieldnames=list(table[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(table)
    write(out/'role_support.json', role_info); write(out/'definitions.json', definitions())
    write(out/'decision.json', dict(
        material_conclusion='CONDITIONAL_MATERIAL_EXISTS' if totals['ALL']['decision:CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'] else 'NO_OBSERVED_CONDITIONAL_CANDIDATES',
        conditional_candidate_loci={a:c['decision:CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'] for a,c in totals.items()},
        source_loci_denominator=loci, candidate_fractions={a:c['decision:CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE']/loci for a,c in totals.items()},
        simulation_ready=False, interpretation='Candidate count, not permission to simulate or validated LAI. Lack of phase accuracy certification alone does not erase observed candidate haplotypes.',
        next_conditions=definitions()['next_conditions'], carrier_unit_support='NOT_IDENTIFIABLE_FROM_AGGREGATED_COUNTS'))
    inputs.unchanged(); guard(out, started, reserve, timeout, maximum)
    result = dict(schema=AGG_SCHEMA, status=AGG_STATUS if len(expected)==22 else PARTIAL_STATUS,
        chromosomes=sorted(expected,key=int), source_loci=loci, role_members=shared[2],
        role_members_sha256=shared[1], roles_sha256=shared[0],
        conditional_candidate_loci={a:c['decision:CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'] for a,c in totals.items()},
        source_campaign_sha256=campaign_sha256, source_decision_sha256=dict(zip((read(p)['chromosome'] for p in manifest_paths),manifest_hashes)),
        contains_sample_identifiers=False, contains_individual_genotypes=False, contains_variant_rows=False,
        source_test_opened=False, genotypes_recomputed=False, simulations_executed=False, simulation_ready=False,
        phase_validated=False, public_distribution_allowed=False,
        code_sha256={Path(p).name:sha(p) for p in (__file__,joint.__file__,support.__file__)},
        outputs_sha256={p.name:sha(p) for p in out.iterdir()}, elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'manifest.json', result); guard(out, started, reserve, timeout, maximum)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    one=subs.add_parser('chromosome'); one.add_argument('--contract',required=True); one.add_argument('--contract-sha256',required=True)
    allp=subs.add_parser('aggregate'); allp.add_argument('--campaign-contract',required=True); allp.add_argument('--campaign-contract-sha256',required=True)
    allp.add_argument('--decision-manifest',action='append',required=True); allp.add_argument('--decision-sha256',action='append',required=True)
    for p in (one,allp):
        p.add_argument('--output-dir',required=True); p.add_argument('--min-free-gib',type=float,default=12)
        p.add_argument('--timeout-seconds',type=int,default=7200); p.add_argument('--max-output-bytes',type=int,default=128*1024**2)
    a=parser.parse_args(); limits=(a.min_free_gib,a.timeout_seconds,a.max_output_bytes)
    if a.command=='chromosome': run(a.contract,a.contract_sha256,a.output_dir,*limits)
    else: aggregate(a.campaign_contract,a.campaign_contract_sha256,a.decision_manifest,a.decision_sha256,a.output_dir,*limits)


if __name__=='__main__':
    os.umask(0o077); main()
