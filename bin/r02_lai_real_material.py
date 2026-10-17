#!/usr/bin/env python3
"""Prepare exact parental role projections, retaining common and rare context.

No donors, demographic scenarios or splits are selected here. Existing role
projections and assembly receipts are authenticated; only the explicit site/map
policy removes records. Version 2 permits one declared autosomal contig alias;
only CHROM and its ##contig ID change, never coordinates, alleles or GT/INFO.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from itertools import zip_longest
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import time

import r02_lai_role_subset as subset
import r02_lai_simulate as sim
import r02_structure_regions as maps
from r02_lai_support_review import require, read, write, sha

SCHEMA = 'r02_lai_real_material_input_v1'
SCHEMA_V2 = 'r02_lai_real_material_input_v2'
OUTPUT_SCHEMA = 'r02_lai_real_material_v1'
STATUS = 'COMPLETE_PARENTAL_MATERIAL_NOT_SIMULATION_AUTHORIZATION'
ROLES = ('SOURCE_VALID', 'REF_TRAIN')
CODE_FILES = ('r02_lai_real_material.py', 'r02_lai_role_subset.py', 'r02_lai_simulate.py',
              'r02_lai_support_review.py', 'r02_structure_regions.py')
GT_PATTERNS = {False: re.compile(r'[01]\|[01](?:\t[01]\|[01])*'),
               True: re.compile(r'(?:[01]\|[01]|0/0|1/1)(?:\t(?:[01]\|[01]|0/0|1/1))*')}


def contig_policy(contract):
    """An alias is a declared identity, not an inferred assembly conversion."""
    chromosome = contract['chromosome']
    if contract['schema'] == SCHEMA:
        require('contig_policy' not in contract, 'Legacy contract cannot authorize contig renaming')
        return dict(mode='EXACT_IDENTITY', output_contig=chromosome,
                    source_contigs={role:chromosome for role in ROLES})
    require(contract['schema'] == SCHEMA_V2, 'Unknown material schema')
    policy = contract.get('contig_policy')
    require(isinstance(policy,dict) and set(policy)=={'mode','source_contigs','output_contig'},
            'Explicit complete contig policy required')
    require(policy['mode']=='EXPLICIT_AUTOSOME_ALIAS' and policy['output_contig']==chromosome,
            'Output contig must match the declared autosome and its downstream map')
    require(isinstance(policy['source_contigs'],dict) and set(policy['source_contigs'])==set(ROLES),
            'Declare source contig separately for both roles')
    require(all(type(value) is str and value in {chromosome,'chr'+chromosome}
                for value in policy['source_contigs'].values()), 'Only the same autosome N or chrN is allowed')
    # Both roles were projected from one authenticated source marker axis.
    require(len(set(policy['source_contigs'].values()))==1, 'Role contigs differ from shared source axis')
    return dict(mode=policy['mode'],output_contig=policy['output_contig'],
                source_contigs=dict(policy['source_contigs']))


def contig_headers(headers, source_contig, output_contig):
    """Preserve every header byte except the single authenticated contig ID."""
    result=[]; seen=[]
    for line in headers:
        if line.startswith('##contig='):
            match=re.fullmatch(r'##contig=<ID=([^,>]+)([^>]*)>',line)
            require(match is not None and len(re.findall(r'(?:<|,)ID=',line))==1,
                    'Malformed or ambiguous contig declaration')
            seen.append(match.group(1))
            require(match.group(1)==source_contig,'Header contig differs from declared source')
            line='##contig=<ID='+output_contig+match.group(2)+'>'
        result.append(line)
    require(seen==[source_contig],'Exactly one source contig header required')
    return result


def contig_row(line, source_contig, output_contig):
    """Only replace the first field; the remaining VCF text is opaque."""
    observed,separator,payload=line.partition('\t')
    require(separator and observed==source_contig,'Record contig differs from declared source')
    return output_contig+separator+payload


def descriptor(entry):
    path = sim.descriptor(entry)
    require(not path.is_symlink() and path.resolve() == path, 'Scientific source must be canonical, not a symlink')
    return path


def role_rows(path):
    selected = {role: [] for role in ROLES}; seen = set()
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require({'sample_id','role','ancestry','atomic_unit_id','canonical_population'} <= set(reader.fieldnames or []), 'Original role metadata required')
        for row in reader:
            require(row['sample_id'] and row['sample_id'] not in seen, 'Duplicate/empty historical person')
            seen.add(row['sample_id'])
            if row['role'] in selected:
                require(row['atomic_unit_id'] and row['ancestry'] and row['canonical_population'], 'Incomplete original role')
                selected[row['role']].append(row)
    require(all(selected.values()), 'Both authorized roles required')
    units = {role: {r['atomic_unit_id'] for r in rows} for role, rows in selected.items()}
    require(units[ROLES[0]].isdisjoint(units[ROLES[1]]), 'Original unit crosses reference/donor roles')
    return selected


def authenticate(path, digest):
    path = Path(path).resolve()
    c = read(sim.descriptor(dict(path=str(path), sha256=digest)))
    require(c.get('schema') in {SCHEMA,SCHEMA_V2} and c.get('scope') in {'REAL_PARENTAL_PREPARATION','SYNTHETIC_CODE_TEST'}, 'Explicit material scope/schema required')
    require(c.get('genome_build') == ('GRCh38' if c['scope'].startswith('REAL') else 'SYNTHETIC'), 'Assembly/scope mismatch')
    require(c.get('chromosome') in {str(i) for i in range(1,23)}, 'Autosome required')
    contig_policy(c)
    require(c.get('marker_policy') == 'BIALLELIC_SNV_SAME_AXIS_WITHIN_MAP_NO_FREQUENCY_SELECTION', 'Undeclared marker selection')
    policy = sim.input_policy(dict(schema=sim.SCHEMA_V2, input_policy=c['input_policy']))
    require(set(c['sources']) == set(ROLES), 'Only complete SOURCE_VALID and REF_TRAIN projections are allowed')
    limits = c['resources']
    require(set(limits) == {'cpus','max_input_bytes','max_output_bytes','max_records','max_samples',
        'max_line_bytes','max_map_points','max_map_bytes','min_free_gib','timeout_seconds','guard_every_records'}, 'Explicit resource limits required')
    require(limits['cpus'] == 3 and all(type(v) is int and v > 0 for k,v in limits.items() if k != 'min_free_gib'), 'Positive integer resource limits / three CPU budget required')
    require(type(limits['min_free_gib']) in (int,float) and math.isfinite(limits['min_free_gib']) and limits['min_free_gib'] >= 12, 'At least12GiB reserve required')
    files = {}; bindings = {str(path): digest}
    def checked(entry):
        p = descriptor(entry); bindings[str(p)] = entry['sha256']; return p
    files['roles'] = checked(c['original_roles']); rows = role_rows(files['roles'])
    files['map'] = checked(c['genetic_map'])
    require(files['map'].stat().st_size <= limits['max_map_bytes'], 'Map input bound exceeded')
    projected = {}; source_axis = set(); original_vcf = set()
    for role in ROLES:
        spec = c['sources'][role]
        require(set(spec) == {'vcf','index','projection_manifest','support_manifest','build_manifest'}, 'Missing source provenance')
        f = {k: checked(v) for k,v in spec.items()}; files[role] = f
        require(str(f['vcf']).endswith('.vcf.gz') and str(f['index']) in {str(f['vcf'])+'.csi',str(f['vcf'])+'.tbi'}, 'Adjacent VCF index required')
        p, s, b = (read(f[k]) for k in ('projection_manifest','support_manifest','build_manifest'))
        require(p.get('schema') == 'r02_lai_role_subset_v1' and p.get('status') == 'COMPLETE_MECHANICAL_SUBSET_NOT_BUILD_QC_OR_PHASE_VALIDATION', 'Incomplete mechanical projection')
        require(p['role'] == role and p['chromosome'] == c['chromosome'] and p['source_test_projected'] is False and p['selected_genotype_text_preserved'] is True, 'Projection scope changed')
        require(p['input_sha256']['roles'] == c['original_roles']['sha256'], 'Projection roles changed')
        require(p['output_sha256'][f['vcf'].name] == spec['vcf']['sha256'] and p['output_sha256'][f['index'].name] == spec['index']['sha256'], 'Projection bytes/index differ')
        expected = [r['sample_id'] for r in rows[role]]
        require(p['selected_samples'] == len(expected) and p['sample_order_sha256'] == subset.axis_hash(expected) and p['sample_members_sha256'] == subset.axis_hash(sorted(expected)), 'Original sample order/membership changed')
        require(s.get('schema') == 'r02_lai_genotype_support_v1' and s.get('status') == 'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION' and s['chromosome'] == c['chromosome'], 'Incomplete L1 support')
        require(s['genotype_contract']['panel_role'] == role and s['input_files']['roles']['sha256'] == c['original_roles']['sha256'] and s['input_files']['panel_vcf']['sha256'] == spec['vcf']['sha256'], 'L1 provenance differs')
        require(s['input_files']['panel_build_verification']['sha256'] == spec['build_manifest']['sha256'], 'L1 assembly receipt differs')
        require(b.get('schema') == 'r02_lai_build_verification_v1' and b.get('status') == 'VERIFIED' and b['build'] == c['genome_build'] and b['vcf_sha256'] == spec['vcf']['sha256'] and b['reference_audit']['scope'] == 'ALL_VCF_RECORDS' and b['reference_audit']['mismatches'] == 0 and b['reference_audit']['records_checked'] == p['records'], 'Exact all-record assembly evidence required')
        require(len(expected) <= limits['max_samples'] and 0 < p['records'] <= limits['max_records'], 'Source dimensions exceed admission')
        projected[role] = p; source_axis.add((p['records'],p['variant_key_order_sha256'])); original_vcf.add(p['input_sha256']['source_vcf'])
    require(len(source_axis) == len(original_vcf) == 1, 'Role projections do not share one source/axis')
    files['donor_audit'] = checked(c['donor_full_audit'])
    audit = read(files['donor_audit'])
    require(audit.get('schema') == 'r02_lai_full_chromosome_audit_v1' and audit.get('status') == 'COMPLETE_PRIVATE_FULL_MARKER_AUDIT_NOT_PHASE_OR_SIMULATION_VALIDATION' and audit['chromosome'] == c['chromosome'] and audit['source_role'] == 'SOURCE_VALID' and audit['source_test_opened'] is False, 'Exact completed donor audit required')
    require({c['sources']['SOURCE_VALID']['vcf']['sha256'], c['original_roles']['sha256'], c['genetic_map']['sha256']} <= set(audit['inputs_sha256'].values()), 'Full audit does not authenticate these donor bytes/roles/map')
    require(audit['summary']['donors'] == len(rows['SOURCE_VALID']) and audit['summary']['markers']['records'] == projected['SOURCE_VALID']['records'], 'Full audit dimensions differ')
    require(sum(Path(p).stat().st_size for p in bindings) <= limits['max_input_bytes'], 'Total input bound exceeded')
    require(c['code_sha256'] == {name:sha(Path(__file__).parent/name) for name in CODE_FILES}, 'Frozen code changed')
    return c,files,rows,projected,bindings,policy


def normalize_map(source, destination, chromosome, max_points, max_bytes):
    # Bound decompressed content BEFORE the existing map reader allocates tuples.
    count = size = 0
    with sim.text_open(source) as stream:
        for line in stream:
            count += 1; size += len(line.encode())
            require(count <= max_points+1 and size <= max_bytes, 'Expanded map budget exceeded')
    points = maps.read_map(source, chromosome, 2**31-1)
    with destination.open('x') as out:
        for bp, cm in points:
            out.write(f'{chromosome}\t{chromosome}:{bp}\t{cm:.17g}\t{bp}\n')
    require(sim.read_map(destination,chromosome) == points, 'PLINK map conversion changed coordinates')
    return points


def validate_calls(row, samples, allow_identical):
    """GT-only C-regex path is tested against the existing per-call validator."""
    require(len(row) == 9+samples, 'VCF width differs')
    if row[8] == 'GT':
        calls = '\t'.join(row[9:])
        require(GT_PATTERNS[allow_identical].fullmatch(calls), 'Noncopyable GT: no imputation, dropping people or new phase allowed')
        het = calls.count('0|1') + calls.count('1|0')
        hom_unphased = calls.count('0/0') + calls.count('1/1')
        return dict(complete_diploid_calls=samples, phased_heterozygous_calls=het,
                    homozygous_calls=samples-het, unphased_homozygous_calls=hom_unphased)
    values = sim.genotypes(row,allow_identical)
    gi = row[8].split(':').index('GT')
    het = sum(a != b for a,b in values)
    return dict(complete_diploid_calls=samples,phased_heterozygous_calls=het,homozygous_calls=samples-het,
                unphased_homozygous_calls=sum(x.split(':')[gi] in {'0/0','1/1'} for x in row[9:]))


def reference_calls(row,samples,allow_identical):
    """Reference unknowns remain present; this is not a FLARE admission gate."""
    try:
        return validate_calls(row,samples,allow_identical)
    except ValueError:
        require(len(row)==9+samples and 'GT' in row[8].split(':'),'Reference width/GT format differs')
        counts=Counter(); gi=row[8].split(':').index('GT')
        for field in row[9:]:
            values=field.split(':'); gt=values[gi] if len(values)>gi else '.'
            alleles=re.split(r'[/|]',gt)
            require(all(x in {'0','1','.',''} for x in alleles),'Reference allele index differs from biallelic marker')
            if any(x=='.' or x=='' for x in alleles):
                counts['missing_or_partial_calls']+=1
                continue
            if len(alleles)!=2:
                counts['non_diploid_calls']+=1
                continue
            require(all(x in {'0','1'} for x in alleles),'Reference allele index differs from biallelic marker')
            counts['complete_diploid_calls']+=1
            if alleles[0]==alleles[1]:
                counts['homozygous_calls']+=1
                if '/' in gt: counts['unphased_homozygous_calls']+=1
            else:
                counts['phased_heterozygous_calls' if '|' in gt else 'unphased_heterozygous_calls']+=1
        return dict(counts)


def vcf_stream(path, limit):
    iterator = iter(subset.lines(path,limit)); headers = []
    for line in iterator:
        require(line.startswith('#'), 'VCF data before sample header')
        headers.append(line)
        if line.startswith('#CHROM\t'):
            return headers,line.split('\t')[9:],iterator
    raise ValueError('VCF sample header absent')


def prepare(path,digest,out):
    started = time.monotonic(); c,files,roles,projected,bindings,policy = authenticate(path,digest)
    contigs = contig_policy(c)
    versions=dict(bcftools=subprocess.check_output(['bcftools','--version'],text=True).splitlines()[0],
                  bgzip=subprocess.check_output(['bgzip','--version'],text=True).splitlines()[0])
    require(c.get('tool_versions')==versions,'Pinned genomic tool versions differ')
    out = Path(out).resolve(); require(out.parent.is_dir() and not out.exists(), 'Fresh output required; use Nextflow resume')
    sim.local_filesystem(out.parent); r = c['resources']
    def guard():
        sim.resource_guard(out if out.exists() else out.parent,dict(timeout_seconds=r['timeout_seconds'],disk_reserve_gib=r['min_free_gib'],max_output_gib=r['max_output_bytes']/2**30),started)
    # Initial guard must not count unrelated files in the parent directory.
    require(sim.shutil.disk_usage(out.parent).free >= r['min_free_gib']*2**30, 'Disk reserve violated')
    out.mkdir(mode=0o700); guard()
    points = normalize_map(files['map'],out/'genetic.map',c['chromosome'],r['max_map_points'],r['max_map_bytes'])
    streams = {}; outputs = {}; compressors = {}; handles = []; hashes = {}; counters = {role:Counter() for role in ROLES}
    payload_hashes = {role:hashlib.sha256() for role in ROLES}
    header_audit = {}
    axis = hashlib.sha256(); source_axis = hashlib.sha256(); n = kept = 0; previous = 0; reasons = Counter()
    first = last = None; last_cm = None; max_gap_bp = 0; max_gap_cm = 0.; count_gaps = 0
    try:
        for role in ROLES:
            headers,samples,iterator = vcf_stream(files[role]['vcf'],r['max_line_bytes'])
            require(samples == [x['sample_id'] for x in roles[role]], 'VCF contains reserved/missing/reordered people')
            source_headers = headers
            headers = contig_headers(headers,contigs['source_contigs'][role],contigs['output_contig'])
            header_audit[role] = dict(source_contig=contigs['source_contigs'][role],output_contig=contigs['output_contig'],
                source_header_sha256=hashlib.sha256(('\n'.join(source_headers)+'\n').encode()).hexdigest(),
                output_header_sha256=hashlib.sha256(('\n'.join(headers)+'\n').encode()).hexdigest(),
                header_changed=source_headers!=headers)
            streams[role] = iterator
            filename = role.lower()+'.chr'+c['chromosome']+'.vcf.gz'
            destination=(out/filename).open('xb'); handles.append(destination)
            error=(out/(role.lower()+'.bgzip.log')).open('xb'); handles.append(error)
            proc=subprocess.Popen(['bgzip','--threads','1','-c'],stdin=subprocess.PIPE,stdout=destination,stderr=error)
            compressors[role]=proc;outputs[role]=proc.stdin; hashes[role] = hashlib.sha256()
            for line in headers:
                data = (line+'\n').encode(); outputs[role].write(data); hashes[role].update(data)
        with gzip.open(out/'marker_territory.tsv.gz','xt') as ledger:
            writer = csv.writer(ledger,delimiter='\t',lineterminator='\n')
            writer.writerow(('source_record','chromosome','position_bp','REF','ALT','source_allele_count','state','genetic_cm','source_contig','output_contig'))
            for donor,reference in zip_longest(streams[ROLES[0]],streams[ROLES[1]]):
                require(donor is not None and reference is not None, 'Unequal role marker counts')
                rows = {ROLES[0]:donor.split('\t'),ROLES[1]:reference.split('\t')}; a,b = rows.values()
                require(len(a)>=10 and len(b)>=10 and a[:9] == b[:9], 'Reference/donor marker/annotation axes differ')
                require(all(rows[role][0]==contigs['source_contigs'][role] for role in ROLES),
                        'Record contig differs from declared source')
                pos = int(a[1]); require(pos > previous, 'Nonunique or unsorted marker positions')
                n += 1; previous = pos; require(n <= r['max_records'], 'Record limit exceeded')
                key = '\t'.join((a[0],a[1],a[3],a[4]))+'\n'; source_axis.update(key.encode())
                allele_count = 1+len(a[4].split(','))
                snv = allele_count == 2 and a[3] in {'A','C','G','T'} and a[4] in {'A','C','G','T'} and a[3] != a[4]
                inside = points[0][0] <= pos <= points[-1][0]
                allowed = a[6] in ({'PASS'} if policy['site_filter']=='PASS_ONLY' else {'PASS','.'})
                excluded = [label for fail,label in ((not snv,'NOT_BIALLELIC_SNV'),(not inside,'OUTSIDE_MAP'),(not allowed,'SITE_FILTER')) if fail]
                state = ';'.join(excluded) if excluded else 'RETAINED'
                cm = maps.interpolate_cm(points,pos) if inside else None
                writer.writerow((n,c['chromosome'],pos,a[3],a[4],allele_count,state,format(cm,'.17g') if cm is not None else 'NA',a[0],contigs['output_contig']))
                reasons[state] += 1
                if not excluded:
                    for role,line in zip(ROLES,(donor,reference)):
                        validator=validate_calls if role=='SOURCE_VALID' else reference_calls
                        counters[role].update(validator(rows[role],len(roles[role]),policy['unphased_homozygotes']=='ALLOW_IDENTICAL_COPIES'))
                        transformed=contig_row(line,contigs['source_contigs'][role],contigs['output_contig'])
                        payload_hashes[role].update((line.partition('\t')[2]+'\n').encode())
                        data=(transformed+'\n').encode(); outputs[role].write(data); hashes[role].update(data)
                    kept += 1
                    axis.update(('\t'.join((contigs['output_contig'],a[1],a[3],a[4]))+'\n').encode())
                    if last is not None:
                        max_gap_bp=max(max_gap_bp,pos-last); max_gap_cm=max(max_gap_cm,cm-last_cm); count_gaps+=1
                    first=pos if first is None else first; last=pos; last_cm=cm
                if n % r['guard_every_records'] == 0: guard()
    finally:
        for handle in outputs.values(): handle.close()
        for proc in compressors.values():
            try: proc.wait(timeout=max(1,int(r['timeout_seconds']-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait(timeout=30);raise
        for handle in handles: handle.close()
        for iterator in streams.values(): iterator.close()
    require(all(proc.returncode==0 for proc in compressors.values()),'BGZF compression failed')
    require(kept > 0 and n == projected[ROLES[0]]['records'] and source_axis.hexdigest() == projected[ROLES[0]]['variant_key_order_sha256'], 'Source axis/count differs from authenticated projection')
    if policy == {'site_filter':'PASS_OR_UNFILTERED','unphased_homozygotes':'ALLOW_IDENTICAL_COPIES'}:
        require(kept == read(files['donor_audit'])['summary']['markers']['eligible_markers'], 'Retained territory differs from complete donor audit')
    index_audit = {}
    for role in ROLES:
        vcf = out/(role.lower()+'.chr'+c['chromosome']+'.vcf.gz')
        observed = hashlib.sha256()
        with gzip.open(vcf,'rb') as stream:
            for data in iter(lambda:stream.read(2**20),b''): observed.update(data)
        require(observed.hexdigest() == hashes[role].hexdigest(), 'BGZF write did not preserve accepted VCF bytes')
        observed_payload=hashlib.sha256(); output_header=[]
        for line in subset.lines(vcf,r['max_line_bytes']):
            if line.startswith('#'): output_header.append(line)
            else:
                contig,separator,payload=line.partition('\t')
                require(separator and contig==contigs['output_contig'],'Output record contig changed')
                observed_payload.update((payload+'\n').encode())
        require(observed_payload.hexdigest()==payload_hashes[role].hexdigest(),
                'Coordinates, alleles, INFO, FORMAT or genotypes changed during alias projection')
        require(hashlib.sha256(('\n'.join(output_header)+'\n').encode()).hexdigest()==header_audit[role]['output_header_sha256'],
                'Output VCF header changed')
        subprocess.run(['bcftools','index','--csi',str(vcf)],check=True,capture_output=True,timeout=max(1,int(r['timeout_seconds']-(time.monotonic()-started))))
        indexed = int(subprocess.check_output(['bcftools','index','-n',str(vcf)],text=True,timeout=60))
        require(indexed == kept, 'Index record count differs')
        require(subset.header(vcf,r['max_line_bytes']) == [x['sample_id'] for x in roles[role]], 'Indexed sample order changed')
        positions=subprocess.check_output(['bcftools','query','-r',f"{c['chromosome']}:{first}-{first}",'-f','%POS\n',str(vcf)],text=True,timeout=60).splitlines()
        require(positions==[str(first)],'Index cannot retrieve first retained marker')
        index_audit[role] = dict(records=indexed,index_accessible=True,accepted_text_sha256=observed.hexdigest(),
            accepted_non_contig_text_sha256=observed_payload.hexdigest(),non_contig_text_preserved=True)
    for name,checksum in bindings.items(): require(sha(name)==checksum,'Source changed during preparation')
    summary = dict(schema=OUTPUT_SCHEMA,status=STATUS,scope=c['scope'],chromosome=c['chromosome'],genome_build=c['genome_build'],
        source_records=n,retained_records=kept,excluded_records=n-kept,territory_counts=dict(reasons),
        source_axis_sha256=source_axis.hexdigest(),retained_axis_sha256=axis.hexdigest(),input_policy=policy,
        contig_policy=contigs,contig_header_audit=header_audit,
        contig_renamed=any(value!=contigs['output_contig'] for value in contigs['source_contigs'].values()),
        role_counts={role:len(rows) for role,rows in roles.items()},calls={role:dict(v) for role,v in counters.items()},
        reference_unknowns_preserved=True,reference_genotype_filtering=False,reference_inference_admission='NOT_ADJUDICATED',
        original_units={role:len({x['atomic_unit_id'] for x in rows}) for role,rows in roles.items()},
        sample_order_sha256={role:subset.axis_hash([x['sample_id'] for x in rows]) for role,rows in roles.items()},
        map_first_bp=points[0][0],map_last_bp=points[-1][0],map_points=len(points),map_extrapolation=False,
        density=dict(scope='ALL_RETAINED_PARENTAL_MARKERS_NOT_COMMON_ONLY_OR_RARE_ONLY',
            first_marker_bp=first,last_marker_bp=last,adjacent_gaps=count_gaps,max_gap_bp=max_gap_bp,max_gap_cm=max_gap_cm,
            markers_per_mapped_cm=kept/(points[-1][1]-points[0][1])),
        indexes=index_audit,input_sha256=bindings,input_contract_sha256=digest,
        source_test_opened=False,simulations_executed=False,training_executed=False,roles_changed=False,donor_split_assigned=False,
        genotypes_modified=False,phase_accuracy_validated=False,quality_validated=False,
        genotype_quality='GT_ONLY_NOT_MEASURED_DP_GQ',original_unsplit_biallelicity='UNKNOWN_NO_NEW_DECOMPOSITION',
        inherited_info_scope='SOURCE3685_NOT_SUBSET_FREQUENCIES; no AC/AN/AF recomputation',
        contains_sample_identifiers=True,public_distribution_allowed=False,
        tool_versions=versions,
        resource_budget=dict(cpus=3,concurrent_compressors=2,compression_threads_each=1,python_processes=1),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES})
    write(out/'summary.json',summary); guard()
    write(out/'manifest.json',dict(summary,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True);parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();os.umask(0o077)
    if args.verify_only:
        authenticate(args.contract,args.contract_sha256);print('AUTHENTICATED_NO_GENOTYPE_PREPARATION')
    else:
        require(args.output_dir,'Output directory required');prepare(args.contract,args.contract_sha256,args.output_dir)


if __name__=='__main__':main()
