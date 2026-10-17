#!/usr/bin/env python3
"""Export authenticated PLINK hardcalls for one existing role and autosome.

This is an input producer with descriptive candidate-label counts, NOT trusted
genotype-support estimation or build discovery.
Only nonpalindromic, unique, unordered exact catalogue pairs are eligible.
PLINK2 assigns REF explicitly to the authenticated FASTA base; a second export
with provisional A2-as-REF provides the pre-transformation hardcalls. Validation
compares unordered nucleotide genotypes and missingness, never raw GT indices.
Neither this operation nor a subsequent REF check certifies historical build,
strand provenance, original multiallelic representation, QC or usable phase.

The output genotype_contract.json conforms to the existing consumer schema,
with GT_ONLY/PASS_OR_UNFILTERED/UNKNOWN. No VERIFIED build receipt is created;
the consumer's existing build gate remains mandatory. There are no counts or
exports for SOURCE_TEST. PLINK may read packed blocks containing other people.

CLI input contract schema r02_lai_plink_export_contract_v1 has exactly:
schema, chromosome, panel_role, reference_contig, expected_plink_version,
input_sha256 (all INPUTS below), limits (all LIMITS below). Each role receives
its own contract. All parameters are explicit; threads=1 is intentional.
Invoke through workflows/r02_lai_plink_export.nf; see its jobs input contract.
The metadata-only `prepare --spec <json> --outdir <new-dir>` subcommand creates
per-chromosome/role contracts and params.json from explicitly authenticated
common files and catalogue manifests; it launches nothing and decodes no GT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import pysam
import r02_lai_allele_support as base
import r02_lai_plink_catalogue as catalogue
import r02_lai_genotype_support as support
from r02_lai_build_verification import reference_contract, file_receipt

INPUTS = ('bed','bim','fam','roles','crosswalk','catalogue_manifest','catalogue_table',
          'bim_table','fasta','fai','reference_contract','contig_map')
LIMITS = ('max_bed_bytes','max_bim_records','max_fam_samples','max_selected_records',
          'max_line_bytes','plink_memory_mib','min_free_gib','timeout_seconds')
ROLES = {'REF_TRAIN','SOURCE_VALID'}
ANCESTRIES = ('AFR', 'EUR', 'NAM')
# Exact aliases used by the authenticated M27F roles and M34/M35 consumers.
# This changes spelling only, never inferred ancestry or biological membership.
ANCESTRY_ALIASES = {'AFR': 'AFR', 'African': 'AFR',
                    'EUR': 'EUR', 'European': 'EUR',
                    'NAM': 'NAM', 'Native_American': 'NAM'}
SCHEMA = 'r02_lai_plink_export_v1'
STATUS = 'COMPLETE_ROLE_EXPORT_BUILD_PROVISIONAL_NOT_GENOTYPE_SUPPORT'
CANDIDATE_SCHEMA = 'r02_lai_plink_candidate_counts_v1'
CANDIDATE_STATUS = 'COMPLETE_CANDIDATE_CODED_ALLELE_COUNTS_NOT_HOMOLOGATED'
MATCH = 'UNORDERED_PAIR_MATCH_ORIENTATION_UNVERIFIED'
CODE_FILES = ('r02_lai_plink_export.py','r02_lai_plink_catalogue.py','r02_lai_allele_support.py',
              'r02_lai_build_verification.py','r02_genomic_pair_evidence.py','r02_lai_genotype_support.py')


def load_contract(args):
    return validate_contract(base.read_json(args.contract),args)


def validate_contract(data,args):
    base.require(set(data)=={'schema','chromosome','panel_role','reference_contig','expected_plink_version','input_sha256','limits'},
                 'Unexpected export contract fields')
    base.require(data['schema']=='r02_lai_plink_export_contract_v1', 'Unknown export contract schema')
    base.require(args.role in ROLES and data['panel_role']==args.role, 'Only explicit REF_TRAIN/SOURCE_VALID roles supported')
    base.require(args.chromosome in {str(i) for i in range(1,23)} and data['chromosome']==args.chromosome, 'Autosome contract mismatch')
    base.require(isinstance(data['reference_contig'],str) and data['reference_contig'], 'Missing reference contig')
    base.require(isinstance(data['expected_plink_version'],str) and data['expected_plink_version'].startswith('PLINK v2.'), 'Explicit PLINK2 version required')
    base.require(set(data['input_sha256'])==set(INPUTS) and set(data['limits'])==set(LIMITS), 'All input hashes/resource bounds required')
    for key in LIMITS:
        value=data['limits'][key]
        base.require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and
                     (value>=0 if key=='min_free_gib' else isinstance(value,int) and value>0), 'Invalid resource bound: '+key)
    base.require(data['limits']['plink_memory_mib']>=640, 'PLINK memory bound must be at least640MiB')
    return data


def lines(path, maximum, compressed=False):
    opener=gzip.open if compressed else open
    with opener(path,'rb') as handle:
        for raw in iter(lambda:handle.readline(maximum+1),b''):
            base.require(len(raw)<=maximum and raw.endswith(b'\n'), 'Metadata line too long or unterminated')
            yield raw.decode('utf-8').rstrip('\r\n')


def table(path, maximum):
    reader=csv.DictReader(lines(path,maximum,compressed=str(path).endswith('.gz')),delimiter='\t')
    base.require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)), 'Missing/duplicate table columns')
    for row in reader:
        base.require(None not in row and all(value is not None for value in row.values()), 'Nonrectangular input table')
        yield row


def canonical_ancestry(label):
    """Resolve documented exact spellings; never infer or relabel ASIA as NAM."""
    base.require(isinstance(label, str) and label in ANCESTRY_ALIASES,
                 'Selected ancestry must be AFR/EUR/NAM or its documented exact long label')
    return ANCESTRY_ALIASES[label]


def canonical_ancestry_groups(path, samples, role):
    """Use the existing role grouping, merging aliases without adding strata."""
    raw = support.ancestry_groups(path, samples, role)
    groups = {'ALL': raw['ALL'], **{ancestry: [] for ancestry in ANCESTRIES}}
    for label, indices in raw.items():
        if label != 'ALL':
            groups[canonical_ancestry(label)].extend(indices)
    for ancestry in ANCESTRIES:
        groups[ancestry].sort()
    flattened = [index for ancestry in ANCESTRIES for index in groups[ancestry]]
    base.require(sorted(flattened) == list(range(len(samples))),
                 'Ancestry strata must partition selected people exactly once')
    return groups


def selected_people(args, limits, receipts):
    fam=[]
    for line in lines(args.fam,limits['max_line_bytes']):
        fields=line.split()
        base.require(len(fields)==6 and len(fam)<limits['max_fam_samples'], 'FAM format or sample cap failed')
        fam.append((fields[0],fields[1]))
    summary=catalogue.sample_metadata(args,receipts)
    role_ids=set()
    ancestry=Counter()
    with Path(args.roles).open() as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            if row['role']==args.role:
                label = canonical_ancestry(row.get('ancestry'))
                role_ids.add(row['sample_id'])
                ancestry[label]+=1
    selected=[pair for pair in fam if '_'.join(pair) in role_ids]
    samples=['_'.join(pair) for pair in selected]
    base.require(samples and set(samples)==role_ids and len(samples)==len(role_ids), 'Role membership must be complete and unambiguous')
    base.require(len(samples)==summary['role_identity_counts'].get(args.role), 'Role count differs from authenticated crosswalk')
    return fam,selected,samples,dict(ancestry)


def metadata_selection(args, data, receipts, guard):
    limits=data['limits']
    manifest=base.read_json(args.catalogue_manifest)
    base.require(manifest.get('schema')=='r02_lai_plink_catalogue_v1' and manifest.get('status')==catalogue.STATUS and
                 str(manifest.get('chromosome'))==args.chromosome, 'Compatible completed BIM catalogue required')
    for key in ('bim','fam','roles','crosswalk','fasta','fai','reference_contract','contig_map'):
        base.require(manifest.get('input_files',{}).get(key,{}).get('sha256')==receipts[key]['sha256'], 'Catalogue input identity differs: '+key)
    for key,name in (('catalogue_table','catalogue_correspondence.tsv.gz'),('bim_table','bim_chr22_compatibility.tsv.gz')):
        # The v1 producer used chr22 in its filename; chromosome identity lives
        # in the manifest, never inferred from a filename. Other chromosomes
        # may use a chromosome-specific name declared in the same inventory.
        inventory=manifest.get('outputs_sha256',{})
        actual_name=Path(getattr(args,key)).name
        base.require(inventory.get(actual_name)==receipts[key]['sha256'], 'Catalogue table hash differs: '+key)
    identity=reference_contract(base.read_json(args.reference_contract))
    base.require(identity['fasta_sha256']==receipts['fasta']['sha256'] and identity['fai_sha256']==receipts['fai']['sha256'],
                 'Reference identity hashes differ')
    base.require(Path(args.fai).resolve()==Path(str(args.fasta)+'.fai').resolve(), 'Authenticated adjacent FASTA index required')
    mapping=base.contig_mapping(args.contig_map,args.chromosome)
    cohort=manifest['source_cohort']
    base.require(isinstance(cohort.get('samples'),int) and cohort['samples']>0, 'Invalid source cohort')
    selected,counts={},Counter()
    previous=0
    with pysam.FastaFile(str(args.fasta)) as reference:
        ref_contig=data['reference_contig']
        base.require(ref_contig in identity['contigs'] and reference.get_reference_length(ref_contig)==identity['contigs'][ref_contig],
                     'Reference contig length differs')
        for row in table(args.catalogue_table,limits['max_line_bytes']):
            pos=int(row['position_1based'])
            base.require(row['chromosome']==args.chromosome and pos>previous, 'Catalogue positions not unique/increasing')
            previous=pos
            counts['source_rows']+=1
            if row['catalogue_status']!=MATCH:
                continue
            ref,alt=row['source_REF'],row['source_ALT']
            base.require(catalogue.snv_pair(ref,alt) and frozenset((ref,alt)) not in {frozenset('AT'),frozenset('CG')},
                         'Palindromic/non-SNV match cannot be exported')
            base.require(int(row['bim_rows_at_position'])==int(row['unordered_exact_pair_rows'])==1 and
                         int(row['complement_pair_rows'])==int(row['unsupported_pair_rows'])==int(row['duplicate_pair_excess_rows'])==0,
                         'Ambiguous catalogue match')
            selected_index=int(row['source_rare_allele_index'])
            ac,an=int(row['source_rare_ac']),int(row['source_rare_an'])
            base.require(selected_index in (0,1) and row['counted_allele']==(ref,alt)[selected_index] and
                         2<=ac and 100*ac<=an<=2*cohort['samples'], 'Invalid preserved source-minor contract')
            base.require(1<=pos<=identity['contigs'][ref_contig] and reference.fetch(ref_contig,pos-1,pos).upper()==ref,
                         'Matched source REF differs from authenticated FASTA')
            base.require(len(selected)<limits['max_selected_records'], 'Selected marker cap exceeded')
            selected[pos]=dict(row)
            counts['source_minor_ref' if selected_index==0 else 'source_minor_alt']+=1
        base.require(counts['source_rows']==manifest['counts']['source_rows'] and len(selected)==manifest['catalogue_status_counts'].get(MATCH,0)
                     and selected, 'Catalogue source/match denominators differ or selection empty')
    matched_ordinals={}
    ids=set()
    seen_positions=Counter()
    for row in table(args.bim_table,limits['max_line_bytes']):
        pos=int(row['position_1based'])
        if pos not in selected:
            continue
        seen_positions[pos]+=1
        source=selected[pos]
        a1,a2=row['A1'],row['A2']
        ordinal=int(row['bim_row_ordinal'])
        base.require(ordinal>0 and ordinal not in matched_ordinals and row['variant_id'] not in ids and
                     row['variant_id'] not in {'','.','0'} and not any(c.isspace() for c in row['variant_id']), 'Ambiguous BIM marker ID/ordinal')
        base.require(frozenset((a1,a2))==frozenset((source['source_REF'],source['source_ALT'])) and
                     row['fasta_base']==source['source_REF'] and row['fasta_compatibility']=='REFERENCE_BASE_IN_UNORDERED_PAIR',
                     'BIM pair or FASTA compatibility differs')
        source.update(bim_row_ordinal=ordinal,variant_id=row['variant_id'],A1=a1,A2=a2)
        ids.add(row['variant_id'])
        matched_ordinals[ordinal]=source
    base.require(set(seen_positions)==set(selected) and all(n==1 for n in seen_positions.values()), 'Missing or duplicated BIM table positions')
    total,observed,selected_id_occurrences=0,[],Counter()
    for ordinal,line in enumerate(lines(args.bim,limits['max_line_bytes']),1):
        fields=line.split()
        base.require(ordinal<=limits['max_bim_records'] and len(fields)==6, 'BIM format/record cap failed')
        total=ordinal
        if fields[0] in mapping and fields[1] in ids:
            selected_id_occurrences[fields[1]]+=1
            base.require(ordinal in matched_ordinals, 'Selected marker ID is duplicated or catalogue ordinal differs')
        if ordinal in matched_ordinals:
            row=matched_ordinals[ordinal]
            base.require(fields[0] in mapping and fields[1]==row['variant_id'] and int(fields[3])==int(row['position_1based']) and
                         fields[4:]==[row['A1'],row['A2']], 'Raw BIM differs from authenticated catalogue row')
            observed.append(row)
        if ordinal%100000==0: guard()
    base.require(total==manifest['counts']['bim_all_rows'] and len(observed)==len(selected) and
                 all(selected_id_occurrences[key]==1 for key in ids), 'Raw BIM denominator or marker uniqueness failed')
    base.require([int(row['position_1based']) for row in observed]==sorted(selected), 'Selected BIM positions must be sorted for streaming VCF')
    return observed,dict(counts),manifest,identity


def nucleotide_gt(call,alleles):
    gt=call.get('GT')
    base.require(gt is not None and len(gt)==2, 'Unexpected haploid/non-diploid call in autosomal export')
    base.require(not call.phased, 'BED-derived export must not assert phase')
    if any(value is None for value in gt):
        base.require(all(value is None for value in gt), 'Unexpected partial missing call in BED-derived export')
        return None
    base.require(all(isinstance(value,int) and 0<=value<len(alleles) for value in gt), 'Invalid exported allele index')
    return tuple(sorted(alleles[value] for value in gt))


def verify_exports(before,after,selected,samples,max_line_bytes,guard):
    for path in (before,after):
        with base.MetadataVCF(path,max_line_bytes) as reader:
            base.require(reader.header.samples==samples, 'Exported sample order/membership differs')
            base.require(sum(1 for unused in reader)==len(selected), 'Exported record count differs')
    checked,missing=0,0
    with pysam.VariantFile(str(before)) as raw,pysam.VariantFile(str(after)) as changed:
        for source,old,new in itertools.zip_longest(selected,raw,changed):
            base.require(source is not None and old is not None and new is not None, 'Export row count changed')
            pos=int(source['position_1based'])
            base.require(old.id==new.id==source['variant_id'] and old.pos==new.pos==pos, 'Export marker order/position changed')
            base.require(old.contig==new.contig==source['chromosome'], 'Export chromosome identity changed')
            base.require(tuple(old.alleles)==(source['A2'],source['A1']), 'Provisional export changed original BIM allele order')
            base.require(tuple(new.alleles)==(source['source_REF'],source['source_ALT']), 'Explicit REF assignment failed')
            base.require(set(old.format)==set(new.format)=={'GT'} and set(old.filter)==set(new.filter), 'Unexpected quality fields/filter transformation')
            for sample in samples:
                lhs=nucleotide_gt(old.samples[sample],old.alleles)
                rhs=nucleotide_gt(new.samples[sample],new.alleles)
                base.require(lhs==rhs, 'REF assignment changed nucleotide genotype or missingness')
                missing+=int(rhs is None)
                checked+=1
            if checked%10000<len(samples): guard()
    return dict(genotype_cells_compared=checked,missing_diploid_cells=missing,nucleotide_genotypes_preserved=True,
                verification_scope='All selected variants and selected-role people only; comparison is integrity validation, not allele-support estimation')


def candidate_counts(path, selected, samples, args, genotype, target, guard):
    """Reuse existing counters, separately labeled; NEVER invoke/waive build gates.

    REF-minor rows here count the recorded nucleotide label, not authenticated
    molecular REF dosage. Representation UNKNOWN and every trusted field remain
    explicit. This is a new descriptive output, not r02_lai_genotype_support_v1.
    """
    groups=canonical_ancestry_groups(args.roles,samples,args.role)
    columns=['chromosome','position_1based','source_REF','source_ALT','counted_allele','source_rare_allele_index',
             'source_rare_ac','source_rare_an','panel_role','panel_ancestry','n_role_members',
             'candidate_ac_called','candidate_an_called','candidate_af_called','candidate_carriers_called',
             'candidate_n_gt_complete','candidate_n_gt_missing','candidate_called_call_rate',
             'candidate_status','historical_build','allele_representation','genotype_quality','phase',
             'trusted_ac','trusted_an','trusted_af','trusted_carriers','trusted_support_status']
    rows=0
    with pysam.VariantFile(str(path)) as reader,gzip.open(target,'xt') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for source,record in itertools.zip_longest(selected,reader):
            base.require(source is not None and record is not None and record.pos==int(source['position_1based']) and
                         tuple(record.alleles)==(source['source_REF'],source['source_ALT']), 'Candidate stream identity changed')
            allele_index=record.alleles.index(source['counted_allele'])
            calls=support.count_record(record,allele_index,samples,genotype)
            for ancestry,indices in groups.items():
                counts=(support.group_support(calls,indices,genotype) if indices else
                        dict(ac_counted_called='NA',an_called=0,af_counted_called='NA',n_carriers_called=0,
                             n_gt_complete=0,n_gt_missing=0,called_call_rate='NA'))
                row={key:source[key] for key in columns[:8]}
                row.update(panel_role=args.role,panel_ancestry=ancestry,n_role_members=len(indices),
                    candidate_ac_called=counts['ac_counted_called'],candidate_an_called=counts['an_called'],
                    candidate_af_called=counts['af_counted_called'],candidate_carriers_called=counts['n_carriers_called'],
                    candidate_n_gt_complete=counts['n_gt_complete'],candidate_n_gt_missing=counts['n_gt_missing'],
                    candidate_called_call_rate=counts['called_call_rate'],
                    candidate_status='CANDIDATE_CODED_ALLELE_BUILD_AND_REPRESENTATION_UNVERIFIED',historical_build='UNVERIFIED',
                    allele_representation='UNKNOWN',genotype_quality='GT_ONLY_NOT_VALIDATED',phase='NOT_AVAILABLE_BED_NOT_PHASED',
                    trusted_ac='NA',trusted_an='NA',trusted_af='NA',trusted_carriers='NA',
                    trusted_support_status='NOT_HOMOLOGATED_BUILD_AND_REPRESENTATION_UNVERIFIED')
                writer.writerow(row)
                rows+=1
            if rows%1000==0: guard()
    return rows


def run(args):
    started=time.monotonic()
    code={name:base.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
    contract_sha=base.sha256(args.contract)
    data=load_contract(args)
    limits=data['limits']
    out=Path(args.outdir)
    base.require(out.parent.is_dir() and not out.exists(), 'Output must be new in existing local directory')
    work=out.parent/(out.name+'_operations')
    base.require(not work.exists(), 'Never overwrite an earlier operation directory')
    def guard():
        base.free_disk(out.parent,limits['min_free_gib'])
        base.require(time.monotonic()-started<limits['timeout_seconds'], 'PLINK export timeout')
    guard()
    base.require(Path(args.bed).stat().st_size<=limits['max_bed_bytes'], 'BED exceeds byte cap')
    receipts={key:file_receipt(getattr(args,key),data['input_sha256'][key]) for key in INPUTS}
    fam,people,samples,ancestries=selected_people(args,limits,receipts)
    selected,selection_counts,catalogue_manifest,identity=metadata_selection(args,data,receipts,guard)
    with Path(args.bed).open('rb') as handle:
        base.require(handle.read(3)==b'\x6c\x1b\x01', 'BED must be variant-major PLINK1 format')
    expected_bytes=3+catalogue_manifest['counts']['bim_all_rows']*((len(fam)+3)//4)
    base.require(receipts['bed']['size_bytes']==expected_bytes, 'BED byte dimensions differ from authenticated BIM/FAM')
    binary=shutil.which('plink2')
    base.require(binary,'PLINK2 executable not found')
    version=subprocess.check_output([binary,'--version'],text=True,timeout=30).strip()
    base.require(version==data['expected_plink_version'],'PLINK2 version differs from contract')
    binary_sha=base.sha256(binary)
    out.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    keep,extract,refs=work/'keep.tsv',work/'extract.txt',work/'reference_alleles.tsv'
    keep.write_text(''.join(fid+'\t'+iid+'\n' for fid,iid in people))
    extract.write_text(''.join(row['variant_id']+'\n' for row in selected))
    refs.write_text(''.join(row['variant_id']+'\t'+row['source_REF']+'\n' for row in selected))
    common=[binary,'--bed',str(args.bed),'--bim',str(args.bim),'--fam',str(args.fam),
            '--chr',args.chromosome,'--keep',str(keep),'--extract',str(extract),
            '--threads','1','--memory',str(limits['plink_memory_mib']),'require',
            '--export','vcf','bgz','id-paste=fid,iid']
    commands=[]
    for label,prefix,extra in (('provisional',work/'provisional',[]),
                               ('assigned',out/'role',['--ref-allele',str(refs),'2','1'])):
        guard()
        command=common+extra+['--out',str(prefix)]
        commands.append(command)
        with (work/(label+'.stdout.log')).open('x') as stdout,(work/(label+'.stderr.log')).open('x') as stderr:
            completed=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=max(1,limits['timeout_seconds']-(time.monotonic()-started)))
        base.require(completed.returncode==0,'PLINK2 '+label+' export failed; inspect private operation logs')
    before,after=work/'provisional.vcf.gz',out/'role.vcf.gz'
    validation=verify_exports(before,after,selected,samples,limits['max_line_bytes'],guard)
    pysam.tabix_index(str(after),preset='vcf',force=False)
    # PLINK operation logs can contain sample/variant metadata; keep them with
    # private operations, not in the consumer's declared artifact directory.
    log=out/'role.log'
    if log.exists(): log.rename(work/'assigned.plink.log')
    selected_path=out/'selected_sites.tsv'
    fields=list(selected[0])
    with selected_path.open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
    vcf_sha=base.sha256(after)
    genotype=dict(schema='r02_lai_genotype_contract_v1',panel_role=args.role,
        panel_sample_order_sha256=base.sample_hash(samples),panel_sample_members_sha256=base.sample_hash(sorted(samples)),
        site_filter_policy='PASS_OR_UNFILTERED',quality=dict(contract_id='PLINK_HARDCALL_GT_ONLY_NO_DP_GQ',mode='GT_ONLY'),
        allele_representation=dict(status='UNKNOWN',panel_vcf_sha256=vcf_sha,
            evidence='BED/BIM and its metadata do not establish original multiallelic decomposition; REF assignment preserves hardcalls but cannot recover lost other-ALT provenance'))
    with (out/'genotype_contract.json').open('x') as handle: json.dump(genotype,handle,indent=2)
    evidence=out/'candidate_evidence'
    evidence.mkdir(mode=0o700)
    candidate_rows=candidate_counts(after,selected,samples,args,genotype,evidence/'candidate_counts.tsv.gz',guard)
    receipt=dict(schema='r02_lai_plink_reference_assignment_v1',status='BUILD_PROVISIONAL_REFERENCE_ASSIGNED_NOT_VERIFIED',
        panel_vcf_sha256=vcf_sha,reference_provenance=identity,
        operation='PLINK2 --ref-allele with a per-marker base from authenticated FASTA; no complement/strand flip/normalization/imputation',
        historical_build='UNVERIFIED',historical_strand='UNVERIFIED',allele_representation='UNKNOWN',
        limitation='A later REF/FASTA check confirms this explicit assignment, not independent historical coordinate/build provenance',
        validation=validation)
    with (out/'reference_assignment.json').open('x') as handle: json.dump(receipt,handle,indent=2)
    consumer=dict(schema='r02_lai_plink_export_consumer_inputs_v1',panel_vcf='role.vcf.gz',genotype_contract='genotype_contract.json',
        role=args.role,chromosome=args.chromosome,roles_sha256=receipts['roles']['sha256'],
        source_catalogue_sha256=catalogue_manifest['input_files']['rare_vcf']['sha256'],
        build_gate='BLOCKED_UNTIL_INDEPENDENT_BUILD_PROVENANCE_AND_EXISTING_ALL_RECORD_REF_RECEIPTS',
        forbidden='Do not pass reference_assignment.json as r02_lai_build_verification_v1 or synthesize VERIFIED evidence',
        next_use='Existing r02_lai_allele_support.nf genotype_support once its unmodified source/panel build gates are legitimately met; retain full authenticated source catalogue and representationUNKNOWN')
    with (out/'consumer_inputs.json').open('x') as handle: json.dump(consumer,handle,indent=2)
    base.verify_inputs_unchanged(receipts)
    base.require(base.sha256(args.contract)==contract_sha and base.sha256(binary)==binary_sha and
                 all(base.sha256(Path(__file__).with_name(name))==digest for name,digest in code.items()), 'Contract/code/tool changed during export')
    guard()
    missing_scope=dict(source_catalogue_rows=selection_counts['source_rows'],selected_rows=len(selected),
        nonselected_rows=selection_counts['source_rows']-len(selected),
        original_catalogue_status_counts=catalogue_manifest['catalogue_status_counts'],
        interpretation='Complete source catalogue remains authenticated by input hash; nonselected positions have no candidate GT count and are UNKNOWN, never0')
    aggregate=dict(schema=CANDIDATE_SCHEMA,status=CANDIDATE_STATUS,created_at=datetime.now(timezone.utc).isoformat(),
        chromosome=args.chromosome,panel_role=args.role,evidence_level='DESCRIPTIVE_PROVISIONAL_CODED_ALLELES_ONLY',
        n_role_members=len(samples),ancestry_counts=ancestries,exported_variants=len(selected),output_rows=candidate_rows,
        ancestry_label_aliases=ANCESTRY_ALIASES,
        source_cohort=catalogue_manifest['source_cohort'],source_selection=selection_counts,source_missing_scope=missing_scope,
        input_files=receipts,contract_sha256=contract_sha,code_sha256=code,package_versions=dict(plink2=version,pysam=pysam.__version__),
        candidate_scope='Recorded nucleotide-label hardcall counts only; source REF-minor counts do not certify molecular REF dosage under unknown decomposition',
        build_status='PROVISIONAL_NOT_HISTORICALLY_CERTIFIED',allele_representation='UNKNOWN',genotype_quality='GT_ONLY_NOT_VALIDATED',
        phase='NOT_AVAILABLE_BED_NOT_PHASED',trusted_counts='ALL_NA',source_test_statistics_computed=False,
        strata_scope='ALL overlaps AFR/EUR/NAM; never add role total to ancestry strata',
        homologated_consumer_gate='UNCHANGED_BLOCKED_UNTIL_INDEPENDENT_PROVENANCE_AND_REQUIRED_AUDITS',
        contains_sample_identifiers=False,contains_individual_genotypes=False,public_distribution_allowed=False,
        outputs_sha256={'candidate_counts.tsv.gz':base.sha256(evidence/'candidate_counts.tsv.gz')})
    with (evidence/'manifest.json').open('x') as handle: json.dump(aggregate,handle,indent=2,allow_nan=False)
    result=dict(schema=SCHEMA,status=STATUS,created_at=datetime.now(timezone.utc).isoformat(),chromosome=args.chromosome,
        panel_role=args.role,input_files=receipts,contract_sha256=contract_sha,
        role_validation=dict(n_role_members=len(samples),ancestry_counts=ancestries,sample_order_sha256=base.sample_hash(samples),
                             membership_sha256=base.sample_hash(sorted(samples))),
        ancestry_label_aliases=ANCESTRY_ALIASES,
        source_cohort=catalogue_manifest['source_cohort'],source_selection=selection_counts,exported_variants=len(selected),
        original_source_minor_preserved=True,genotype_quality='GT_ONLY_NOT_VALIDATED',phase='NOT_AVAILABLE_BED_NOT_PHASED',
        build_status='PROVISIONAL_NOT_HISTORICALLY_CERTIFIED',allele_representation='UNKNOWN',
        source_test_exported=False,source_test_statistics_computed=False,
        parsing_scope='PLINK may read packed source blocks including excluded people; export and integrity comparisons restricted to selected role',
        genotype_support_computed=False,candidate_coded_allele_counts_computed=True,candidate_output_rows=candidate_rows,
        candidate_scope='Descriptive recorded-label hardcalls only; not homologated support, molecular REF dosage, phase, genotype QC or usable LAI material',
        trusted_counts='ALL_NA',strata_scope='ALL overlaps AFR/EUR/NAM, never sum role total and ancestry strata',
        source_missing_scope=missing_scope,aggregate_only_publication_directory='candidate_evidence',
        validation=validation,package_versions=dict(plink2=version,pysam=pysam.__version__),
        binary_sha256=binary_sha,code_sha256=code,commands=commands,limits=limits,
        resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        contains_sample_identifiers=True,contains_individual_genotypes=True,public_distribution_allowed=False,
        outputs_sha256={p.relative_to(out).as_posix():base.sha256(p) for p in out.rglob('*') if p.is_file()})
    with (out/'manifest.json').open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in INPUTS+('contract','role','chromosome','outdir'):
        p.add_argument('--'+name.replace('_','-'),required=True)
    return p


def prepare(spec_path,outdir):
    """Create contracts/Nextflow params, with no BED genotype interpretation.

    spec schema r02_lai_plink_export_preparation_v1 exact keys: schema, common
    (bed,bim,fam,roles,crosswalk paths), common_sha256, chromosomes, panel_roles,
    limits, expected_plink_version, source_scripts. Each chromosomes entry has
    chromosome,reference_contig,catalogue_manifest,catalogue_manifest_sha256,
    catalogue_table,bim_table,fasta,fai,reference_contract,contig_map.
    Common hashes and each catalogue manifest digest must be supplied by the
    caller's authenticated staging/publication receipts, never guessed here.
    Other expected hashes are recovered from those authenticated manifests.
    """
    spec_sha=base.sha256(spec_path)
    generator_sha=base.sha256(__file__)
    spec=base.read_json(spec_path)
    base.require(set(spec)=={'schema','common','common_sha256','chromosomes','panel_roles','limits','expected_plink_version','source_scripts'} and
                 spec['schema']=='r02_lai_plink_export_preparation_v1', 'Invalid preparation specification')
    common_names={'bed','bim','fam','roles','crosswalk'}
    base.require(set(spec['common'])==set(spec['common_sha256'])==common_names, 'Exact five common inputs/hashes required')
    base.require(isinstance(spec['panel_roles'],list) and spec['panel_roles'] and len(spec['panel_roles'])==len(set(spec['panel_roles'])) and
                 set(spec['panel_roles'])<=ROLES, 'Preparation roles must be unique REF_TRAIN/SOURCE_VALID')
    base.require(isinstance(spec['chromosomes'],list) and spec['chromosomes'], 'Explicit nonempty chromosome list required')
    out=Path(outdir)
    base.require(out.parent.is_dir() and not out.exists(), 'Preparation output must be new')
    base.free_disk(out.parent,spec['limits']['min_free_gib'])
    scripts=[Path(path).resolve() for path in spec['source_scripts']]
    base.require(len(scripts)==len(CODE_FILES) and {path.name for path in scripts}==set(CODE_FILES) and
                 all(path.is_file() for path in scripts), 'Explicit six-file source closure required')
    source_hashes={str(path):base.sha256(path) for path in scripts}
    common={key:str(Path(path).resolve()) for key,path in spec['common'].items()}
    receipts={key:file_receipt(path,spec['common_sha256'][key]) for key,path in common.items()}
    role_validation={}
    # Resolve real metadata before scheduling any genotype-export task. The
    # same validation is repeated within each task against its frozen inputs.
    for role in spec['panel_roles']:
        _,_,samples,ancestries=selected_people(argparse.Namespace(**common,role=role),spec['limits'],receipts)
        role_validation[role]=dict(n_role_members=len(samples),ancestry_counts=ancestries,
                                  sample_order_sha256=base.sample_hash(samples),
                                  membership_sha256=base.sample_hash(sorted(samples)))
    contracts,jobs={},[]
    seen=set()
    fields={'chromosome','reference_contig','catalogue_manifest','catalogue_manifest_sha256','catalogue_table','bim_table',
            'fasta','fai','reference_contract','contig_map'}
    for entry in spec['chromosomes']:
        base.require(set(entry)==fields and entry['chromosome'] not in seen,'Invalid or repeated preparation chromosome')
        chrom=entry['chromosome'];seen.add(chrom)
        metadata=file_receipt(entry['catalogue_manifest'],entry['catalogue_manifest_sha256'])
        manifest=base.read_json(entry['catalogue_manifest'])
        base.require(manifest.get('schema')=='r02_lai_plink_catalogue_v1' and manifest.get('status')==catalogue.STATUS and
                     str(manifest.get('chromosome'))==chrom,'Invalid completed catalogue for preparation')
        hashes={key:receipt['sha256'] for key,receipt in receipts.items()}
        for key in common_names-{'bed'}:
            base.require(manifest.get('input_files',{}).get(key,{}).get('sha256')==hashes[key], 'Catalogue/common identity mismatch: '+key)
        paths={key:str(Path(entry[key]).resolve()) for key in fields-{'chromosome','reference_contig','catalogue_manifest_sha256'}}
        hashes['catalogue_manifest']=metadata['sha256']
        for key in ('catalogue_table','bim_table','fasta','fai','reference_contract','contig_map'):
            expected=(manifest.get('outputs_sha256',{}).get(Path(paths[key]).name) if key in ('catalogue_table','bim_table')
                      else manifest.get('input_files',{}).get(key,{}).get('sha256'))
            hashes[key]=file_receipt(paths[key],expected)['sha256']
        for role in spec['panel_roles']:
            contract=dict(schema='r02_lai_plink_export_contract_v1',chromosome=chrom,panel_role=role,
                reference_contig=entry['reference_contig'],expected_plink_version=spec['expected_plink_version'],input_sha256=hashes,limits=spec['limits'])
            validate_contract(contract,argparse.Namespace(role=role,chromosome=chrom))
            name=f'chr{chrom}.{role}.contract.json'
            contracts[name]=contract
            jobs.append(dict(chromosome=chrom,role=role,contract=str(out.resolve()/name),**paths))
    base.require(base.sha256(spec_path)==spec_sha and base.sha256(__file__)==generator_sha and
                 all(base.sha256(path)==digest for path,digest in source_hashes.items()),
                 'Preparation spec or source closure changed during authentication')
    out.mkdir(mode=0o700)
    for name,data in contracts.items():
        with (out/name).open('x') as handle: json.dump(data,handle,indent=2)
    with (out/'params.json').open('x') as handle:
        json.dump(dict(**common,source_scripts=[str(path) for path in scripts],jobs=jobs),handle,indent=2)
    receipt=dict(schema='r02_lai_plink_export_prepared_v1',status='PREPARED_NO_GT_READ_NO_LAUNCH_BUILD_UNVERIFIED',
        spec_sha256=spec_sha,generator_sha256=generator_sha,jobs=len(jobs),chromosomes=sorted(seen,key=int),roles=spec['panel_roles'],
        source_sha256={Path(path).name:digest for path,digest in source_hashes.items()},common_input_receipts=receipts,
        role_validation=role_validation,ancestry_label_aliases=ANCESTRY_ALIASES,
        outputs_sha256={path.name:base.sha256(path) for path in out.iterdir()},
        next_action='Review resources/immutable closure, then launch r02_lai_plink_export.nf with params.json; never manufacture build verification')
    with (out/'preparation_manifest.json').open('x') as handle: json.dump(receipt,handle,indent=2)
    return receipt


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        p=argparse.ArgumentParser(description=prepare.__doc__)
        p.add_argument('--spec',required=True);p.add_argument('--outdir',required=True)
        args=p.parse_args(sys.argv[2:]);prepare(args.spec,args.outdir)
    else:
        run(parser().parse_args())
