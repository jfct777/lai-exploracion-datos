#!/usr/bin/env python3
"""Metadata-only BIM/source-minor/FASTA audit: no BED, GT or strand repair.

A1/A2 are an unordered catalogue pair, NEVER identified as REF/ALT. FASTA
compatibility cannot identify the panel assembly or certify phase/quality.
SQLite retains every target-chromosome BIM row; source output retains every M02.1 row.
Resource bounds are engineering limits, not biological filters.
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
import sqlite3
import time
from types import SimpleNamespace

import pysam
import r02_lai_allele_support as base
from r02_lai_build_verification import reference_contract

INPUTS = ('bim', 'fam', 'roles', 'crosswalk', 'rare_vcf', 'rare_contract',
          'fasta', 'fai', 'reference_contract', 'contig_map')
STATUS = 'COMPLETE_CATALOGUE_ONLY_NO_GENOTYPES_NO_BUILD_CERTIFICATION'
COMPLEMENT = str.maketrans('ACGT', 'TGCA')


def snv_pair(a1, a2):
    return len(a1) == len(a2) == 1 and a1 in 'ACGT' and a2 in 'ACGT' and a1 != a2


def fasta_status(a1, a2, reference):
    if reference == 'OUT_OF_RANGE':
        return 'INVALID_POSITION'
    if a1 in {'0', '.'} or a2 in {'0', '.'}:
        return 'MISSING_ALLELE_UNKNOWN'
    if not snv_pair(a1, a2):
        return 'UNSUPPORTED_NON_DISTINCT_SNV_PAIR'
    if reference not in 'ACGT':
        return 'FASTA_NON_ACGT_UNKNOWN'
    if reference in {a1, a2}:
        return 'REFERENCE_BASE_IN_UNORDERED_PAIR'
    if reference.translate(COMPLEMENT) in {a1, a2}:
        return 'COMPLEMENT_REFERENCE_CANDIDATE_NOT_HARMONIZED'
    return 'REFERENCE_BASE_NOT_IN_PAIR'


def correspondence(ref, alt, rows):
    pairs = [frozenset((row[3], row[4])) for row in rows if snv_pair(row[3], row[4])]
    expected = frozenset((ref, alt))
    complement = frozenset((ref.translate(COMPLEMENT), alt.translate(COMPLEMENT)))
    direct, reverse = pairs.count(expected), pairs.count(complement)
    detail = [len(rows), direct, reverse, len(set(pairs)), len(rows) - len(pairs), len(pairs) - len(set(pairs))]
    if not rows:
        status = 'NO_BIM_POSITION_UNKNOWN'
    elif len(rows) > 1:
        status = 'AMBIGUOUS_MULTIPLE_BIM_ROWS'
    elif not pairs:
        status = 'UNSUPPORTED_BIM_ALLELES'
    elif direct and expected == complement:
        status = 'PALINDROMIC_PAIR_STRAND_AMBIGUOUS'
    elif direct:
        status = 'UNORDERED_PAIR_MATCH_ORIENTATION_UNVERIFIED'
    elif reverse:
        status = 'COMPLEMENT_PAIR_CANDIDATE_NOT_HARMONIZED'
    else:
        status = 'POSITION_PRESENT_PAIR_MISMATCH'
    return status, detail


def sample_metadata(args, receipts):
    proof = base.read_json(args.crosswalk)
    base.require(proof.get('schema') == 'r02_l1_documented_id_crosswalk_v1' and
                 proof.get('status') == 'BROAD_PANEL_ROLE_ID_CROSSWALK_VERIFIED_METADATA_ONLY', 'Invalid crosswalk receipt')
    base.require(proof.get('hashes', {}).get('roles') == receipts['roles']['sha256'] and
                 proof.get('hashes', {}).get('broad_fam') == receipts['fam']['sha256'], 'Crosswalk input hashes differ')
    fam_keys = set()
    with Path(args.fam).open() as handle:
        for line in handle:
            fields = line.split()
            base.require(len(fields) == 6, 'Malformed FAM metadata')
            key = fields[0] + '_' + fields[1]
            base.require(key not in fam_keys, 'Duplicate documented FID_IID key')
            fam_keys.add(key)
    roles, matched, seen = Counter(), Counter(), set()
    with Path(args.roles).open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        base.require({'sample_id', 'role'} <= set(reader.fieldnames or []), 'Missing role columns')
        for row in reader:
            sample, role = row['sample_id'], row['role']
            base.require(sample and role and sample not in seen, 'Empty or duplicate role ID')
            seen.add(sample)
            roles[role] += 1
            if sample in fam_keys:
                matched[role] += 1
    broad = proof.get('broad', {})
    base.require(len(fam_keys) == broad.get('fam_rows') and dict(roles) == proof.get('m27f', {}).get('role_denominators') and
                 dict(matched) == broad.get('role_counts') and broad.get('methods_agree') is True and
                 broad.get('ambiguous_keys') == 0, 'Crosswalk aggregates fail independent documented FID_IID check')
    return dict(fam_rows=len(fam_keys), role_identity_counts=dict(matched), roles_total=dict(roles),
                interpretation='Identity metadata only; no genotype denominators or genotype concordance')


def guard(args, db, started):
    base.require(time.monotonic() - started <= args.timeout_seconds, 'Catalogue audit timeout')
    base.free_disk(db.parent, args.min_free_gib)
    total = sum(path.stat().st_size for path in db.parent.glob(db.name + '*') if path.is_file())
    base.require(total <= args.max_sqlite_bytes, 'SQLite and sidecars exceed max-sqlite-bytes')
    return total


def bounded_lines(path, maximum):
    with Path(path).open('rb') as handle:
        for raw in iter(lambda: handle.readline(maximum + 1), b''):
            base.require(len(raw) <= maximum and raw.endswith(b'\n'), 'BIM line exceeds limit or is unterminated')
            yield raw.decode('utf-8').split()


def run(args):
    started = time.monotonic()
    code_paths = [Path(module.__file__).resolve() for module in
                  (__import__(__name__),base,__import__('r02_lai_build_verification'),__import__('r02_genomic_pair_evidence'))]
    code_hashes = {str(path):base.sha256(path) for path in code_paths}
    for key in ('max_bim_bytes', 'max_bim_records', 'max_line_bytes', 'max_sqlite_bytes',
                'sqlite_cache_mib', 'batch_rows', 'max_rows_per_position', 'timeout_seconds',
                'expected_source_samples', 'expected_rare_sites'):
        base.require(getattr(args, key) > 0, 'Positive resource bounds and denominators required')
    base.require(math.isfinite(args.min_free_gib) and args.min_free_gib >= 0, 'Invalid disk reserve')
    base.require(args.chromosome in {str(i) for i in range(1, 23)}, 'Canonical autosome number 1-22 required')
    out = Path(args.outdir)
    base.require(not out.exists() and out.parent.is_dir(), 'Output must be new under an existing local directory')
    base.free_disk(out.parent, args.min_free_gib)
    base.require(Path(args.bim).stat().st_size <= args.max_bim_bytes, 'BIM exceeds max-bim-bytes')
    base.require(args.input_hashes, 'Exact expected hashes for all ten inputs required')
    receipts = base.input_receipts(args, INPUTS)
    identity = reference_contract(base.read_json(args.reference_contract))
    base.require(identity['fasta_sha256'] == receipts['fasta']['sha256'] and
                 identity['fai_sha256'] == receipts['fai']['sha256'], 'Reference identity digest mismatch')
    base.require(Path(args.fai).resolve() == Path(str(args.fasta) + '.fai').resolve(), 'FASTA requires authenticated adjacent FAI')
    base.require(args.reference_contig in identity['contigs'], 'Reference contig missing')
    mapping = base.contig_mapping(args.contig_map, args.chromosome)
    metadata = sample_metadata(args, receipts)
    contract = base.read_json(args.rare_contract)
    out.mkdir(mode=0o700)
    db = out.parent / (out.name + '.sqlite')
    base.require(not db.exists(), 'Refuse existing SQLite state; Nextflow resumes completed tasks only')
    con = sqlite3.connect(db)
    con.execute('PRAGMA page_size=4096')
    con.execute('PRAGMA journal_mode=OFF')  # disposable task-local index; never resume a partial DB
    con.execute('PRAGMA temp_store=FILE')
    con.execute(f'PRAGMA cache_size={-args.sqlite_cache_mib * 1024}')
    con.execute(f'PRAGMA max_page_count={max(1, args.max_sqlite_bytes // 4096)}')
    # Page cap bounds the main DB; temp sorting files may be unlinked by SQLite.
    # Check actual filesystem reserve/time during SQL as well as between batches.
    con.set_progress_handler(lambda: (guard(args, db, started) and 0), 10000)
    con.execute('CREATE TABLE bim (row_no INTEGER PRIMARY KEY, pos INTEGER, variant_id TEXT, a1 TEXT, a2 TEXT)')
    counts, states, bim_states, contigs = Counter(), Counter(), Counter(), Counter()
    max_db_bytes = 0
    try:
        with pysam.FastaFile(str(args.fasta)) as reference, gzip.open(out/f'bim_chr{args.chromosome}_compatibility.tsv.gz', 'wt') as handle:
            base.require(reference.get_reference_length(args.reference_contig) == identity['contigs'][args.reference_contig],
                         'FASTA contig length mismatch')
            length = identity['contigs'][args.reference_contig]
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['bim_row_ordinal', 'position_1based', 'variant_id', 'A1', 'A2', 'fasta_base', 'fasta_compatibility'])
            batch = []
            for ordinal, fields in enumerate(bounded_lines(args.bim, args.max_line_bytes), 1):
                base.require(ordinal <= args.max_bim_records and len(fields) == 6, 'BIM record limit or six-column format failed')
                chrom, variant_id, cm, bp, a1, a2 = fields
                base.require(math.isfinite(float(cm)), 'Nonfinite BIM genetic coordinate')
                pos = int(bp)
                counts['bim_all_rows'] += 1
                contigs[chrom] += 1
                base.require(len(contigs) <= 10000, 'Too many BIM contig labels')
                if chrom in mapping:
                    ref = reference.fetch(args.reference_contig, pos - 1, pos).upper() if 1 <= pos <= length else 'OUT_OF_RANGE'
                    state = fasta_status(a1, a2, ref)
                    writer.writerow([ordinal, pos, variant_id, a1, a2, ref, state])
                    bim_states[state] += 1
                    counts['bim_target_rows'] += 1
                    batch.append((ordinal, pos, variant_id, a1, a2))
                if ordinal % args.batch_rows == 0:
                    con.executemany('INSERT INTO bim VALUES (?,?,?,?,?)', batch)
                    con.commit()
                    batch.clear()
                    max_db_bytes = max(max_db_bytes, guard(args, db, started))
            con.executemany('INSERT INTO bim VALUES (?,?,?,?,?)', batch)
            con.commit()
            base.require(counts['bim_target_rows'] > 0, f'No chr{args.chromosome} rows under explicit contig mapping')
            con.execute('CREATE INDEX position_index ON bim(pos,row_no)')
            con.execute('CREATE INDEX variant_id_index ON bim(variant_id)')
            con.commit()
            max_db_bytes = max(max_db_bytes, guard(args, db, started))
            counts['bim_target_duplicate_id_excess_rows'] = con.execute(
                'SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM bim GROUP BY variant_id HAVING count(*)>1)').fetchone()[0]
            groups = itertools.groupby(con.execute('SELECT row_no,pos,variant_id,a1,a2 FROM bim ORDER BY pos,row_no'), key=lambda row: row[1])

            def next_group():
                item = next(groups, None)
                if item is None:
                    return None, []
                position, cursor = item
                rows = list(itertools.islice(cursor, args.max_rows_per_position + 1))
                base.require(len(rows) <= args.max_rows_per_position, 'BIM rows per position exceed bound')
                return position, rows

            panel_pos, panel_rows = next_group()
            previous = 0
            with base.MetadataVCF(args.rare_vcf, args.max_line_bytes) as source, gzip.open(out/'catalogue_correspondence.tsv.gz', 'wt') as target:
                base.validate_source_contract(contract, source.header, args)
                writer = csv.writer(target, delimiter='\t', lineterminator='\n')
                writer.writerow(['chromosome','position_1based','source_REF','source_ALT','counted_allele',
                    'source_rare_allele_index','source_rare_ac','source_rare_an','fasta_base','bim_rows_at_position',
                    'unordered_exact_pair_rows','complement_pair_rows','distinct_snv_pairs','unsupported_pair_rows',
                    'duplicate_pair_excess_rows','catalogue_status'])
                for fields in source:
                    chrom, pos, ref, alt = base.record_key(fields, mapping)
                    base.require(previous < pos <= length, f'Source positions must be unique, increasing and inside chr{args.chromosome}')
                    previous = pos
                    info = base.rare_info(fields)
                    selected = base.validate_rare_site(SimpleNamespace(info=info, alleles=(ref, alt)))
                    base.require(ref != alt, 'Source REF and ALT must differ')
                    base.require(info['RARE_AN'] <= 2 * args.expected_source_samples,
                                 'Source allele denominator exceeds diploid cohort')
                    fasta_base = reference.fetch(args.reference_contig, pos - 1, pos).upper()
                    base.require(fasta_base == ref, 'Source REF differs from authenticated FASTA')
                    while panel_pos is not None and panel_pos < pos:
                        panel_pos, panel_rows = next_group()
                    state, detail = correspondence(ref, alt, panel_rows if panel_pos == pos else [])
                    writer.writerow([chrom,pos,ref,alt,(ref,alt)[selected],selected,info['RARE_AC'],info['RARE_AN'],fasta_base,*detail,state])
                    states[state] += 1
                    counts['source_rows'] += 1
                    counts['source_minor_ref' if selected == 0 else 'source_minor_alt'] += 1
                    base.require(counts['source_rows'] <= args.expected_rare_sites, 'Too many source rows')
                    if counts['source_rows'] % args.batch_rows == 0:
                        max_db_bytes = max(max_db_bytes, guard(args, db, started))
                while panel_pos is not None:
                    panel_pos, panel_rows = next_group()
        base.require(counts['source_rows'] == args.expected_rare_sites and
                     counts['source_minor_ref'] == contract['counts']['rare_ref'] and
                     counts['source_minor_alt'] == contract['counts']['rare_alt'], 'Source denominators differ from contract')
        base.verify_inputs_unchanged(receipts)
        base.require(all(base.sha256(path)==code_hashes[str(path)] for path in code_paths), 'Source code changed during execution')
        max_db_bytes = max(max_db_bytes, guard(args, db, started))
        # Preserve the original chr22 result keys for existing consumers; use
        # target-specific names for every new autosome and never label it chr22.
        counts[f'bim_chr{args.chromosome}_rows'] = counts['bim_target_rows']
        counts[f'bim_chr{args.chromosome}_duplicate_id_excess_rows'] = counts['bim_target_duplicate_id_excess_rows']
        result = dict(schema='r02_lai_plink_catalogue_v1', status=STATUS, chromosome=args.chromosome,
            created_at=datetime.now(timezone.utc).isoformat(), input_files=receipts,
            reference_provenance=identity, panel_build_status='NOT_CERTIFIED_BY_BIM_OR_FASTA_COMPATIBILITY',
            sample_metadata=metadata, counts=dict(counts), catalogue_status_counts=dict(states),
            source_cohort=dict(samples=args.expected_source_samples,sha256=contract['cohort_sha256'],
                               rare_sites=args.expected_rare_sites,original_allele_count=2,frequency_denominator='called_alleles'),
            bim_fasta_status_counts=dict(bim_states), bim_contig_counts=dict(contigs),
            genotype_fields_decoded=0, bed_opened=False, allele_order_semantics='A1_A2_NOT_REF_ALT',
            genotype_quality='NOT_EVALUATED', phase='NOT_EVALUATED', absence_means='UNKNOWN_NOT_ZERO_COPIES',
            public_distribution_allowed=False, contains_sample_identifiers=False,
            limits={key: getattr(args,key) for key in ('max_bim_bytes','max_bim_records','max_line_bytes','max_sqlite_bytes',
                'sqlite_cache_mib','batch_rows','max_rows_per_position','timeout_seconds','min_free_gib')},
            resources=dict(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                max_sqlite_and_sidecar_bytes=max_db_bytes), package_versions=dict(pysam=pysam.__version__,sqlite=sqlite3.sqlite_version),
            sqlite_limit_scope='Main database and named sidecars; SQLite temporary sorts additionally guarded by live filesystem reserve and task memory/time caps',
            code_sha256={Path(path).name:digest for path,digest in code_hashes.items()},
            outputs={path.name:dict(sha256=base.sha256(path),bytes=path.stat().st_size) for path in out.glob('*.tsv.gz')})
        result['outputs_sha256'] = {name:entry['sha256'] for name,entry in result['outputs'].items()}
        with (out/'manifest.json').open('x') as handle:
            json.dump(result,handle,indent=2,allow_nan=False)
            handle.write('\n')
        return result
    finally:
        con.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in INPUTS + ('input_hashes', 'outdir'):
        p.add_argument('--'+key.replace('_','-'), required=True)
    p.add_argument('--chromosome',required=True)
    p.add_argument('--reference-contig',required=True)
    for key in ('expected_source_samples','expected_rare_sites','max_bim_bytes','max_bim_records','max_line_bytes',
                'max_sqlite_bytes','sqlite_cache_mib','batch_rows','max_rows_per_position','timeout_seconds'):
        p.add_argument('--'+key.replace('_','-'),type=int,required=True)
    p.add_argument('--min-free-gib',type=float,required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
