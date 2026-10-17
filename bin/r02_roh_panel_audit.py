#!/usr/bin/env python3
"""Audit historical ROH panel metadata and PLINK missingness, without filtering.

Hardcall counts are observations on selected SNPs, not a genomic callability mask.
Keep BIM A1/A2 semantics; never label them REF/ALT or ancestral/minor alleles.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import math
from pathlib import Path
import sqlite3

from r02_lai_support_review import read, require, sha, write


def fam_rows(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            require(len(fields) == 6, 'FAM must contain six fields')
            rows.append(tuple(fields[:2]))
    require(rows and len(rows) == len(set(rows)), 'Empty or duplicate FID/IID roster')
    return rows


def count_record(row, expected_observations):
    missing, observations = int(row['MISSING_CT']), int(row['OBS_CT'])
    require(observations == expected_observations and 0 <= missing <= observations,
            'Missingness denominator/count differs')
    fraction = float(row['F_MISS'])
    require(math.isfinite(fraction) and math.isclose(
        fraction, missing / observations, rel_tol=5e-6, abs_tol=0),
        'Missingness fraction differs beyond six-significant-digit text rounding')
    return missing


def audit(*, bim, fam, smiss, vmiss, chromosome_sizes, output,
          expected_people, expected_sites, max_index_bytes):
    require(expected_people > 0 and expected_sites > 0 and max_index_bytes > 0,
            'Positive dimensions and index limit required')
    paths = {key: Path(value) for key, value in dict(
        bim=bim, fam=fam, smiss=smiss, vmiss=vmiss, chromosome_sizes=chromosome_sizes).items()}
    pins = {key: sha(value) for key, value in paths.items()}
    roster = fam_rows(fam)
    require(len(roster) == expected_people, 'FAM person count differs')
    sizes = read(chromosome_sizes)
    require(set(sizes) == {str(c) for c in range(1, 23)}
            and all(isinstance(v, int) and v > 0 for v in sizes.values()),
            'Explicit positive sizes for all 22 autosomes required')
    total_samples = 0
    with paths['smiss'].open() as stream:
        rows = csv.DictReader(stream, delimiter='\t')
        require(set(rows.fieldnames or []) == {'#FID', 'IID', 'MISSING_CT', 'OBS_CT', 'F_MISS'},
                'Unexpected sample missingness columns')
        for identity, row in itertools.zip_longest(roster, rows):
            require(identity is not None and row is not None, 'Sample report length differs')
            require((row['#FID'], row['IID']) == identity, 'Sample report order/identity differs')
            total_samples += count_record(row, expected_sites)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    database = output / 'audit-index.sqlite'
    counts, by_chr, total_variants, nrows = Counter(), {}, 0, 0
    previous = None
    with sqlite3.connect(database) as db, paths['bim'].open() as source, paths['vmiss'].open() as reports, (output / 'anomalies.tsv').open('x') as anomalies:
        db.execute('PRAGMA cache_size=-32768')
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID')
        db.execute('CREATE TABLE positions (chr INTEGER, pos INTEGER, PRIMARY KEY(chr,pos)) WITHOUT ROWID')
        writer = csv.writer(anomalies, delimiter='\t')
        writer.writerow(['bim_row_1based', 'chromosome', 'position_bp', 'marker_id', 'issue'])
        rows = csv.DictReader(reports, delimiter='\t')
        require(set(rows.fieldnames or []) == {'#CHROM', 'POS', 'ID', 'MISSING_CT', 'OBS_CT', 'F_MISS'},
                'Unexpected variant missingness columns')
        for line, row in itertools.zip_longest(source, rows):
            require(line is not None and row is not None, 'Variant report length differs')
            f = line.split()
            require(len(f) == 6, 'BIM must contain six fields')
            chrom, marker, cm_text, bp_text, a1, a2 = f
            require(chrom in sizes, 'Only the declared 22 autosomes are admitted')
            bp = int(bp_text)
            require((row['#CHROM'], int(row['POS']), row['ID']) == (chrom, bp, marker),
                    'Variant report order/identity differs')
            missing = count_record(row, expected_people)
            total_variants += missing
            nrows += 1
            issues = []
            if db.execute('INSERT OR IGNORE INTO ids VALUES (?)', (marker,)).rowcount == 0:
                issues.append('DUPLICATE_MARKER_ID')
            if db.execute('INSERT OR IGNORE INTO positions VALUES (?,?)', (int(chrom), bp)).rowcount == 0:
                issues.append('DUPLICATE_POSITION')
            if not 1 <= bp <= sizes[chrom]:
                issues.append('POSITION_OUTSIDE_DECLARED_CHROMOSOME')
            if previous is not None and (int(chrom), bp) < previous:
                issues.append('BIM_ORDER_DECREASES')
            if not math.isfinite(float(cm_text)):
                issues.append('NONFINITE_BIM_CM')
            if a1 not in 'ACGT' or a2 not in 'ACGT' or len(a1) != 1 or len(a2) != 1 or a1 == a2:
                issues.append('NOT_DISTINCT_CANONICAL_SNV_ALLELES')
            stats = by_chr.setdefault(chrom, dict(sites=0, missing_genotypes=0,
                min_bp=bp, max_bp=bp, max_nonnegative_adjacent_gap_bp=None))
            stats['sites'] += 1
            stats['missing_genotypes'] += missing
            stats['min_bp'], stats['max_bp'] = min(stats['min_bp'], bp), max(stats['max_bp'], bp)
            if previous is not None and previous[0] == int(chrom) and bp >= previous[1]:
                stats['max_nonnegative_adjacent_gap_bp'] = max(
                    stats['max_nonnegative_adjacent_gap_bp'] or 0, bp - previous[1])
            for issue in issues:
                counts[issue] += 1
                writer.writerow([nrows, chrom, bp, marker, issue])
            previous = (int(chrom), bp)
            if nrows % 10000 == 0:
                db.commit()
                require(database.stat().st_size <= max_index_bytes, 'Duplicate index disk limit exceeded')
        db.commit()
        require(database.stat().st_size <= max_index_bytes, 'Duplicate index disk limit exceeded')
    require(nrows == expected_sites, 'BIM/report site count differs')
    require(total_samples == total_variants, 'Sample and variant missingness sums differ')
    require(all(sha(path) == pins[name] for name, path in paths.items()), 'Input changed during audit')
    result = dict(schema='r02_historical_roh_panel_audit_v1',
        status='COMPLETE_MISSINGNESS_AND_MARKER_AUDIT_NOT_GENOMIC_CALLABILITY',
        input_sha256=pins, people=expected_people, sites=expected_sites,
        genotype_observations=expected_people * expected_sites, missing_genotypes=total_samples,
        per_chromosome=by_chr, anomalies=dict(counts), rows_filtered=0,
        genotype_matrix_loaded=False, roh_computed=False, genomic_callable_mask=False,
        allele_semantics='BIM_A1_A2_UNORIENTED_NOT_REF_ALT',
        fraction_check='Six significant digits relative rounding bound; integer counts authoritative',
        anomalies_sha256=sha(output / 'anomalies.tsv'), code_sha256=sha(__file__))
    write(output / 'summary.json', result)
    # This newly created index has no scientific payload beyond source BIM metadata.
    database.unlink()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bim', 'fam', 'smiss', 'vmiss', 'chromosome-sizes', 'output']:
        p.add_argument('--' + name, required=True)
    for name in ['expected-people', 'expected-sites', 'max-index-bytes']:
        p.add_argument('--' + name, type=int, required=True)
    print(audit(**vars(p.parse_args()))['status'])


if __name__ == '__main__':
    main()
