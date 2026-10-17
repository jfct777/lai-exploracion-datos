#!/usr/bin/env python3
"""Derive identified donor-copy overlaps from authenticated simulated mosaics.

These are positives conditional on the copied donor catalogue, not genealogical
IBD truth between donors. Unresolved copy intervals remain explicit and cannot
be counted as negatives. No observed DNABR genotype or ancestry estimate is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import heapq
import json
from pathlib import Path


FIELDS = ['sample_id', 'haplotype', 'chromosome', 'start_bp_inclusive',
          'end_bp_inclusive', 'ancestry', 'donor_id',
          'compatible_donor_copies_zero_based', 'markers', 'copy_identity_status']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def unique_overlaps(rows):
    """Sweep one donor/copy/chromosome group; retain haplotype identities.

    A heap expires intervals once; matching output is streamed. Complexity is
    O(S log S + output), without constructing the full person-pair matrix.
    Rows are (start, end, sample, haplotype). Coordinates are inclusive.
    """
    active, expiry = {}, []
    for sequence, current in enumerate(sorted(rows)):
        start, end, sample, hap = current
        while expiry and expiry[0][0] < start:
            _, old = heapq.heappop(expiry)
            active.pop(old, None)
        for other in active.values():
            _, other_end, other_sample, other_hap = other
            require((sample, hap) != (other_sample, other_hap),
                    'Overlapping intervals within one simulated haplotype')
            left, right = sorted(((sample, hap), (other_sample, other_hap)))
            yield (*left, *right, start, min(end, other_end))
        active[sequence] = current
        heapq.heappush(expiry, (end, sequence))


def run(manifest_path, manifest_sha, truth_path, out):
    require(sha(manifest_path) == manifest_sha, 'Simulation manifest changed')
    source = json.loads(Path(manifest_path).read_text())
    require(source['schema'] == 'r02_lai_simulate_output_v1' and
            source['status'] == 'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION' and
            source['genome_build'] == 'GRCh38' and
            source['verification']['truth_is_simulated_not_observed_dnabr'] is True,
            'Authenticated simulated GRCh38 source required')
    require(sha(truth_path) == source['outputs_sha256']['truth.private.tsv.gz'],
            'Truth file differs from simulation manifest')
    rows, counts, groups = [], Counter(), defaultdict(list)
    last_end, people, haplotypes = {}, set(), set()
    with gzip.open(truth_path, 'rt', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(reader.fieldnames == FIELDS, 'Truth columns differ')
        for record in reader:
            sample, hap, chrom = record['sample_id'], int(record['haplotype']), record['chromosome']
            start, end = int(record['start_bp_inclusive']), int(record['end_bp_inclusive'])
            markers = int(record['markers'])
            require(sample.startswith(source['sample_namespace'] + '__') and hap in (1, 2)
                    and chrom in source['chromosomes'] and 0 < start <= end and markers >= 0,
                    'Simulation identity/coordinate differs')
            axis = sample, hap, chrom
            require(start > last_end.get(axis, 0), 'Unordered or overlapping source blocks')
            last_end[axis] = end
            people.add(sample); haplotypes.add((sample, hap))
            status = record['copy_identity_status']; copies = record['compatible_donor_copies_zero_based']
            require(status in {'UNIQUE_BY_ALLELE_COMPATIBILITY', 'AMBIGUOUS_IDENTICAL_ALLELES',
                               'NO_OBSERVED_MARKER'}, 'Unknown copy identity state')
            if status == 'UNIQUE_BY_ALLELE_COMPATIBILITY':
                require(copies in ('0', '1') and markers > 0 and record['donor_id'] != 'UNKNOWN',
                        'Identified copy lacks supporting source')
                groups[(chrom, record['donor_id'], int(copies))].append((start, end, sample, hap))
            elif status == 'AMBIGUOUS_IDENTICAL_ALLELES':
                require(copies == '0,1' and markers > 0 and record['donor_id'] != 'UNKNOWN',
                        'Ambiguous copy schema differs')
            else:
                require(copies == '.' and markers == 0, 'Unobserved block has copy evidence')
            counts[status] += 1
            rows.append(record)
    require(len(people) == source['summary']['simulated_individuals'] and
            len(haplotypes) == source['summary']['copied_haplotypes'] and
            dict(counts) == source['verification']['blocks_by_copy_status'],
            'Source population or block counts differ')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    # Preserve the complete mask, including unknown intervals, before positives.
    with gzip.open(out / 'copy_identifiability.private.tsv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    overlap_counts, bp_counts = Counter(), Counter()
    header = ['chromosome', 'sample_a', 'haplotype_a', 'sample_b', 'haplotype_b',
              'start_bp_inclusive', 'end_bp_inclusive', 'donor_id', 'donor_copy', 'relationship']
    with gzip.open(out / 'identified_copy_overlaps.private.tsv.gz', 'wt', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t'); writer.writerow(header)
        for (chrom, donor, copy), intervals in sorted(groups.items()):
            for a, ha, b, hb, start, end in unique_overlaps(intervals):
                kind = 'WITHIN_PERSON_COPY_MATCH' if a == b else 'BETWEEN_PERSON_COPY_MATCH'
                writer.writerow([chrom, a, ha, b, hb, start, end, donor, copy, kind])
                overlap_counts[kind] += 1; bp_counts[kind] += end - start + 1
    products = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    result = dict(schema='r02_identified_donor_copy_truth_v1',
                  status='COMPLETE_CONDITIONAL_COPY_OVERLAPS_NO_DETECTOR_CALIBRATION',
                  job_id=source['job_id'], donor_split=source['donor_split'],
                  chromosomes=source['chromosomes'], people=len(people), haplotypes=len(haplotypes),
                  source_manifest_sha256=manifest_sha, source_truth_sha256=sha(truth_path),
                  code_sha256=sha(__file__), block_counts=dict(counts),
                  overlap_rows=dict(overlap_counts), overlapping_haplotype_pair_bp_sum=dict(bp_counts),
                  outputs_sha256=products, contains_sample_identifiers=True,
                  inference='Same uniquely identified copied donor and homolog at overlapping positions',
                  negative_truth_available=False, independent_replicates_claimed=False,
                  detector_calibration_completed=False, demographic_inference=False,
                  limits=['Donor copies may be genealogically related; different donor IDs are not IBD negatives.',
                          'Ambiguous and unobserved copy intervals remain unknown, not negative.',
                          'Copy identity is conditional on allele compatibility and the simulated donor catalogue.',
                          'Adjacent rows are not automatically a single genealogical tract.',
                          'Existing simulations cover their declared chromosomes/scenarios, not all DNABR demography.'])
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = run(args.manifest, args.manifest_sha256, args.truth, args.out)
    print(json.dumps({'status': value['status'], 'job_id': value['job_id'], 'overlap_rows': value['overlap_rows']}))
