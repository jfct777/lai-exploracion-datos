#!/usr/bin/env python3
"""Bounded, read-only validation of R02 sequential preprocessing counts."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct


SCHEMA = "r02_preprocess_record_count_validation_v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tabix_record_statistics(path, chrom):
    """Inspect optional TBI counts, not genotypes or index-query correctness.

    Older valid TBI files can omit htslib's metadata bin 37450. In that case
    hts_idx_get_stat returns -1 and zero counters: zero is not a measured count.
    See samtools/hts-specs tabix.pdf and htslib hts.c:hts_idx_get_stat. This
    bounded reader has no genomics dependency in the controller environment.
    """
    path = Path(path)
    limit = 64 * 1024**2
    if path.stat().st_size > limit:
        raise ValueError('TBI exceeds bounded metadata inspection limit')
    checksum = sha(path)
    with gzip.open(path, 'rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit or data[:4] != b'TBI\x01':
        raise ValueError('Invalid or oversized TBI metadata')
    offset = 4

    def take(size):
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise ValueError('Truncated or invalid TBI metadata')
        value = data[offset:offset + size]
        offset += size
        return value

    def number(fmt):
        return struct.unpack('<' + fmt, take(struct.calcsize('<' + fmt)))

    nref, fmt, col_seq, col_beg, col_end, meta, skip, names_size = number('8i')
    names_raw = take(names_size)
    if (nref != 1 or (fmt, col_seq, col_beg, col_end, meta, skip) != (2, 1, 2, 0, 35, 0)
            or names_raw not in (f'chr{chrom}\0'.encode(), f'{chrom}\0'.encode())):
        raise ValueError('TBI is not the expected single-chromosome VCF index')
    nbins, = number('i')
    if not 0 <= nbins <= 37450:
        raise ValueError('Invalid TBI bin count')
    seen = set()
    mapped = unmapped = None
    chunks = 0
    for _ in range(nbins):
        bid, nchunks = number('Ii')
        if bid in seen or bid > 37450 or nchunks < 0:
            raise ValueError('Invalid or duplicate TBI bin')
        seen.add(bid)
        values = take(nchunks * 16)
        if bid == 37450:
            if nchunks != 2:
                raise ValueError('Invalid TBI record-statistics metadata bin')
            mapped, unmapped = struct.unpack_from('<QQ', values, 16)
        else:
            chunks += nchunks
    nintervals, = number('i')
    take(nintervals * 8)
    if len(data) - offset not in (0, 8):
        raise ValueError('Unexpected trailing TBI metadata')
    unplaced = number('Q')[0] if len(data) - offset == 8 else 0
    if sha(path) != checksum:
        raise ValueError('TBI changed during metadata inspection')
    return dict(path=str(path), sha256=checksum, record_statistics_present=mapped is not None,
                mapped_records=mapped, unmapped_records=unmapped, unplaced_records=unplaced,
                data_bins=len(seen - {37450}), data_chunks=chunks, linear_intervals=nintervals)


def validate_raw_record_count(folder, chrom, *, source_index=None):
    """Keep genuine index mismatches fatal; label absent legacy counts unknown.

    No source index, historical count table or frozen task output is rewritten.
    Only the zero-without-statistics case can use a successful sequential M01
    count, corroborated by normalization's independent input count.
    """
    folder = Path(folder)
    counts=folder/'preprocess/02_filter'/f'dnabr.hg38.2723.chr{chrom}.counts.tsv'
    with counts.open() as handle:
        rows=list(csv.DictReader(handle, delimiter='\t'))
    raw=[int(row['n_variants']) for row in rows if row['step']=='raw' and row['chr']==str(chrom)]
    trace = folder/'trace.tsv'
    with trace.open() as handle:
        tasks = list(csv.DictReader(handle, delimiter='\t'))

    def completed_task(processes):
        selected = [row for row in tasks if row['name'].rsplit(':', 1)[-1]
                    in {f'{name} (chr{chrom})' for name in processes}]
        if len(selected) != 1 or selected[0]['status'] not in ('CACHED', 'COMPLETED') or selected[0]['exit'] != '0':
            raise ValueError(f'chr{chrom}: expected one successful M01 task in trace')
        task_hash = selected[0]['hash']
        if not re.fullmatch(r'[0-9a-f]{2}/[0-9a-f]{6,30}', task_hash):
            raise ValueError('Invalid Nextflow task hash')
        directories = list((folder/'work').glob(task_hash + '*'))
        if len(directories) != 1 or (directories[0]/'.exitcode').read_text().strip() != '0':
            raise ValueError(f'chr{chrom}: M01 task directory is ambiguous or not successful')
        return directories[0]

    annotation = completed_task({'PREPROCESS_NORM_LEFTALIGN', 'ANNOTATE_ORIGINAL_ALLELES'})
    log = annotation/'.command.out'
    matches = re.findall(r'^Annotated (\d+) records at (\d+) supplied-input sites$', log.read_text(), re.MULTILINE)
    if len(raw) != 1 or raw[0] < 0 or len(matches) != 1:
        raise ValueError(f'chr{chrom}: missing or ambiguous raw/M01 record count')
    records, sites = map(int, matches[0])
    if not 0 < sites <= records:
        raise ValueError(f'chr{chrom}: invalid sequential M01 record/site count')
    result = dict(schema=SCHEMA, raw_index_records=raw[0], raw_records=records,
                  m01_records=records, m01_original_sites=sites, counts_sha256=sha(counts),
                  trace_sha256=sha(trace), annotation_log_sha256=sha(log),
                  raw_record_count_source='source_index_and_sequential_m01')
    if raw[0] == records:
        return result
    if raw[0] != 0:
        raise ValueError(f'chr{chrom}: source index count does not match completed sequential M01 annotation')
    if source_index is None:
        parameters = json.loads((folder/'parameters.json').read_text())
        source_index = Path(parameters['r02_raw_vcf'] + '.tbi')
    staged_index = annotation/f'dnabr.hg38.2723.chr{chrom}.vcf.gz.tbi'
    if staged_index.resolve(strict=True) != Path(source_index).resolve(strict=True):
        raise ValueError(f'chr{chrom}: source index differs from the successful M01 input')
    index = tabix_record_statistics(source_index, chrom)
    if (index['record_statistics_present'] or index['unplaced_records'] != 0
            or index['data_bins'] == 0 or index['data_chunks'] == 0 or index['linear_intervals'] == 0):
        raise ValueError(f'chr{chrom}: zero source count is not explained by absent TBI statistics')
    normalization = completed_task({'PREPROCESS_NORM_LEFTALIGN', 'NORMALIZE_ANNOTATED_ALLELES'})
    norm_log = normalization/f'dnabr.hg38.2723.chr{chrom}.norm.log'
    norm_counts = re.findall(r'^Lines\s+total/split/realigned/skipped:\s*(\d+)/(\d+)/(\d+)/(\d+)\s*$',
                             norm_log.read_text(), re.MULTILINE)
    if len(norm_counts) != 1 or int(norm_counts[0][0]) != records or int(norm_counts[0][3]) != 0:
        raise ValueError(f'chr{chrom}: normalization does not corroborate sequential M01 count')
    result.update(raw_record_count_source='sequential_m01_with_missing_tbi_statistics',
                  source_index=index, normalization_log_sha256=sha(norm_log),
                  normalization_input_records=int(norm_counts[0][0]),
                  normalization_skipped_records=int(norm_counts[0][3]))
    return result
