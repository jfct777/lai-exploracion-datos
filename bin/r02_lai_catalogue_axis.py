#!/usr/bin/env python3
"""Reuse complete M02.1 metadata to classify the retained LAI reference axis.

No genotype arrays, ancestry truth or predictions are opened. Same-position
allele discrepancies are reported separately, never silently called non-rare.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import shutil
import time
import r02_lai_individual_reference
import r02_lai_source_rare_catalogue

import numpy as np

from r02_lai_individual_reference import IndividualReferenceArchive, _chromosome
from r02_lai_source_rare_catalogue import require, sha256


def pinned(pin):
    path = Path(pin['path'])
    require(path.is_file() and sha256(path) == pin['sha256'], 'Input SHA differs: ' + str(path))
    return path


def reference_keys(archive, chrom):
    manifest = archive._manifest(chrom)
    require(manifest['people'] == 753, 'Reference people count differs')
    keys, previous = {}, 0
    for row in manifest['chunks']:
        raw = archive._member_bytes(f"published/chr{chrom}/{row['path']}")
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                'Reference chunk identity differs')
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            arrays = [data[name] for name in ('chromosome', 'position_bp', 'ref', 'alt')]
            require(all(a.ndim == 1 and len(a) == row['loci'] for a in arrays),
                    'Reference metadata dimensions differ')
            require(np.issubdtype(arrays[1].dtype, np.integer), 'Non-integer reference position')
            for c, p, ref, alt in zip(*arrays):
                p, ref, alt = int(p), str(ref), str(alt)
                require(_chromosome(c) == chrom and p > previous,
                        'Reference chromosome/order/uniqueness differs')
                require(len(ref) == len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT'
                        and ref != alt, 'Reference allele is not a distinct biallelic SNV')
                keys[p] = (chrom, p, ref, alt)
                previous = p
    require(len(keys) == manifest['source_rows'], 'Reference source count differs')
    return keys, manifest


def intersect(table, chrom, keys, expected, output):
    counts, overlap, discordant = Counter(), Counter(), []
    previous = 0
    with gzip.open(table, 'rt', newline='') as stream, gzip.open(output, 'wt', newline='') as dest:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = {'chromosome', 'position_1based', 'source_REF', 'source_ALT',
                  'counted_allele', 'source_rare_allele_index'}
        require(fields <= set(reader.fieldnames or ()), 'Source metadata fields differ')
        writer = csv.writer(dest, delimiter='\t', lineterminator='\n')
        writer.writerow(['chromosome', 'position_bp', 'REF', 'ALT', 'source_minor'])
        for row in reader:
            c, p = _chromosome(row['chromosome']), int(row['position_1based'])
            ref, alt, i = row['source_REF'], row['source_ALT'], row['source_rare_allele_index']
            require(c == chrom and p > previous, 'Source chromosome/order/uniqueness differs')
            require(len(ref) == len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT'
                    and ref != alt and i in ('0', '1'), 'Invalid source allele')
            require(row['counted_allele'] == (ref, alt)[int(i)], 'Source counted allele differs')
            side = ('REF', 'ALT')[int(i)]
            counts['total'] += 1
            counts[side] += 1
            previous = p
            if p not in keys:
                continue
            key = (c, p, ref, alt)
            if keys[p] != key:
                discordant.append({'source': key, 'reference': keys[p]})
                continue
            overlap['total'] += 1
            overlap[side] += 1
            writer.writerow([*key, side])
    require({k: counts[k] for k in ('total', 'REF', 'ALT')} == expected,
            'Complete source metadata counts differ')
    return {k: counts[k] for k in ('total', 'REF', 'ALT')}, {
        k: overlap[k] for k in ('total', 'REF', 'ALT')}, discordant


def run(request_path, request_hash, output):
    start = time.monotonic()
    request = json.loads(pinned({'path': request_path, 'sha256': request_hash}).read_text())
    require(request['schema'] == 'r02_lai_catalogue_axis_request_v1', 'Request schema differs')
    chrom = _chromosome(request['chromosome'])
    meta = json.loads(pinned(request['catalogue_manifest']).read_text())
    source = json.loads(pinned(request['rare_evidence_manifest']).read_text())
    contract_path = pinned(request['source_contract'])
    contract = json.loads(contract_path.read_text())
    reference_origin_path = pinned(request['reference_origin_manifest'])
    reference_origin = json.loads(reference_origin_path.read_text())
    table = pinned(request['catalogue_table'])
    watched = [Path(request_path), table, contract_path, reference_origin_path,
               Path(request['catalogue_manifest']['path']),
               Path(request['rare_evidence_manifest']['path']),
               Path(request['reference_archive']['path'])]
    def signature(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns
    before = {str(p): signature(p) for p in watched}
    code = [Path(__file__).resolve(), Path(r02_lai_individual_reference.__file__).resolve(),
            Path(r02_lai_source_rare_catalogue.__file__).resolve()]
    code_hashes = {p.name: sha256(p) for p in code}
    expected = {'total': source['counts']['sites'], 'REF': source['counts']['source_minor_ref'],
                'ALT': source['counts']['source_minor_alt']}
    require(meta['schema'] == 'r02_lai_plink_catalogue_v1'
            and meta['status'] == 'COMPLETE_CATALOGUE_ONLY_NO_GENOTYPES_NO_BUILD_CERTIFICATION'
            and _chromosome(meta['chromosome']) == chrom
            and meta['source_cohort']['samples'] == source['source_cohort_n'] == 2723
            and source['chromosomes'] == [chrom], 'Source manifest scope differs')
    require(meta['reference_provenance']['assembly'] == 'GRCh38'
            and meta['input_files']['rare_vcf']['sha256'] == source['input_vcf_sha256']
            and meta['input_files']['rare_contract']['sha256'] == request['source_contract']['sha256']
            and meta['outputs']['catalogue_correspondence.tsv.gz']['sha256'] == request['catalogue_table']['sha256'],
            'Source provenance linkage differs')
    require(contract['schema'] == 'm02_1_minor_v1' and _chromosome(contract['chromosome']) == chrom
            and contract['criteria']['GT_REF_ALT_preserved'] is True
            and contract['counts']['rare'] == meta['counts']['source_rows'] == expected['total']
            and contract['counts']['rare_ref'] == meta['counts']['source_minor_ref'] == expected['REF']
            and contract['counts']['rare_alt'] == meta['counts']['source_minor_alt'] == expected['ALT'],
            'Source contract/counts differ')
    out = Path(output)
    require(not out.exists(), 'Output exists; reuse a completed Nextflow task')
    require(shutil.disk_usage(out.parent).free >= request['min_free_bytes'], 'Insufficient output reserve')
    out.mkdir()
    archive_pin = request['reference_archive']
    with IndividualReferenceArchive(archive_pin['path'], archive_pin['sha256'], 1) as archive:
        keys, manifest = reference_keys(archive, chrom)
        require(reference_origin['schema'] == 'r02_lai_real_material_v1'
                and reference_origin['genome_build'] == 'GRCh38'
                and _chromosome(reference_origin['chromosome']) == chrom
                and reference_origin['outputs_sha256'][f'ref_train.chr{chrom}.vcf.gz']
                    == manifest['source_vcf_sha256']
                and reference_origin['role_counts']['REF_TRAIN'] == manifest['people']
                and reference_origin['retained_records'] == manifest['source_rows']
                    == sum(row['loci'] for row in manifest['chunks']),
                'Reference origin assembly, VCF hash, people or retained locus count differs')
        full, overlap, discordant = intersect(table, chrom, keys, expected, out/'source-rare-present.tsv.gz')
        axis_hash = hashlib.sha256()
        for key in keys.values():
            axis_hash.update(('\t'.join(map(str, key)) + '\n').encode())
        reference_manifest_sha = archive._inventory[chrom]['manifest_sha256']
    require(all(signature(p) == before[str(p)] for p in watched), 'Input changed during classification')
    require(all(sha256(p) == code_hashes[p.name] for p in code), 'Code changed during classification')
    result = dict(schema='r02_lai_catalogue_axis_receipt_v1', status='COMPLETE_METADATA_INTERSECTION',
        chromosome=chrom, request_sha256=request_hash, source_original=full, source_present=overlap,
        source_absent_exact_key=full['total']-overlap['total'], reference_loci=len(keys),
        source_position_absent=full['total']-overlap['total']-len(discordant),
        source_position_present_alleles_different=len(discordant),
        reference_non_catalogue_exact_keys=len(keys)-overlap['total'],
        reference_loci_without_source_position=len(keys)-overlap['total']-len(discordant),
        same_position_allele_discrepancies=discordant,
        reference_axis_sha256=axis_hash.hexdigest(), reference_manifest_sha256=reference_manifest_sha,
        reference_source_vcf_sha256=manifest['source_vcf_sha256'], reference_people=manifest['people'],
        reference_build_recorded='GRCh38', reference_origin_manifest_sha256=request['reference_origin_manifest']['sha256'],
        source_original_vcf_sha256=source['input_vcf_sha256'],
        source_producer_recorded_hashes=meta['code_sha256'], source_original_vcf_reread=False,
        executed_code_sha256=code_hashes,
        outputs_sha256={'source-rare-present.tsv.gz': sha256(out/'source-rare-present.tsv.gz')},
        arrays_loaded=['chromosome','position_bp','ref','alt'], genotype_arrays_loaded=False,
        scope='REFERENCE_LOCUS_AXIS_ONLY_UNDER_RECORDED_COORDINATES',
        callability_verified=False, target_axis_verified=False, training_admitted=False,
        limit='Exact keys do not certify observed genotypes, phase, donor independence or biological effect; absence is unknown, not 0/0.',
        elapsed_seconds=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    encoded = (json.dumps(result, indent=2)+'\n').encode()
    require(sum(p.stat().st_size for p in out.iterdir()) + len(encoded)
            <= request['max_output_bytes'], 'Output budget exceeded')
    temporary = out/'receipt.tmp'
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.rename(out/'receipt.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.request, args.sha256, args.output)))
