#!/usr/bin/env python3
"""Exact common discordance bounds conditional on authenticated Gnomix hardcalls.

The outcome is defined on the entire supplied FIT-selected chromosome catalogue.
Missing calls remain missing. This produces pairwise identification bounds, not
confidence intervals, and neither fits a predictor nor imputes phase.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time
import numpy as np
import r02_structure_common_inputs as common
import r02_rare_ancestry_pair_bounds as rare
import r02_k04_identified_comparison as identified

SCHEMA = 'r02_k04_common_target_input_v1'
OUTPUT = 'r02_k04_common_target_v1'
require = common.require
sha = common.shared.sha


def implementation():
    names = set(common.CODE) | set(rare.CODE_FILES) | {
        'r02_k04_common_target.py', 'r02_k04_identified_comparison.py'}
    return {n: sha(Path(__file__).with_name(n)) for n in sorted(names)}


def exact_counts(dosage, observed, h0, h1, ancestry):
    """Bilinear endpoint extrema via exact integer-valued float64 products.

    Ambiguity is possible only for D=A=1. For nonambiguous partners the
    incremental coefficient is A-2L; two ambiguous partners contribute [0,1].
    Integer products/sums are exact below 2**53. No response midpoint is used.
    Diagonal is removed: a person is never their own comparison pair.
    """
    require(dosage.shape == observed.shape == h0.shape == h1.shape,
            'Dosage/observation/hardcall dimensions differ')
    require(dosage.dtype == np.uint8 and observed.dtype == np.bool_ and
            np.all(dosage[observed] <= 2) and np.all(dosage[~observed] == 255),
            'Diploid dosage or missing sentinel differs')
    require(np.isin(h0, (-1, 0, 1, 2)).all() and np.isin(h1, (-1, 0, 1, 2)).all(),
            'Ancestry hardcall code differs')
    valid = observed & (h0 >= 0) & (h1 >= 0)
    a = ((h0 == ancestry).astype(np.int16) + (h1 == ancestry)) * valid
    d = np.where(valid, dosage, 0).astype(np.int16)
    lower = np.maximum(0, d + a - 2).astype(np.float64)
    ambiguous = ((a == 1) & (d == 1)).astype(np.float64)
    a = a.astype(np.float64)
    z = (a - 2 * lower) * (1 - ambiguous)
    neg, pos = np.minimum(z, 0), np.maximum(z, 0)
    cross = lower @ a.T
    base = cross + cross.T - 2 * (lower @ lower.T)
    correction = ambiguous @ neg.T
    lo = base + correction + correction.T
    correction = ambiguous @ pos.T
    hi = base + correction + correction.T + ambiguous @ ambiguous.T
    denominator = a @ a.T
    for matrix in (lo, hi, denominator):
        np.fill_diagonal(matrix, 0)
        require(np.all(matrix == np.rint(matrix)), 'Inexact count arithmetic')
    require(np.all((0 <= lo) & (lo <= hi) & (hi <= denominator)),
            'Mismatch numerator bounds inconsistent')
    return lo.astype(np.uint64), hi.astype(np.uint64), denominator.astype(np.uint64)


def preflight_selected(view, starts, ends):
    """Read locus metadata only; reject missing, repeated, or mismatched markers.

    This pass completes across ALL chunks before the shared reader opens any GT.
    MSP has coordinates but no alleles; REF/ALT is checked against the frozen
    common catalogue, not falsely claimed to be checked against MSP.
    """
    seen, ordered, covered = set(), [], 0
    chrom = view.task['chromosome']
    for c, path, desc in view.chunks:
        require(c == chrom, 'Common chromosome differs')
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == common.continuous.EVENT_FIELDS, 'Chunk fields differ')
            p = data['position_bp']
            require(p.dtype == np.int64 and p.shape == (desc['n_sites'],) and
                    np.all(np.diff(p) > 0) and p[0] == desc['start_bp'] and p[-1] == desc['end_bp'],
                    'Common coordinates differ before genotype access')
            ref, alt, counted, frequency = (data[k] for k in ('ref','alt','counted_allele','frequency'))
            require(all(x.shape == p.shape for x in (ref, alt, counted, frequency)),
                    'Common allele metadata shape differs')
            chosen = []
            for i, coordinate in enumerate(p):
                coordinate = int(coordinate)
                if coordinate not in view.selected[chrom]:
                    continue
                row = view.selected[chrom][coordinate]
                require(coordinate not in seen and ref[i] == 'ACGT'.index(row['ref']) and
                        alt[i] == 'ACGT'.index(row['alt']) and counted[i] == 1 and
                        np.isfinite(frequency[i]) and abs(frequency[i]-row['fit_alt_af']) <= 1e-12,
                        'Duplicate locus or REF/ALT/counting/FIT frequency mismatch before GT')
                seen.add(coordinate); chosen.append(coordinate)
            require(chosen, 'Unexpected empty common chunk')
            idx = rare.locus_windows(np.asarray(chosen), starts, ends)
            covered += int(np.count_nonzero(idx >= 0))
            ordered.append(idx)
    require(seen == set(view.selected[chrom]), 'Selected common catalogue incomplete before GT')
    return ordered, dict(n_loci=len(seen), loci_in_msp=covered,
                        loci_outside_msp=len(seen)-covered, genotype_arrays_read=False,
                        allele_check='COMMON_CATALOGUE_VERSUS_GT_METADATA_MSP_HAS_NO_ALLELES')


class TargetAccumulator:
    def __init__(self, n, indexes, hard, *, max_memory_bytes, max_chunk_sites):
        # Nine persistent uint64 matrices; at most 16 dense float64 temporaries,
        # four working genotype/hardcall blocks plus BLAS margin.
        bound = 8 * 25 * n * n + 96 * n * max_chunk_sites
        require(bound <= max_memory_bytes, 'Exact pair accumulator memory admission failed')
        self.n, self.indexes, self.hard = n, indexes, hard
        self.counts = {a: [np.zeros((n,n), np.uint64) for _ in range(3)]
                       for a in identified.ANCESTRIES}
        self.cursor = self.n_sites = 0
        self.person_joint = np.zeros(n, np.uint64)
        self.person_observed = np.zeros(n, np.uint64)
        self.memory_bound_bytes = bound

    def add(self, dosage, observed, frequency):
        require(self.cursor < len(self.indexes), 'Too many genotype chunks')
        idx = self.indexes[self.cursor]
        require(dosage.shape == (self.n,len(idx)), 'Metadata/GT selected ordering differs')
        self.cursor += 1; self.n_sites += len(idx)
        require(self.n_sites * 4 < 2**53, 'Exact integer accumulation range exceeded')
        h0,h1 = rare.hard_for_loci(idx, self.hard, self.n)
        self.person_observed += observed.sum(axis=1, dtype=np.uint64)
        self.person_joint += (observed & (h0>=0) & (h1>=0)).sum(axis=1, dtype=np.uint64)
        for ancestry, code in enumerate(identified.ANCESTRIES):
            for dst, values in zip(self.counts[code], exact_counts(dosage,observed,h0,h1,ancestry)):
                dst += values


def run(request, digest, output, scratch):
    started = time.monotonic()
    request = common.continuous.pinned(Path(request).resolve(), digest, 4*2**20)
    r = common.shared.load_json(request)
    require(r['schema'] == SCHEMA and r['implementation_sha256'] == implementation(),
            'Target specification/code differs')
    require(r['operation'] == 'DESCRIPTIVE_CONDITIONAL_TARGET_ONLY' and
            r['catalogue_scope'] == 'ALL_FIT_SELECTED_WHOLE_CHROMOSOME_COMMON_LOCI',
            'Explicit conditional outcome/catalogue scope required')
    view = common.MaterializedCommonInput(r['bundle'], r['common_request']['path'],
                                         r['common_request']['sha256'])
    require(view.task['intervals'] == [[1,view.source['sources'][view.task['chromosome']]['length_bp']]],
            'Whole chromosome common request required')
    pins = {}
    for name, desc in r['ancestry'].items():
        path = common.continuous.pinned(Path(desc['path']).resolve(),desc['sha256'],r['resources']['max_ancestry_bytes'])
        pins[name] = path
    require(set(pins) == {'ancestry_contract','msp','fb','crosswalk'}, 'Exact ancestry inventory required')
    anc = common.shared.load_json(pins['ancestry_contract'])
    samples = [p['sample_id'] for p in view.persons]
    require(anc['genome_build'] == view.source['genome_build'] == 'GRCh38' and
            anc['coordinate_system'] == '1-based-inclusive' and
            str(anc['chromosome']) == view.task['chromosome'] and
            anc['analytical_samples_sha256'] == common.material.evidence.sample_hash(samples) and
            anc['population_order'] == list(rare.core.ancestry.POPULATIONS),
            'Reference/chromosome/coordinate/person/ancestry mismatch before GT')
    for key in ('msp','fb','crosswalk'):
        require(anc[key]['sha256'] == sha(pins[key]), 'Ancestry binding differs: '+key)
    with pins['crosswalk'].open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    mapping = {p['source_id']:p['sample_id'] for p in rows}
    require(len(mapping) == len(rows) == len(samples) and len(set(mapping.values())) == len(samples)
            and set(mapping.values()) == set(samples), 'Person crosswalk not bijective')
    out, scratch = Path(output), Path(scratch)
    require(not out.exists() and scratch.is_dir(), 'Fresh output and existing scratch required')
    limits = r['resources']
    require(shutil.disk_usage(scratch).free >= limits['min_free_bytes'] + limits['max_output_bytes'],
            'Insufficient local disk reserve')
    out.mkdir(parents=True,mode=0o700)
    with tempfile.TemporaryDirectory(prefix='k04-common-',dir=scratch) as temp:
        windows, hard, hard_meta = rare.prepare_hard_windows(pins,anc,samples,mapping,Path(temp),
                        limits['max_window_matrix_mb'],max(1,limits['min_free_bytes']//2**20))
        starts,ends = (np.asarray([w[i] for w in windows]) for i in (0,1))
        indexes, coordinate = preflight_selected(view, starts, ends)
        common.material.write_json(out/'coordinate-preflight.json',coordinate)
        require(coordinate['loci_in_msp'] > 0, 'NO_COORDINATE_SUPPORT: no GT opened')
        accumulator = TargetAccumulator(len(samples),indexes,hard,
            max_memory_bytes=limits['max_memory_bytes'],max_chunk_sites=view.r['resources']['max_chunk_sites'])
        common.continuous.accumulate_selected_chunks(view.persons,view.roles['FIT'],view.selected,
            view.chunks,accumulator,min_fit_called=max(view.source['common']['min_fit_called'],
            int(np.ceil(len(view.roles['FIT'])*(1-view.source['common']['max_fit_missing'])))))
        require(accumulator.cursor == len(indexes) and accumulator.n_sites == coordinate['n_loci'],
                'Common target coverage differs')
        view.verify()
        for key,path in pins.items():
            require(sha(path) == r['ancestry'][key]['sha256'], 'Ancestry source changed')
        arrays = {f'{kind}_{a}':value for a,values in accumulator.counts.items()
                  for kind,value in zip(('lower_count','upper_count','opportunities'),values)}
        arrays['person_joint_loci'] = accumulator.person_joint
        arrays['person_observed_loci'] = accumulator.person_observed
        np.savez(out/'counts.npz',**arrays)
        require((out/'counts.npz').stat().st_size <= limits['max_output_bytes'], 'Output bound exceeded')
        with (out/'persons.tsv').open('x') as stream:
            writer=csv.writer(stream,delimiter='\t');writer.writerow(['person_index','sample_id','role'])
            writer.writerows((i,p['sample_id'],p['role']) for i,p in enumerate(view.persons))
        summary = {a:dict(eligible_unordered_pairs=int(np.count_nonzero(np.triu(v[2],1))),
                         ambiguous_unordered_pairs=int(np.count_nonzero(np.triu(v[1]>v[0],1))))
                   for a,v in accumulator.counts.items()}
        manifest = dict(schema=OUTPUT,status='COMPLETE_CONDITIONAL_COMMON_TARGET_NOT_MODEL_COMPARISON',
            request_sha256=digest,implementation_sha256=implementation(),chromosome=view.task['chromosome'],
            n_people=len(samples),role_counts=dict(Counter(p['role'] for p in view.persons)),
            n_loci=accumulator.n_sites,catalogue_scope=r['catalogue_scope'],
            common_selection=view.source['common'],source_common_request=r['common_request'],
            ancestry_hardcalls=hard_meta,coordinate_preflight=coordinate,by_ancestry=summary,
            memory_bound_bytes=accumulator.memory_bound_bytes,elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            phase_imputed=False,ancestry_truth_calibrated=False,biological_comparison=False,
            self_pairs='EXCLUDED_ZERO_OPPORTUNITIES',zero_denominator='NOT_ESTIMABLE',
            bounds='PAIRWISE_OUTER_NOT_JOINTLY_ATTAINABLE_NOT_CONFIDENCE_INTERVAL',
            outputs_sha256={n:sha(out/n) for n in ('counts.npz','persons.tsv','coordinate-preflight.json')})
        require(sha(request)==digest and implementation()==r['implementation_sha256'], 'Code/request changed')
        common.material.write_json(out/'manifest.json',manifest)
        return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--scratch',required=True)
    a=p.parse_args();run(a.request,a.request_sha256,a.output,a.scratch)


if __name__ == '__main__':
    main()
