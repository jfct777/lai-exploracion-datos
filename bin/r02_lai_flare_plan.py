#!/usr/bin/env python3
"""Prepare one explicit, truth-blind FLARE baseline without executing inference.

Reuses the established M34 command/map implementation, but not its chr22 or
12-generation scientific contract. Reference/target calls are streamed together;
no genotype matrix or whole marker catalogue is retained in memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from itertools import zip_longest
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import m34_run_flare as legacy
from r02_lai_role_units import role_units
import r02_lai_simulate as sim
from r02_lai_support_review import require, read, write, sha

SCHEMA = 'r02_lai_flare_plan_input_v1'
INPUTS = {'reference_vcf','reference_index','target_vcf','target_index','sample_map',
          'genetic_map','flare_jar','roles','marker_inventory','material_provenance','simulation_manifest'}
OPTIONS = legacy.PARAMETER_MEMBERS | {'panel-cm','model','gt-samples','gt-ancestries','excludemarkers'}
CODE_FILES = ('r02_lai_flare_plan.py','r02_lai_role_units.py','m34_run_flare.py','r02_lai_simulate.py','r02_lai_support_review.py')
MARKER_CONDITIONS = {
    'COMMON_ONLY': {'COMMON'},
    'COMMON_PLUS_RARE': {'COMMON','RARE'},
    'BASELINE_EXCLUDING_EVALUATED_DNABR_RARE': {'BASELINE_OTHER'},
    'BASELINE_PLUS_EVALUATED_DNABR_RARE': {'BASELINE_OTHER','EVALUATED_DNABR_RARE'},
}
OFFICIAL_JAR_SHA256 = '8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420'
FLOAT_EXPORT = 'DNABR_STORED_FLOAT32_ROUNDTRIP'


def export_inputs(config):
    return {'export_derivation','export_equivalence'} if config.get('probability_export') == FLOAT_EXPORT else set()


def expected_channels(config):
    channels=MARKER_CONDITIONS[config['marker_condition']]
    if config['marker_condition']=='BASELINE_PLUS_EVALUATED_DNABR_RARE' and config.get('evaluated_candidates_retained')==0:
        return {'BASELINE_OTHER'}
    return channels


def validate_export(config, files):
    """Bind a named formatter derivative to its limited change and real comparison."""
    label = config.get('probability_export', 'OFFICIAL_TWO_DECIMAL_EXPORT')
    require(label in {'OFFICIAL_TWO_DECIMAL_EXPORT', FLOAT_EXPORT}, 'Unknown probability export semantics')
    actual = sha(files['flare_jar'])
    if label == 'OFFICIAL_TWO_DECIMAL_EXPORT':
        require(actual == OFFICIAL_JAR_SHA256, 'Unlabelled or substituted FLARE JAR')
        return dict(label=label, jar_sha256=actual, stored_float_roundtrip=False)
    built, checked = read(files['export_derivation']), read(files['export_equivalence'])
    require(built.get('schema') == 'r02_flare_probability_export_derivative_v1' and
            checked.get('schema') == 'r02_flare_export_inference_equivalence_v1' and
            built.get('derivative_jar_sha256') == checked.get('derivative_jar_sha256') == actual and
            built.get('original_jar_sha256') == checked.get('original_jar_sha256') == OFFICIAL_JAR_SHA256,
            'Formatter provenance does not bind both exact JARs')
    require(built.get('all_other_jar_entries_byte_identical') is True and
            set(built.get('modified_entries',[])) == {'admix/AdmixRecBuilder$ProbFormatter.class','admix/AdmixRecBuilder.java'} and
            built.get('probe',{}).get('float_bits_equal_after_export_parse') is True,
            'Derivative exceeds the formatter-only change')
    require(checked.get('status') == 'SYNTHETIC_INFERENCE_HARDCALLS_MODEL_GLOBAL_IDENTICAL_AND_FLOAT_EXPORT_VERIFIED' and
            all(checked.get(k) is True for k in ('model_equal','hardcalls_and_gt_equal','global_ancestry_equal',
                'official_values_reproduced_by_official_float_quantization')) and
            checked.get('source_test_or_human_genotypes_used') is False,
            'Formatter inference equivalence was not demonstrated')
    return dict(label=label, jar_sha256=actual, original_jar_sha256=OFFICIAL_JAR_SHA256,
                stored_float_roundtrip=True, recovers_original_double=False,
                evidence_sha256={k:sha(files[k]) for k in export_inputs(config)})


def validate(config):
    require(config.get('schema') == SCHEMA, 'Explicit FLARE plan schema required')
    require(config.get('scope') in {'SYNTHETIC_TEST','CONDITIONAL_DEVELOPMENT'}, 'No confirmatory or SOURCE_TEST scope')
    require(config.get('method') == 'DIRECT_MACROANCESTRY', 'This adapter does not implement FLARE2 or fine panels')
    expected_build = 'SYNTHETIC' if config.get('scope') == 'SYNTHETIC_TEST' else 'GRCh38'
    require(config.get('genome_build') == expected_build, 'Explicit compatible assembly required')
    require(str(config.get('chromosome')) in {str(i) for i in range(1,23)}, 'Autosome1–22 required')
    require(config.get('ancestry_names') == ['AFR','EUR','NAM'], 'Explicit AFR/EUR/NAM axis required')
    require(config.get('marker_condition') in MARKER_CONDITIONS, 'Explicit marker condition required')
    require(config.get('site_filter') in {'PASS_ONLY','PASS_OR_UNFILTERED'}, 'Explicit site filter required')
    require(config.get('target_population_assumption','').strip(), 'Document the target population model assumption')
    p = config.get('parameters', {})
    require(set(p) == OPTIONS, 'Every documented FLARE option must be explicit, including unused null options')
    require(all(type(p[key]) is bool for key in ('array','probs','em','update-p','panel-probs')), 'Boolean option type mismatch')
    require(p['probs'] and not p['update-p'] and not p['panel-probs'], 'Direct probability baseline required, not FLARE2')
    require(all(p[key] is None for key in ('model','gt-samples','gt-ancestries','excludemarkers')),
            'This adapter requires explicit null for unsupported file options; materialize the exact input cohort/markers')
    require(type(p['gen']) in {int,float} and math.isfinite(p['gen']) and p['gen'] >= 1, 'gen must be finite and >=1')
    require(type(p['min-maf']) in {int,float} and math.isfinite(p['min-maf']) and 0 <= p['min-maf'] < .5, 'Invalid min-maf')
    require(type(p['min-mac']) is int and p['min-mac'] >= 0, 'Invalid min-mac')
    require(type(p['nthreads']) is int and 1 <= p['nthreads'] <= 2**31-1 and
            type(p['seed']) is int and -2**63 <= p['seed'] < 2**63, 'Threads and signed64-bit seed required')
    require(type(p['panel-cm']) in {int,float} and math.isfinite(p['panel-cm']) and p['panel-cm'] > 0, 'Invalid panel-cm')
    expected = 'ESTIMATE_PER_CHROMOSOME' if p['em'] else 'FIXED_SUPPLIED_GEN'
    require(config.get('generation_policy') == expected, 'Declare whether gen is estimated or fixed')
    require(set(config.get('inputs', {})) == INPUTS | export_inputs(config), 'Missing or extra FLARE input descriptor; truth is not accepted')
    require(config.get('flare_version') == '0.6.0', 'This adapter was checked against FLARE0.6.0 only')
    resources = config.get('resources', {})
    require(set(resources) == {'max_markers','max_samples','max_input_bytes','max_map_points','max_map_bytes',
                              'java_heap_mib','max_wall_seconds','max_output_bytes','disk_reserve_gib'},
            'Explicit resource bounds required')
    require(all(type(resources[k]) is int and resources[k] > 0 for k in resources if k != 'disk_reserve_gib'), 'Invalid positive resource bound')
    require(type(resources['disk_reserve_gib']) in {int,float} and math.isfinite(resources['disk_reserve_gib']) and resources['disk_reserve_gib'] >= 0,
            'Invalid disk reserve')
    require(resources['max_map_points'] <= 1_000_000 and resources['max_map_bytes'] <= 64*2**20,
            'Map bounds exceed this 2GiB preparation implementation; split or revise resources explicitly')
    return p, resources


def effective_min_mac(parameters, reference_samples):
    """Mirror AdmixReader.minMac: Java float multiplication then ceil, REF only."""
    as_float = lambda x: struct.unpack('f',struct.pack('f',x))[0]
    threshold = math.ceil(as_float(as_float(parameters['min-maf'])*(2*reference_samples)))
    return threshold if parameters['array'] else max(threshold,parameters['min-mac'])


def bound_map(path, resources):
    require(path.stat().st_size <= resources['max_map_bytes'], 'Uncompressed map-byte bound exceeded')
    count = 0
    with path.open() as stream:
        for line in stream:
            require(len(line) <= 4096, 'Map row exceeds bounded normalization contract')
            if line.strip() and not line.lstrip().startswith('#'): count += 1
            require(count <= resources['max_map_points'], 'Map-point bound exceeded')


def axes(files, config, resources, deadline):
    ref_iter, target_iter = sim.vcf_rows(files['reference_vcf']), sim.vcf_rows(files['target_vcf'])
    references, sentinel = next(ref_iter); require(sentinel is None, 'Missing reference header')
    targets, sentinel = next(target_iter); require(sentinel is None, 'Missing target header')
    require(references and targets and len(set(references)) == len(references) and len(set(targets)) == len(targets), 'Empty/duplicate samples')
    require(set(references).isdisjoint(targets), 'Reference and target samples overlap')
    require(len(references)+len(targets) <= resources['max_samples'], 'Sample bound exceeded')
    p = config['parameters']; require(p['array'] or p['min-mac'] < len(references), 'min-mac must be less than reference sample count')
    threshold = effective_min_mac(p,len(references))
    axis = hashlib.sha256(); previous = 0; first = None; count = 0; conditions = Counter(); filters = Counter()
    with files['marker_inventory'].open() as stream:
        catalogue = csv.DictReader(stream, delimiter='\t')
        require(set(catalogue.fieldnames or []) == {'chromosome','position','ref','alt','channel','frequency_definition'},
                'Marker inventory needs allele identity and frequency-definition provenance')
        for a,b,c in zip_longest(ref_iter,target_iter,catalogue):
            require(a is not None and b is not None and c is not None, 'REF/TARGET/inventory marker counts differ')
            samples, row = a; target_samples, other = b
            require(row is not None and other is not None and samples == references and target_samples == targets, 'Unexpected repeated VCF header')
            require(row[0] == other[0] == str(config['chromosome']), 'VCF chromosome labels differ; no silent renaming')
            pos = int(row[1]); require(pos > previous, 'Duplicate or unsorted marker')
            require((row[1],row[3],row[4]) == (other[1],other[3],other[4]), 'REF/TARGET allele axes differ')
            require(row[3] in {'A','C','G','T'} and row[4] in {'A','C','G','T'} and row[3] != row[4], 'Biallelic SNV required')
            require(other[8] == 'GT' and other[7] == '.', 'Target must contain predictor-only GT, without truth annotations')
            require((c['chromosome'],c['position'],c['ref'],c['alt']) == (row[0],row[1],row[3],row[4]), 'Marker inventory axis differs')
            require(c['channel'] in expected_channels(config) and c['frequency_definition'].strip(),
                    'Marker channel differs from the explicit evaluated catalogue/frequency definition')
            allowed = {'PASS'} if config['site_filter'] == 'PASS_ONLY' else {'PASS','.'}
            require(row[6] in allowed and other[6] in allowed, 'FILTER violates explicit policy')
            calls = sim.genotypes(row); sim.genotypes(other)  # FLARE input remains strictly phased.
            alt = sum(x+y for x,y in calls); mac = min(alt,2*len(references)-alt)
            require(mac >= threshold,
                    'A declared input marker would be removed by the supplied FLARE frequency filters')
            axis.update(f'{row[0]}\t{pos}\t{row[3]}\t{row[4]}\n'.encode())
            conditions[c['channel']] += 1; filters['REF:'+row[6]] += 1; filters['TARGET:'+other[6]] += 1
            first = pos if first is None else first; previous = pos; count += 1
            require(count <= resources['max_markers'], 'Marker bound exceeded')
            if count % 1024 == 0: require(time.monotonic() <= deadline, 'FLARE preparation deadline exceeded')
    require(count >= 2, 'FLARE needs at least two distinct marker positions per chromosome')
    require(set(conditions) == expected_channels(config), 'Declared condition needs all of its channels')
    return dict(references=references,targets=targets,first_bp=first,last_bp=previous,marker_count=count,
                marker_axis_sha256=axis.hexdigest(),channels=dict(conditions),filters=dict(filters),
                effective_min_mac_reference_only=threshold)


def prepare(path, digest, out):
    started = time.monotonic(); path = Path(path).resolve()
    config = read(sim.descriptor(dict(path=str(path),sha256=digest))); parameters, resources = validate(config)
    files = {key:sim.descriptor(entry) for key,entry in config['inputs'].items()}
    export = validate_export(config,files)
    require(sum(p.stat().st_size for p in files.values()) <= resources['max_input_bytes'], 'Input-byte bound exceeded')
    for prefix in ('reference','target'):
        require(str(files[prefix+'_vcf']).endswith('.vcf.gz') and str(files[prefix+'_index']) in
                {str(files[prefix+'_vcf'])+'.tbi',str(files[prefix+'_vcf'])+'.csi'}, 'VCF and index filenames required; binary index not validated here')
    provenance = read(files['material_provenance'])
    expected = {k:v['sha256'] for k,v in config['inputs'].items() if k != 'material_provenance'}
    require(provenance.get('schema') == 'r02_lai_flare_material_v1' and provenance.get('genome_build') == config['genome_build']
            and provenance.get('inputs_sha256') == expected, 'Material provenance must bind assembly and every supplied input')
    require(provenance.get('source_test_genotypes_used') is False and provenance.get('original_donor_units_preserved') is True,
            'No test use or redefined donor units admitted')
    source = read(files['simulation_manifest'])
    require(source.get('schema') == 'r02_lai_simulate_output_v1' and
            source.get('scope') == ('SYNTHETIC_TEST' if config['scope']=='SYNTHETIC_TEST' else 'DNABR_AUTHORIZED_SIMULATION') and
            source.get('genome_build') == config['genome_build'] and str(config['chromosome']) in source.get('chromosomes',[]),
            'Compatible simulation manifest required; no real target admitted by this adapter')
    expected_status = 'COMPLETE_SYNTHETIC_SOFTWARE_TEST' if config['scope']=='SYNTHETIC_TEST' else 'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION'
    require(source.get('status') == expected_status, 'Simulation status and scope are inconsistent or incomplete')
    require(provenance.get('simulation_manifest_sha256') == config['inputs']['simulation_manifest']['sha256'] and
            provenance.get('source_predictors_sha256') == source.get('outputs_sha256',{}).get('genotypes.vcf.gz') and
            re.fullmatch('[0-9a-f]{64}',provenance.get('source_predictors_sha256','')),
            'Material must identify the actual simulator predictor product')
    out = Path(out).resolve(); require(not out.exists() and out.parent.is_dir(), 'Fresh local output required')
    sim.local_filesystem(out.parent)
    require(sim.shutil.disk_usage(out.parent).free >= resources['disk_reserve_gib']*2**30, 'Disk reserve violated')
    data = axes(files,config,resources,started+resources['max_wall_seconds'])
    ancestry, unit_counts = role_units(files['roles'],data['references'],data['targets'],config['ancestry_names'],config['scope']=='SYNTHETIC_TEST')
    with files['sample_map'].open() as stream:
        panel = [line.split() for line in stream if line.strip() and not line.lstrip().startswith('#')]
    require(panel == [[sid,ancestry[sid]] for sid in data['references']], 'Reference panel differs from roles or order')
    out.mkdir(mode=0o700)
    panel_audit = legacy.normalize_sample_map(files['sample_map'],out/'reference-panel.tsv',data['references'],config['ancestry_names'])
    bound_map(files['genetic_map'],resources)
    map_audit = legacy.normalize_genetic_map(files['genetic_map'],out/'genetic.map',str(config['chromosome']),
                                            str(config['chromosome']),data['first_bp'],data['last_bp'])
    # Help/version only: this invocation never receives genetic inputs.
    info = subprocess.run(['java','-Xmx64m','-jar',str(files['flare_jar'])],capture_output=True,text=True,timeout=min(20,resources['max_wall_seconds']),check=True)
    header = (info.stdout+info.stderr).splitlines()[0]
    require(header.startswith('flare version '+config['flare_version']+' '), 'JAR runtime version differs')
    command = legacy.build_command('java',files['flare_jar'],files['reference_vcf'],files['target_vcf'],
                                   out/'reference-panel.tsv',out/'genetic.map',out/'inference',parameters)
    command.insert(1,f"-Xmx{resources['java_heap_mib']}m")
    # Legacy formatting uses six significant digits; preserve supplied precision.
    command = [f'{key}={parameters[key]}' if '=' in item and (key:=item.split('=',1)[0]) in {'gen','min-maf'} else item for item in command]
    command.append('panel-cm='+str(parameters['panel-cm']))
    summary = dict(schema='r02_lai_flare_plan_v1',status='PREFLIGHT_ONLY_NOT_INFERENCE_OR_SCIENTIFIC_APPROVAL',
        chromosome=str(config['chromosome']),genome_build=config['genome_build'],scope=config['scope'],method=config['method'],
        marker_condition=config['marker_condition'],parameters=parameters,generation_policy=config['generation_policy'],probability_export=export,
        generation_is_initial_value=parameters['em'],command_argv=command,flare_runtime=header,
        shape={k:data[k] for k in ('marker_count','marker_axis_sha256','channels','filters')},
        effective_min_mac_reference_only=data['effective_min_mac_reference_only'],
        reference_samples=len(data['references']),target_samples=len(data['targets']),original_units=unit_counts,
        maps=map_audit,reference_panel=panel_audit,truth_accessed=False,flare_inference_executed=False,
        genotype_quality_or_phase_accuracy_validated=False,assembly_is_authenticated_upstream_declaration=True,
        index_accessibility_validated=False,bgzf_validated=False,index_required_by_flare=False,
        simulation_manifest_sha256=provenance['simulation_manifest_sha256'],scientific_scenario_approved=False,
        input_sha256={k:v['sha256'] for k,v in config['inputs'].items()},input_contract_sha256=digest,
        limitations=['DIRECT_MACROANCESTRY_ONLY','NO_FLARE2','NO_PARAMETER_SEARCH_EXECUTED',
                     'NO_TRANSFERRED_MODEL_SUPPORT','INTERNAL_CONSTANTS_PINNED_BY_JAR_NOT_OPTIMIZED',
                     'NO_UPSTREAM_MATERIALIZATION_PERFORMED','NO_PRODUCTION_TIME_OR_COST_MEASURED'])
    write(out/'plan.json',summary)
    sim.descriptor(dict(path=str(path),sha256=digest))
    for entry in config['inputs'].values(): sim.descriptor(entry)
    sim.resource_guard(out,dict(timeout_seconds=resources['max_wall_seconds'],
        disk_reserve_gib=resources['disk_reserve_gib'],max_output_gib=resources['max_output_bytes']/2**30),started)
    write(out/'manifest.json',dict(schema='r02_lai_flare_plan_manifest_v1',status=summary['status'],
          outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
          code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES},
          contains_sample_identifiers=True,public_distribution_allowed=False,flare_inference_executed=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);prepare(args.contract,args.contract_sha256,args.output_dir)


if __name__=='__main__':main()
