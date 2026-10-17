#!/usr/bin/env python3
"""Prepare an authenticated common-only K03 FIT/SELECT checkpoint.

This bridge deliberately has no rare graph or community-profile dependency.
It preserves the complete materialized common spectrum and the established
FIT/SELECT target, weight and paired-control arrays.  EVALUATE is never read.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np

import r02_structure_increment as core
import r02_structure_k03_prepare as prepare
import r02_structure_k03_fit as fit


require = core.require
CODE_FILES = tuple(dict.fromkeys((
    'r02_structure_k03_common_fit.py', *prepare.CODE_FILES,
)))


def implementation():
    return {name: core.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def _identity(records, names, roles=('FIT', 'SELECT')):
    digest = hashlib.sha256()
    for role in roles:
        for name in names:
            value = np.asarray(records[role][name])
            digest.update(f'{role}:{name}:{value.dtype}:{value.shape}'.encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def write_common_product(output, persons, points, records, *, metadata,
                         target_audit, max_output_bytes, min_free_bytes,
                         alpha, parameters, fit_resources,
                         fit_implementation_sha256):
    """Write one legacy-compatible CONTINUOUS input and its unadmitted fit request."""
    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(), 'New common-only output required')
    require(shutil.disk_usage(out.parent).free >= min_free_bytes + max_output_bytes,
            'Common-only output disk reserve insufficient')
    out.mkdir(mode=0o700)
    prepare.features.discrete.materialize.input_plan.write_table(out/'persons.tsv', persons)
    np.savez(out/'arrays.private.npz', points=np.asarray(points), **{
        role+'_'+name: value for role, record in records.items()
        for name, value in record.items()
    })
    candidate_id = hashlib.sha256(
        ('COMMON_ONLY:'+metadata['block_id']+':'+metadata['exclusion_scope']).encode()
    ).hexdigest()
    manifest = dict(
        schema='r02_k03_fit_select_arrays_v1',
        stage='FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS',
        arm='CONTINUOUS', candidate_id=candidate_id,
        target_identity=_identity(records, ('i', 'j', 'y', 'n', 'w')),
        mask_control_identity=_identity(records, ('i', 'j', 'n', 'w', 'extra')),
        select_target_identity=_identity(records, ('i', 'j', 'y', 'n', 'w'), ('SELECT',)),
        target_audit=target_audit, common_only_profile=True,
        common_profile_purpose='REAL_FIT_NUMERICAL_CONVERGENCE_AND_RESOURCE_MEASUREMENT',
        fit_weight_sum=float(np.asarray(records['FIT']['w']).sum()),
        alpha=float(alpha),
        alpha_over_fit_weight_sum=float(alpha/np.asarray(records['FIT']['w']).sum()),
        alpha_interpretation='ABSOLUTE_INHERITED_NUMERICAL_PROFILE_POINT_NOT_HUMAN_NORMALIZED_LAMBDA',
        evaluate_targets_computed=False, alpha_selection_performed=False,
        biological_validation=False, **metadata,
    )
    manifest['outputs_sha256'] = {
        'persons.tsv': core.sha(out/'persons.tsv'),
        'arrays.private.npz': core.sha(out/'arrays.private.npz'),
    }
    core.write_json(out/'manifest.json', manifest)
    fit_request = dict(
        schema='r02_k03_fit_select_request_v1',
        evidence_scope=metadata['evidence_scope'],
        scientific_spec_sha256=metadata['scientific_spec_sha256'],
        inputs=dict(directory=str(out), manifest_sha256=core.sha(out/'manifest.json')),
        implementation_sha256=fit_implementation_sha256,
        alpha=alpha, parameters=parameters, resources=fit_resources,
    )
    core.write_json(out/'fit-request.json', fit_request)
    require(sum(p.stat().st_size for p in out.iterdir()) <= max_output_bytes and
            shutil.disk_usage(out).free >= min_free_bytes,
            'Common-only output byte/disk bound exceeded')
    return dict(
        schema='r02_k03_common_fit_bridge_v1',
        stage='COMMON_ONLY_INPUT_AND_UNADMITTED_FIT_REQUEST_COMPLETE',
        block_id=metadata['block_id'], exclusion_scope=metadata['exclusion_scope'],
        common_axes=int(np.asarray(points).shape[1]), alpha=float(alpha),
        input_manifest_sha256=core.sha(out/'manifest.json'),
        fit_request_sha256=core.sha(out/'fit-request.json'),
        fit_admission_generated=False, evaluate_targets_computed=False,
        select_used_for_model_choice=False, biological_validation=False,
    )


def run_request(request_path, request_sha256, admission_path, admission_sha256, output):
    """Authenticate real common/target inputs and create one runnable fit request."""
    os.umask(0o077)
    path = Path(request_path)
    require(path.stat().st_size <= 2**20 and core.sha(path) == request_sha256,
            'Common-only bridge request changed')
    request = core.load_json(path)
    required = {'schema','evidence_scope','scientific_spec_sha256','source_contract_sha256',
                'implementation_sha256','fit_implementation_sha256','block_id','sources',
                'weighting','alpha','parameters','fit_resources','resources'}
    require(set(request) == required and request['schema'] == 'r02_k03_common_fit_bridge_request_v1'
            and request['implementation_sha256'] == implementation()
            and request['fit_implementation_sha256'] == fit.implementation(),
            'Exact common-only request/code required')
    require(request['evidence_scope'] in {'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'}
            and request['weighting'] == 'COMPONENT_EQUAL',
            'Explicit common-only population/FIT weighting required')
    require(core.sha(admission_path) == admission_sha256, 'Common-only admission changed')
    admission = core.load_json(admission_path)
    require(admission == dict(
        schema='r02_k03_common_fit_bridge_admission_v1',
        status='COMMON_FIT_INPUT_ADMITTED', evidence_scope=request['evidence_scope'],
        scientific_spec_sha256=request['scientific_spec_sha256'],
        request_sha256=request_sha256, evaluate_allowed=False,
    ), 'Exact external common FIT-input admission required')
    limits = request['resources']
    require(set(limits) == {'max_metadata_bytes','max_input_bytes','max_array_bytes',
            'max_memory_bytes','max_target_workspace_bytes','max_output_bytes','min_free_bytes'}
            and all(type(value) is int and value > 0 for value in limits.values()),
            'Explicit positive common-only resource limits required')
    sources = request['sources']
    require(set(sources) == {'common','prepared','evaluation'},
            'Common-only bridge accepts common and target sources only')
    auth = prepare.Authenticator(limits)
    prepared = sources['prepared']
    require(set(prepared) == {'directory','contract_sha256'}, 'Exact prepared target catalogue required')
    root = Path(prepared['directory'])
    contract = core.load_json(auth.pin(root/'contract.json', prepared['contract_sha256'], metadata=True))
    evaluation_desc = sources['evaluation']
    require(set(evaluation_desc) == {'path','sha256'}, 'Pinned target evaluation configuration required')
    evaluation = core.load_json(auth.pin(evaluation_desc['path'], evaluation_desc['sha256'], metadata=True))
    require(evaluation['prepared_contract_sha256'] == prepared['contract_sha256']
            and contract['source_contract_sha256'] == request['source_contract_sha256'],
            'Common-only target source differs')
    store = core.BlockBundles(root, contract, evaluation)
    matches = [index for index, block in enumerate(store.blocks)
               if block['block_id'] == request['block_id']]
    require(len(matches) == 1, 'Exact common-only target block required')
    index = matches[0]
    shard = store.path(store.entries[index])
    receipt = core.load_json(shard/'manifest.json')
    for name, digest in receipt['outputs_sha256'].items():
        if name.endswith('.npz'):
            auth.array(shard/name, digest)
        else:
            auth.pin(shard/name, digest, metadata=True)
    people = store.persons
    allowed = validate_common_territory(
        store.blocks[index], contract['exclusion_scope'],
        common_manifest_allowed(sources['common'], auth))
    common, common_manifest = prepare.common_scores(
        sources['common'], auth, people,
        source_sha256=request['source_contract_sha256'], block_id=request['block_id'],
        scope=contract['exclusion_scope'],
        allowed=allowed,
        weighting=request['weighting'])
    values, block_contract = store.load(index)
    loaded_people, _, blocks, markers, genotypes, _, pairs = values
    records, target_audit = prepare.target_arrays(
        loaded_people, blocks[0], markers, genotypes, pairs[request['block_id']],
        min_fit_called=block_contract['min_fit_called'],
        min_pair_markers=block_contract['min_pair_markers'],
        max_pairs=block_contract['max_pairs'],
        max_workspace_bytes=limits['max_target_workspace_bytes'])
    memory_bound = 6*sum(auth.array_sizes.values()) + limits['max_target_workspace_bytes']
    require(memory_bound <= limits['max_memory_bytes'],
            'Common/target bridge workspace exceeds memory bound')
    metadata = dict(
        evidence_scope=request['evidence_scope'],
        scientific_spec_sha256=request['scientific_spec_sha256'],
        source_contract_sha256=request['source_contract_sha256'],
        block_id=request['block_id'], exclusion_scope=contract['exclusion_scope'],
        common_manifest_sha256=sources['common']['manifest_sha256'],
        common_numerical_rank=common_manifest['numerical_rank'],
        weighting=request['weighting'], implementation_sha256=implementation(),
        input_request_sha256=request_sha256, memory_bound_bytes=memory_bound,
    )
    result = write_common_product(
        output, people, common, records, metadata=metadata,
        target_audit=target_audit, max_output_bytes=limits['max_output_bytes'],
        min_free_bytes=limits['min_free_bytes'], alpha=request['alpha'],
        parameters=request['parameters'], fit_resources=request['fit_resources'],
        fit_implementation_sha256=request['fit_implementation_sha256'])
    auth.verify(); store.verify()
    require(core.sha(path) == request_sha256 and implementation() == request['implementation_sha256'],
            'Common-only bridge changed during run')
    core.write_json(Path(output)/'bridge-manifest.json', result)
    return result


def common_manifest_allowed(descriptor, auth):
    """Read the pinned exclusion territory before common_scores validates it again."""
    root = Path(descriptor['directory'])
    manifest = core.load_json(auth.pin(root/'manifest.json', descriptor['manifest_sha256'], metadata=True))
    require('allowed_input_intervals' in manifest,
            'Common manifest lacks its authenticated allowed territory')
    return manifest['allowed_input_intervals']


def validate_common_territory(block, scope, allowed):
    """Bind common axes to the target shard's real exclusion geometry."""
    require(scope in {'BLOCK_AND_LD_MARGIN', 'WHOLE_CHROMOSOME'},
            'Unknown common-only exclusion scope')
    chromosome = block['chromosome']
    target = [chromosome, int(block['start_bp']), int(block['end_bp'])]
    excluded = [chromosome, int(block['excluded_start_bp']),
                int(block['excluded_end_bp'])]
    require(0 < excluded[1] <= target[1] <= target[2] <= excluded[2],
            'Prepared target is outside its exclusion interval')
    require(isinstance(allowed, list) and allowed,
            'Common-only allowed territory is empty')
    for interval in allowed:
        require(isinstance(interval, list) and len(interval) == 3 and
                isinstance(interval[0], str) and interval[0] and
                type(interval[1]) is int and type(interval[2]) is int and
                0 < interval[1] <= interval[2],
                'Malformed common-only allowed interval')
        require(interval[0] != chromosome or interval[2] < excluded[1] or
                interval[1] > excluded[2],
                'Common-only axes overlap the target exclusion interval')
    if scope == 'WHOLE_CHROMOSOME':
        require(all(interval[0] != chromosome for interval in allowed),
                'Whole-chromosome common axes retain the target chromosome')
    else:
        require(any(interval[0] == chromosome for interval in allowed),
                'Block-and-margin common axes removed the target chromosome')
    return allowed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request','request-sha256','admission','admission-sha256','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(run_request(args.request, args.request_sha256,
                                 args.admission, args.admission_sha256,
                                 args.output)))
