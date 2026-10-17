#!/usr/bin/env python3
"""Authenticate FIT search policy and resource amendments before using labels.

This is a provenance gate, not a statistical claim. A larger budget is acceptable
only with its pinned amendment; an incomplete profile is never made admissible.
"""
from pathlib import Path
import copy
import json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate(manifest, manifest_sha256, descriptor, auth):
    """Use the caller's bounded hash authenticator for every provenance file."""
    require(isinstance(descriptor, dict) and set(descriptor) in (
        {'request', 'admission'}, {'request', 'admission', 'amendment'}),
        'Pinned original profile request/admission required')

    def read(ref):
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'},
                'Exact profile provenance reference required')
        return json.loads(Path(auth.pin(ref['path'], ref['sha256'], metadata=True)).read_text())

    request, admission = read(descriptor['request']), read(descriptor['admission'])
    request_sha, admission_sha = descriptor['request']['sha256'], descriptor['admission']['sha256']
    require(request.get('schema') == 'r02_fit_profile_queue_request_v1'
            and admission.get('schema') == 'r02_fit_profile_queue_admission_v1'
            and admission.get('status') == 'FIT_PROFILE_POLICY_ADMITTED'
            and admission.get('evaluate_allowed') is False
            and admission.get('request_sha256') == request_sha
            and manifest.get('request_sha256') == request_sha
            and manifest.get('admission_sha256') == admission_sha,
            'Profile parent request/admission identity differs')
    require(request['scientific_spec_sha256'] == admission['scientific_spec_sha256']
            and request['evidence_scope'] == admission['evidence_scope'] == manifest['evidence_scope']
            and request['graph'] == manifest['graph']
            and request['implementation_sha256'] == manifest['implementation_sha256'],
            'Profile scientific source, graph or implementation differs')
    require(manifest.get('profile_complete') is True
            and manifest.get('seed_coverage_complete') is True
            and manifest.get('status') == 'NO_NEW_ENVELOPE_IMPROVEMENT_UNDER_DECLARED_ORACLE'
            and manifest.get('query_edges_used') is False
            and manifest.get('evaluate_scored') is False,
            'Incomplete or outcome-dependent profile is not admissible')
    expected = copy.deepcopy(request['search'])
    if 'amendment' not in descriptor:
        require(manifest['seed_policy'] == expected, 'Changed profile policy needs an amendment')
        return {'amended': False, 'request_sha256': request_sha}

    refs = descriptor['amendment']
    require(isinstance(refs, dict) and set(refs) == {'request', 'admission', 'envelope'},
            'Exact amendment request/admission/envelope required')
    change_request, approval, envelope = (read(refs[k]) for k in ('request', 'admission', 'envelope'))
    change_sha, approval_sha = refs['request']['sha256'], refs['admission']['sha256']
    require(change_request.get('schema') == 'r02_k03_profile_seed_budget_amendment_v1'
            and approval == dict(schema='r02_k03_profile_seed_budget_amendment_admission_v1',
                status='APPROVED_TECHNICAL_RESOURCE_AMENDMENT', amendment_request_sha256=change_sha,
                scientific_policy_changed=False, evaluate_allowed=False),
            'Technical amendment approval differs')
    parent = change_request['parent']
    require(parent['request_sha256'] == request_sha and parent['admission_sha256'] == admission_sha,
            'Amendment belongs to another parent profile')
    change = change_request['change']
    require(set(change) == {'field', 'old', 'new'} and change['field'] == 'search.max_total_seeds'
            and type(change['old']) is int and type(change['new']) is int
            and 0 < expected['max_total_seeds'] <= change['old'] < change['new'],
            'Only increasing a technical seed budget is supported')
    expected['max_total_seeds'] = change['new']
    require(manifest['seed_policy'] == expected, 'Amendment changed a scientific search field')
    require(envelope.get('schema') == 'r02_k03_profile_amended_planner_envelope_v1'
            and envelope.get('status') == 'FIT_PROFILE_COMPLETE'
            and envelope.get('parent_profile_request_sha256') == request_sha
            and envelope.get('amendment_request_sha256') == change_sha
            and envelope.get('amendment_admission_sha256') == approval_sha
            and envelope.get('effective_max_total_seeds') == change['new'],
            'Incomplete or mismatched amendment envelope')
    result = envelope['planner_result']
    require(result.get('status') == 'FIT_PROFILE_COMPLETE'
            and result.get('manifest_sha256') == manifest_sha256
            and result.get('biological_validation') is False,
            'Envelope does not authenticate this completed profile')
    return dict(amended=True, request_sha256=request_sha, amendment_sha256=change_sha,
                envelope_sha256=refs['envelope']['sha256'], effective_max_total_seeds=change['new'])
