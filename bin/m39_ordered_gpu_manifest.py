"""Pure validation of frozen paired GPU-training plans; no genotype reads."""
from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

from m39_gpu_serial_profile import sha256

SCHEMA = 'm39-ordered-gpu-training-plan-v1'
SCOPE = 'exploratory_chr22_R0_development_anchors_only'
ARMS = ('common', 'pooled', 'real', 'sham')
STAGE_ARMS = {'exploratory_screen': ('common', 'real'), 'controlled_followup': ARMS}
DEVELOPMENT_FIELDS = frozenset(('alt', 'anchor_indices', 'baseline', 'chrom', 'coords',
    'full_baseline', 'locus_id', 'pos', 'ref', 'sample_key_sha256', 'select_indices',
    'source_indices', 'state_names', 'train_indices', 'truth_state'))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    require(set(plan) == {'schema_version', 'scope', 'stage', 'resources', 'inputs', 'groups'},
            'training plan fields differ')
    require(plan['schema_version'] == SCHEMA and plan['scope'] == SCOPE, 'training plan scope differs')
    require(plan['stage'] in STAGE_ARMS, 'explicit screen or controlled followup stage required')
    arms = STAGE_ARMS[plan['stage']]
    require(set(plan['inputs']) == {'train_manifest_sha256', 'select_manifest_sha256', 'development_sha256'}
            and all(digest(x) for x in plan['inputs'].values()), 'input seals differ')
    resources = plan['resources']
    require(set(resources) == {'max_workers', 'task_seconds', 'controller_seconds'}, 'resource fields differ')
    require(type(resources['max_workers']) is int and 1 <= resources['max_workers'] <= 2,
            'at most two concurrent GPU workers')
    require(type(resources['task_seconds']) is int and 60 <= resources['task_seconds'] <= 8 * 3600,
            'invalid per-worker timeout')
    require(type(resources['controller_seconds']) is int
            and resources['task_seconds'] < resources['controller_seconds'] <= 24 * 3600,
            'invalid total controller timeout')
    require(isinstance(plan['groups'], list) and 1 <= len(plan['groups']) <= 16,
            'expected one to sixteen explicit paired configurations')
    groups, files, case_ids = set(), set(), set()
    for group in plan['groups']:
        require(set(group) == {'id', 'configs'} and isinstance(group['id'], str)
                and re.fullmatch(r'[a-z0-9][a-z0-9-]{2,79}', group['id'])
                and group['id'] not in groups, 'invalid or duplicate group')
        groups.add(group['id'])
        require(isinstance(group['configs'], list) and len(group['configs']) == len(arms),
                'stage requires its exact paired-arm inventory')
        reference = None
        for arm, spec in zip(arms, group['configs']):
            require(set(spec) == {'file', 'sha256'} and isinstance(spec['file'], str)
                    and re.fullmatch(r'[a-z0-9][a-z0-9-]*\.json', spec['file'])
                    and spec['file'] not in files and digest(spec['sha256']), 'unsafe config member')
            config_path = path.parent / spec['file']
            require(config_path.is_file() and not config_path.is_symlink()
                    and sha256(config_path) == spec['sha256'], 'config hash differs')
            cfg = json.loads(config_path.read_text())
            require(cfg.get('schema_version') == 'm39-ordered-anchor-training-v1'
                    and cfg.get('scope') == SCOPE and cfg.get('device') == 'cuda:0'
                    and cfg.get('arm') == arm and cfg.get('paired_budget_id') == group['id'],
                    'config scope/device/paired arm differs')
            require(isinstance(cfg.get('case_id'), str) and re.fullmatch(r'[a-z0-9][a-z0-9-]{2,159}', cfg['case_id'])
                    and cfg['case_id'] not in case_ids, 'invalid or duplicate case')
            require(all(cfg.get(key) == value for key, value in plan['inputs'].items()),
                    'config input binding differs')
            require(type(cfg.get('max_runtime_seconds')) is int
                    and 0 < cfg['max_runtime_seconds'] < resources['task_seconds'], 'invalid arm timeout')
            comparable = {key: value for key, value in cfg.items() if key not in ('case_id', 'arm')}
            require(reference is None or comparable == reference, 'paired arms have different budgets or models')
            reference = comparable
            files.add(spec['file'])
            case_ids.add(cfg['case_id'])
    return plan


def store_inventory(path: Path, expected: str) -> int:
    """Authenticate the exact array inventory without materializing its values."""
    require(path.is_dir() and not path.is_symlink(), 'store must be a regular directory')
    manifest_path = path / 'manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink()
            and sha256(manifest_path) == expected, 'store manifest hash differs')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema_version') == 'm39-ordered-context-factorized-v1', 'store schema differs')
    names, total = {'manifest.json'}, manifest_path.stat().st_size
    for spec in manifest['arrays'].values():
        name = spec['file']
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_]+\.npy', name)
                and name not in names, 'unsafe store member')
        member = path / name
        require(member.is_file() and not member.is_symlink() and sha256(member) == spec['sha256'],
                'store member hash differs')
        names.add(name)
        total += member.stat().st_size
    require({item.name for item in path.iterdir()} == names and len(names) == 29,
            'store contains unexpected members')
    return total


def development_inventory(path: Path, expected: str) -> int:
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
            'development hash differs')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(DEVELOPMENT_FIELDS)
                and set(names) == {name + '.npy' for name in DEVELOPMENT_FIELDS},
                'development inventory differs; SCORE and other roles must not be staged')
    return path.stat().st_size
