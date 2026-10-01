#!/usr/bin/env python3
"""Freeze a bounded R02 amendment without stopping or launching cohort work."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import types

ROOT = Path(__file__).resolve().parents[1]
CHANGED_SOURCE_ALLOWLIST = {
    'bin/r02_autosome_pipeline.py', 'bin/r02_exec_task.py',
    'bin/rare_allele_sharing_painter.py',
}
ADDED_SOURCE_ALLOWLIST = {
    'bin/r02_apply_amendment.py', 'bin/r02_m14_configuration_diagnostics.py',
    'bin/r02_prepare_boundary_amendment.py', 'bin/r02_stage_boundary_handoff.py',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    root = Path('/proc') / str(pid)
    stat = (root/'stat').read_text().rsplit(') ', 1)[1].split()
    command = (root/'cmdline').read_bytes()
    return dict(pid=pid, start_ticks=int(stat[19]),
                cmdline_sha256=hashlib.sha256(command).hexdigest()), int(stat[1]), command


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def load_module(path, name):
    """Do not create bytecode caches in the original frozen source."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def source_inventory(root):
    inventory = {}
    for directory, suffixes in [('bin', ('.py', '.sh')), ('modules', ('.nf',)), ('workflows', ('.nf',))]:
        for source in (root/directory).rglob('*'):
            if source.is_file() and not source.is_symlink() and source.suffix in suffixes:
                inventory[str(source.relative_to(root))] = sha(source)
    return inventory


def validate_source_delta(root, run, old_hashes, adapter):
    """Fail before creating an amendment, not after interrupting the supervisor."""
    hashes = source_inventory(root)
    missing = set(old_hashes) - set(hashes)
    changed = {name for name in set(old_hashes) & set(hashes) if old_hashes[name] != hashes[name]}
    added = set(hashes) - set(old_hashes)
    if (missing or changed - CHANGED_SOURCE_ALLOWLIST or added - ADDED_SOURCE_ALLOWLIST):
        raise ValueError('Source delta exceeds the bounded amendment allowlist')
    for relative in adapter.PROTECTED:
        if relative not in hashes or old_hashes.get(relative) != hashes[relative]:
            raise ValueError('This handoff must preserve preprocessing/workflows: ' + relative)
    for relative in set(old_hashes) & set(hashes):
        if relative.startswith(('modules/', 'workflows/')) and old_hashes[relative] != hashes[relative]:
            raise ValueError('Existing workflow/module changed: ' + relative)
    if adapter.method_ast(root/'bin/r02_autosome_pipeline.py', 'Runner', 'preprocess') != adapter.method_ast(
            run/'source/bin/r02_autosome_pipeline.py', 'Runner', 'preprocess'):
        raise ValueError('Runner preprocessing implementation changed')
    needed = {*adapter.ANALYSIS_TOOLS, 'r02_apply_amendment.py', 'r02_stage_boundary_handoff.py',
              'r02_prepare_boundary_amendment.py', 'r02_autosome_pipeline.py'}
    if any('bin/' + name not in hashes for name in needed):
        raise ValueError('Missing amendment programs or dependencies')
    tree = ast.parse((root/'bin/r02_exec_task.py').read_text())
    whitelist = next((ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'ALLOWED' for target in node.targets)), None)
    outer_tools = set(adapter.ANALYSIS_TOOLS) - {'r02_exec_task.py', 'ibd_community_enhanced.py'}
    if not isinstance(whitelist, set) or not outer_tools <= whitelist:
        raise ValueError('Analysis executor does not allow every amended tool, including diagnostics')
    return hashes


def prepare(run, amendment, supervisor, child):
    run, amendment = run.resolve(), amendment.resolve()
    if not (ROOT/'.git').is_dir():
        raise ValueError('Prepare only from the working project, never from a frozen copy')
    if amendment.parent != run/'repairs' or amendment.exists():
        raise ValueError('Use a new direct subdirectory of this run repairs directory')
    adapter = load_module(ROOT/'bin/r02_apply_amendment.py', 'r02_amendment_validator')
    adapter.verify_manifest(run, adapter.load_json(run/'frozen.sha256.json'))
    old_hashes = adapter.load_json(run/'source.sha256.json')
    adapter.verify_manifest(run/'source', old_hashes)
    module = load_module(run/'source/bin/r02_autosome_pipeline.py', 'r02_original_controller')
    module.Runner(run)  # authenticate original frozen manifests, no job or GCS writes
    state = json.loads((run/'status.json').read_text())
    stage = state['stage']
    if (stage != 'chr21_M01_M02_M021' or state.get('state') != 'RUNNING'
            or state.get('pid') != child):
        raise ValueError('This amendment is only for the currently authenticated chr21 boundary')
    parent_id, _, parent_cmd = identity(supervisor)
    child_id, ppid, child_cmd = identity(child)
    if (ppid != supervisor or str(run).encode() not in parent_cmd
            or b'r02_autosome_pipeline.py' not in parent_cmd or b'nextflow' not in child_cmd
            or str(run/'chr21').encode() not in child_cmd):
        raise ValueError('Unexpected supervisor/child command or parentage')
    if (run/'checkpoints'/f'{stage}.json').exists():
        raise ValueError('Boundary already completed; re-check actual state')
    run_config = adapter.load_json(run/'run.json')
    if adapter.load_json(run/'chr21/parameters.json') != adapter.expected_preprocess_parameters(run, run_config, 21):
        raise ValueError('Current preprocessing parameters differ from the original protocol')
    expected_hashes = validate_source_delta(ROOT, run, old_hashes, adapter)
    os.umask(0o077)
    amendment.mkdir()
    for directory, suffixes in [('bin',('.py','.sh')),('modules',('.nf',)),('workflows',('.nf',))]:
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and not source.is_symlink() and source.suffix in suffixes:
                target=amendment/'source'/source.relative_to(ROOT)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    source_hashes={str(p.relative_to(amendment/'source')):sha(p)
                   for p in sorted((amendment/'source').rglob('*')) if p.is_file()}
    if source_hashes != expected_hashes:
        raise ValueError('Source changed while freezing; do not arm this amendment')
    write_new(amendment/'source.sha256.json',source_hashes)
    template=dict(schema_version=1,run_dir=str(run),
        original_frozen_sha256=sha(run/'frozen.sha256.json'),
        original_source_manifest_sha256=sha(run/'source.sha256.json'),
        boundary=dict(chromosome=21,stage=stage,
                      parameters_sha256=sha(run/'chr21/parameters.json'),
                      runtime_config_sha256=sha(run/'chr21/runtime.config')),
        overrides=dict(analysis_protocol_version=2,m14_skip_legacy_windows=True,
            evaluate_all_m14_configurations=True,
            m14_diagnostic_edge_thresholds_bp=[0,250000,500000,750000,1000000],
            m14_diagnostic_kinship_thresholds=[.0221,.0442]))
    request=dict(schema_version=1,run_dir=str(run),amendment_dir=str(amendment),
        supervisor=parent_id,child=child_id,boundary_stage=stage,
        free_disk_min_gib=run_config['free_disk_min_gib'],
        original_limits=list(resource.prlimit(supervisor,resource.RLIMIT_NPROC)),
        amendment_source_manifest_sha256=sha(amendment/'source.sha256.json'),
        amendment_template=template)
    # Do not prepare against a different live process or stage after freezing.
    if identity(supervisor)[0]!=parent_id or identity(child)[0]!=child_id:
        raise ValueError('Process identity changed during preparation; do not arm')
    latest=json.loads((run/'status.json').read_text())
    if latest.get('stage')!=stage or latest.get('pid')!=child or latest.get('state')!='RUNNING':
        raise ValueError('Stage changed during preparation; do not arm')
    write_new(amendment/'request.json',request)
    print(json.dumps(dict(status='PREPARED_NOT_ARMED',request=str(amendment/'request.json'),
                          request_sha256=sha(amendment/'request.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--amendment-dir',type=Path,required=True)
    p.add_argument('--supervisor-pid',type=int,required=True)
    p.add_argument('--child-pid',type=int,required=True)
    a=p.parse_args()
    prepare(a.run_dir,a.amendment_dir,a.supervisor_pid,a.child_pid)
