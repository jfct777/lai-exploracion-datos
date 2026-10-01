#!/usr/bin/env python3
"""Execute one frozen analysis command inside a resource-bounded Nextflow task."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ALLOWED = {
    'r02_genomic_pair_evidence.py', 'r02_common_grm.sh',
    'rare_allele_sharing_painter.py', 'rare_segment_sensitivity.py',
    'm165_autosome_sweep.py', 'm165_chr22_sweep.py', 'm165_spectral_figures.py',
    'm165_sweep_figures.py', 'r02_weighted_communities.py',
    'm165_graph_kinship.py', 'r02_weighted_kinship.py',
    'r02_m14_configuration_diagnostics.py',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def run(command_path, source_bin, receipt):
    payload = json.loads(Path(command_path).read_text())
    command = payload['command']
    if (not isinstance(command,list) or len(command)<2 or command[0] not in ('python3','bash')
            or not all(isinstance(s,str) and '\x00' not in s for s in command)):
        raise ValueError('Invalid analysis command')
    tool = Path(command[1])
    if not tool.is_absolute() or tool.name not in ALLOWED:
        raise ValueError('Only explicitly allowed frozen tools may run')
    staged = Path(source_bin)/tool.name
    if sha(tool)!=sha(staged) or sha(tool)!=payload['tool_sha256']:
        raise ValueError('Scientific tool differs from frozen command')
    if Path(receipt).exists(): raise ValueError('Task receipt exists; no overwrite')
    before = datetime.now(timezone.utc).isoformat()
    # Argument array, no shell expansion or interpolation of sample names.
    subprocess.run(command,check=True)
    if sha(tool)!=payload['tool_sha256']: raise ValueError('Tool changed during execution')
    result = dict(schema_version=1,started_utc=before,completed_utc=datetime.now(timezone.utc).isoformat(),
                  command=command,tool_sha256=sha(tool),command_sha256=sha(command_path),
                  status='COMMAND_COMPLETED_OUTPUT_VALIDATION_BY_MODULE_AND_SUPERVISOR')
    with Path(receipt).open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--command',required=True)
    p.add_argument('--source-bin',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args()
    run(a.command,a.source_bin,a.receipt)


if __name__=='__main__': main()
