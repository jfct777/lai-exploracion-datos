#!/usr/bin/env python3
"""Publish the fixed R02 fleet provenance without overwriting earlier records."""
import argparse
from pathlib import Path
from r02_parallel_fleet import upload_one, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.name != 'r02-autosomes-20261001b':
        parser.error('This publication is limited to the authorized R02 run')
    root = run/'repairs/parallel-v1'
    if args.verification.resolve().parent != root:
        parser.error('Verification must come from this fleet')
    destination = ('gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/'
                   'biologico/R02_20260930/r02-autosomes-20261001b/'
                   '00_datos_y_diseno/parallel-launch-20261001/')
    files = [(root/name, name) for name in ('fleet.json', 'upload.json', 'manifest.json', 'coordinator.py')]
    files += [(root/f'worker{i:02d}.creation.json', f'worker{i:02d}.creation.json') for i in range(1,13)]
    files += [(root/'cleanup'/name, 'cleanup/'+name) for name in ('manifest.json','cleanup.py')]
    files += [(run/name, 'parent/'+name) for name in ('run.json','source.sha256.json','frozen.sha256.json',
               'h_settings.json','weighted_settings_primary.json','weighted_settings_sensitivity.json')]
    files += [(root/'bootstrap/startup.sh','startup.sh'),
              (args.verification,args.verification.name), (args.report,args.report.name)]
    for path, _ in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing or unsafe provenance file: '+str(path))
    receipts = [upload_one(path,destination+relative) for path,relative in files]
    target = root/'launch_publication.json'
    write_new(target,dict(state='PUBLISHED_VERIFIED_CREATE_ONLY',destination=destination,files=receipts,
                          scientific_results_complete=False,biological_validation_complete=False))
    upload_one(target,destination+target.name)
    print(str(target))


if __name__ == '__main__':
    main()
