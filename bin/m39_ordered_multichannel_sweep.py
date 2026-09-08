#!/usr/bin/env python3
"""Freeze and audit the multichannel pilot using the shared ordered primary auditor.

BOTH is the focal arm, NONE the common-context comparator. These names are never
rewritten as historical REAL/COMMON. No model execution or SCORE access occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m39_ordered_gpu_manifest import load_plan, require
from m39_ordered_multichannel_training import load_config
import m39_ordered_training_sweep as shared


PROTOCOL = shared.SweepProtocol(
    config_loader=load_config, focal_arm='both', common_arm='none',
    screen_stage='multichannel_screen', followup_stage='multichannel_followup',
    technical_stage='multichannel_technical',
    comparison_schema='m39-ordered-multichannel-comparison-v1',
    initial_checkpoint_eligible=False, preserve_followup_geometry=True)


def freeze_plan(base_path: Path, resources: dict, recipes: list[dict], stage: str,
                outdir: Path) -> Path:
    require(stage in (PROTOCOL.screen_stage, PROTOCOL.followup_stage, PROTOCOL.technical_stage),
            'multichannel stage required')
    return shared.freeze_plan(base_path, resources, recipes, stage, outdir, protocol=PROTOCOL)


def audit_results(plan_path: Path, outputs: Path, outdir: Path | None = None) -> dict:
    plan = load_plan(plan_path)
    require(plan['stage'] in (PROTOCOL.screen_stage, PROTOCOL.followup_stage, PROTOCOL.technical_stage),
            'multichannel stage required')
    return shared.audit_results(plan_path, outputs, outdir, protocol=PROTOCOL)


def selected_learning_rates(summary: dict) -> dict:
    return shared.selected_learning_rates(summary, protocol=PROTOCOL)


def freeze_followup(base_path: Path, resources: dict, screen_plan: Path,
                    screen_outputs: Path, seeds: list[int], outdir: Path) -> Path:
    return shared.freeze_followup(base_path, resources, screen_plan, screen_outputs,
                                  seeds, outdir, protocol=PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare = modes.add_parser('prepare')
    for name in ('base-config', 'recipe-file', 'outdir'):
        prepare.add_argument('--' + name, type=Path, required=True)
    audit = modes.add_parser('audit')
    for name in ('plan', 'outputs', 'outdir'):
        audit.add_argument('--' + name, type=Path, required=True)
    followup = modes.add_parser('prepare-followup')
    for name in ('base-config', 'resources-file', 'screen-plan', 'screen-outputs', 'outdir'):
        followup.add_argument('--' + name, type=Path, required=True)
    followup.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        spec = json.loads(args.recipe_file.read_text())
        require(set(spec) == {'stage', 'resources', 'recipes'}, 'recipe file fields differ')
        freeze_plan(args.base_config, spec['resources'], spec['recipes'], spec['stage'], args.outdir)
    elif args.mode == 'audit':
        audit_results(args.plan, args.outputs, args.outdir)
    else:
        freeze_followup(args.base_config, json.loads(args.resources_file.read_text()),
                        args.screen_plan, args.screen_outputs, args.seeds, args.outdir)


if __name__ == '__main__':
    main()
