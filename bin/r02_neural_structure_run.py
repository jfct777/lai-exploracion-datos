#!/usr/bin/env python3
"""Run one admitted HUMAN instrumentation arm and measure checkpoint recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path, digest=None):
    path = Path(path)
    require(path.is_file() and (digest is None or sha(path) == digest),
            f"missing or changed input: {path}")
    return json.loads(path.read_text())


def command(args, output):
    return [sys.executable, str(Path(__file__).with_name("r02_context_train.py")),
            "--stage", "train", "--contract", args.contract,
            "--contract-sha256", args.contract_sha256,
            "--resolved", args.resolved, "--resolved-sha256", args.resolved_sha256,
            "--arm", args.arm, "--seed", str(args.seed), "--output", str(output),
            "--checkpoint-dir", args.checkpoint_dir, "--checkpoint-every-steps", "1",
            "--adam-state-precision", args.adam_state_precision]


def wait_step_one(process, checkpoint, poll_seconds, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    current = checkpoint / "CURRENT"
    while time.monotonic() < deadline:
        if current.is_file():
            pointer = load(current)
            if pointer.get("next_step") == 1:
                require(process.poll() is None,
                        "trainer finished before a step-1 interruption could be observed")
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                require(process.returncode != 0, "first process was not interrupted")
                return pointer
            require(pointer.get("next_step", 0) < 2,
                    "checkpoint advanced beyond step 1 before forced recovery")
        require(process.poll() is None, "trainer failed before the first checkpoint")
        time.sleep(poll_seconds)
    process.terminate(); process.wait(timeout=30)
    raise TimeoutError("step-1 checkpoint was not observed within the operator limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("contract", "contract-sha256", "resolved", "resolved-sha256",
                "arm", "output", "checkpoint-dir", "measurement"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force-recovery-step1", action="store_true")
    parser.add_argument("--recover-owned-run", action="store_true")
    parser.add_argument("--synthetic-code-test", action="store_true",
                        help="accept only SYNTHETIC_CODE_TEST for the container integration fixture")
    parser.add_argument("--checkpoint-timeout-seconds", type=int, default=86400)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--adam-state-precision", choices=("fp32", "fp64"), default="fp32")
    args = parser.parse_args()
    require(args.arm in ("NONE", "COUNTS", "RARE") and args.seed >= 0,
            "invalid arm/seed")
    contract = load(args.contract, args.contract_sha256)
    expected_scope = "SYNTHETIC_CODE_TEST" if args.synthetic_code_test else "DEVELOPMENT"
    require(contract.get("steps") == 2 and contract.get("scope") == expected_scope,
            "runner scope differs from the explicit two-update mode")
    load(args.resolved, args.resolved_sha256)
    output, checkpoint = Path(args.output), Path(args.checkpoint_dir)
    measurement = Path(args.measurement)
    require(not measurement.exists(), "new measurement path required")
    output.parent.mkdir(parents=True, exist_ok=True)
    measurement.parent.mkdir(parents=True, exist_ok=True)
    current_path = checkpoint / "CURRENT"
    current = load(current_path) if current_path.is_file() else None
    if current is not None:
        require(args.recover_owned_run and current.get("next_step") in (1, 2),
                "pre-existing checkpoint requires --recover-owned-run")
        if output.exists():
            previous = output.parent / ("." + output.name +
                                        f".previous-step{current['next_step']}")
            require(not previous.exists(), "owned recovery backup already exists")
            output.rename(previous)
    else:
        require(not output.exists(), "fresh run output already exists")
    started = time.monotonic(); interrupted = None
    forced_marker = checkpoint / "FORCED_STEP1.json"
    if args.force_recovery_step1 and current is None:
        first = output.parent / ("." + output.name + ".interrupted-step1")
        require(not first.exists() and not checkpoint.exists(),
                "forced recovery requires fresh wrapper-owned paths")
        process = subprocess.Popen(command(args, first))
        interrupted = wait_step_one(process, checkpoint, args.poll_seconds,
                                    args.checkpoint_timeout_seconds)
        forced_marker.write_text(json.dumps(interrupted, sort_keys=True) + "\n")
        current = interrupted
    elif args.force_recovery_step1:
        require(forced_marker.is_file(),
                "owned recovery lacks evidence of the intentional step-1 interruption")
        interrupted = load(forced_marker)
    completed = subprocess.run(command(args, output), check=False)
    require(completed.returncode == 0, "final training process failed; preserve disk/checkpoint")
    receipt_path = output / "training.receipt.json"
    receipt = load(receipt_path)
    expected_restore = current["next_step"] if current is not None else 0
    internal = receipt.get("internal_checkpoint", {})
    require(receipt.get("status") == "COMPLETE" and receipt.get("arm") == args.arm
            and receipt.get("steps") == 2 and internal.get("restored_steps") == expected_restore
            and internal.get("adam_state_precision") == args.adam_state_precision
            and internal.get("final_confirmed_step") == 2,
            "training receipt does not prove the required final/checkpoint state")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"schema": "r02_neural_structure_measurement_v1", "status": "COMPLETE",
              "arm": args.arm, "seed": args.seed, "optimizer_updates": 2,
              "adam_state_precision": args.adam_state_precision,
              "forced_interruption_after_step": 1 if interrupted else None,
              "restored_steps": internal["restored_steps"],
              "final_confirmed_step": internal["final_confirmed_step"],
              "elapsed_seconds": time.monotonic() - started,
              "children_max_rss_kib": usage.ru_maxrss,
              "receipt": {"path": str(receipt_path.resolve()), "sha256": sha(receipt_path)},
              "checkpoint_pointer": interrupted,
              "scope": expected_scope,
              "scientific_conclusion": "NONE_ENGINEERING_INSTRUMENTATION_ONLY"}
    measurement.write_text(json.dumps(result, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
