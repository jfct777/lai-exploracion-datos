#!/usr/bin/env python3
"""Run one command while enforcing a fail-closed Linux process-tree RSS limit."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _process_table() -> dict[int, tuple[int, int]]:
    """Return PID -> (PPID, RSS KiB) for readable Linux processes."""
    table: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            status = (entry / "status").read_text(encoding="utf-8").splitlines()
            ppid = int(fields[3])
            rss_line = next(line for line in status if line.startswith("VmRSS:"))
            rss_kib = int(rss_line.split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, ValueError, IndexError):
            continue
        table[int(entry.name)] = (ppid, rss_kib)
    return table


def process_tree_rss_kib(root_pid: int, table: dict[int, tuple[int, int]]) -> int:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in table.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(table.get(pid, (0, 0))[1] for pid in descendants)


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(command: list[str], report_path: Path, max_rss_gib: float, poll_seconds: float) -> dict:
    if max_rss_gib <= 0 or poll_seconds <= 0:
        raise ValueError("RSS threshold and poll interval must be positive")
    threshold_kib = int(max_rss_gib * 1024 * 1024)
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    peak_kib = 0
    exceeded = False
    while process.poll() is None:
        peak_kib = max(peak_kib, process_tree_rss_kib(process.pid, _process_table()))
        if peak_kib > threshold_kib:
            exceeded = True
            _terminate_group(process)
            break
        time.sleep(poll_seconds)
    peak_kib = max(peak_kib, process_tree_rss_kib(process.pid, _process_table()))
    returncode = process.returncode if process.returncode is not None else process.wait()
    report = {
        "stage": "M29_PROCESS_TREE_RSS_GATE",
        "command": command,
        "duration_seconds": time.monotonic() - started,
        "peak_rss_kib": peak_kib,
        "peak_rss_gib": peak_kib / (1024 * 1024),
        "max_rss_gib": max_rss_gib,
        "poll_seconds": poll_seconds,
        "command_returncode": returncode,
        "threshold_exceeded": exceeded,
        "decision": "STOP_RSS_LIMIT_EXCEEDED" if exceeded else ("PASS_RSS_GATE" if returncode == 0 else "STOP_COMMAND_FAILED"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-rss-gib", required=True, type=float)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.command, arguments.report, arguments.max_rss_gib, arguments.poll_seconds)
    if result["decision"] != "PASS_RSS_GATE":
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        raise SystemExit(88 if result["threshold_exceeded"] else result["command_returncode"] or 1)
    print(json.dumps({"decision": result["decision"], "peak_rss_gib": result["peak_rss_gib"]}, sort_keys=True))
