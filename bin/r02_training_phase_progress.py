"""Durable stage progress, separate from an optimizer checkpoint or success receipt."""
import json
import os
import time
from pathlib import Path


class TrainingPhaseProgress:
    """One invocation's atomic progress and append-only phase history.

    No genotype values or person identifiers are recorded. A completed input
    check never authorizes skipping that check in a later invocation.
    """

    def __init__(self, directory, contract_sha256, resolved_sha256, condition):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.identity = dict(schema="r02_training_phase_progress_v1", pid=os.getpid(),
                             invocation_started_epoch=time.time(),
                             contract_sha256=contract_sha256,
                             resolved_sha256=resolved_sha256, condition=condition)

    def record(self, stage, **details):
        if set(details) & (set(self.identity) | {"stage", "elapsed_seconds"}):
            raise ValueError("progress details cannot replace invocation identity")
        value = dict(self.identity, stage=stage,
                     elapsed_seconds=time.monotonic()-self.started, **details)
        text = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
        with (self.root / "PHASE_HISTORY.jsonl").open("a") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        path = self.root / "PHASE_PROGRESS.json"
        temporary = path.with_suffix(f".tmp-{os.getpid()}")
        with temporary.open("w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def authenticate(self, batches, validate, *, fit_only):
        count = 0
        total = sum(len(entries) for role, entries in batches.items()
                    if not fit_only or role == "FIT")
        self.record("INPUT_AUTHENTICATION_STARTED", completed=0, total=total)
        for role, entries in batches.items():
            if fit_only and role != "FIT":
                continue
            for index, entry in enumerate(entries):
                before = time.monotonic()
                self.record("INPUT_ENTRY_STARTED", role=role, entry_index=index,
                            completed=count, total=total)
                try:
                    validate(entry, role)
                except Exception as error:
                    self.record("INPUT_ENTRY_FAILED", role=role, entry_index=index,
                                completed=count, total=total,
                                duration_seconds=time.monotonic()-before,
                                exception_type=type(error).__name__)
                    raise
                count += 1
                self.record("INPUT_ENTRY_AUTHENTICATED", role=role, entry_index=index,
                            completed=count, total=total,
                            duration_seconds=time.monotonic()-before)
        self.record("INPUT_AUTHENTICATION_COMPLETE", completed=count, total=total)
