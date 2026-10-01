#!/usr/bin/env python3
"""Reject remote scratch and insufficient free space before large preprocessing.

The check applies to temporary output directories, never to read-only genomic
inputs. It resolves the actual mount, including a block-device bind mount below
a FUSE directory. It neither creates directories nor changes mounts or data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil


GIB = 1024 ** 3
# Deliberately fail closed: neither a remote filesystem, tmpfs nor an unknown
# filesystem is a certified disk for large intermediates. Overlay is accepted
# only for small fixtures explicitly using container-local scratch.
DISK_FILESYSTEMS = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs"})


def unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def find_mount(directory: Path, mountinfo: str) -> dict:
    candidates = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if "-" not in fields or len(fields) < 10:
            raise ValueError("Malformed mount information")
        separator = fields.index("-")
        if separator < 6 or len(fields) < separator + 4:
            raise ValueError("Incomplete mount information")
        mountpoint = Path(unescape_mount(fields[4]))
        if directory == mountpoint or mountpoint in directory.parents:
            candidates.append((len(mountpoint.parts), int(fields[0]), {
                "mountpoint": str(mountpoint),
                "device": fields[2],
                "filesystem": fields[separator + 1],
                "mount_options": fields[5].split(","),
                "super_options": fields[separator + 3].split(","),
            }))
    if not candidates:
        raise ValueError("Cannot determine scratch filesystem")
    return max(candidates, key=lambda item: item[:2])[2]


def inspect_storage(directory: Path, required_free_bytes: int = 0,
                    minimum_free_gib: float = 12, *, allow_overlay: bool = False,
                    mountinfo: str | None = None) -> dict:
    directory = Path(directory)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("Scratch must be an existing absolute directory")
    if type(required_free_bytes) is not int or required_free_bytes < 0:
        raise ValueError("Required free bytes must be a nonnegative integer")
    if isinstance(minimum_free_gib, bool) or not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("Minimum free GiB must be finite and nonnegative")
    directory = directory.resolve(strict=True)
    mount = find_mount(directory, mountinfo if mountinfo is not None
                       else Path("/proc/self/mountinfo").read_text())
    device_number = directory.stat().st_dev
    visible_device = f"{os.major(device_number)}:{os.minor(device_number)}"
    if mount["device"] != visible_device:
        raise ValueError("Scratch mount is hidden or changed during inspection")
    allowed = DISK_FILESYSTEMS | ({"overlay"} if allow_overlay else set())
    if mount["filesystem"] not in allowed:
        raise ValueError("Scratch is not a verified local disk filesystem: " + mount["filesystem"])
    if "ro" in mount["mount_options"] or "ro" in mount["super_options"]:
        raise ValueError("Scratch filesystem is read-only")
    if not os.access(directory, os.W_OK | os.X_OK):
        raise ValueError("Scratch is not writable by this process")
    free = shutil.disk_usage(directory).free
    required = max(required_free_bytes, math.ceil(minimum_free_gib * GIB))
    if free < required:
        raise ValueError(f"Insufficient scratch space: {free} free bytes, {required} required")
    return {"schema_version": 1, "directory": str(directory), **mount,
            "free_bytes": free, "required_free_bytes": required,
            "status": "LOCAL_SCRATCH_VERIFIED",
            "scope": "Current mount and free capacity; not a throughput benchmark or future-space guarantee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--required-free-bytes", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=12)
    parser.add_argument("--allow-overlay", action="store_true",
                        help="Only for bounded container fixtures; not cloud genomic scratch")
    args = parser.parse_args()
    print(json.dumps(inspect_storage(args.directory, args.required_free_bytes,
                                    args.minimum_free_gib, allow_overlay=args.allow_overlay), sort_keys=True))


if __name__ == "__main__":
    main()
