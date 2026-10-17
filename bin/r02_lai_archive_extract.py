#!/usr/bin/env python3
"""Verify a retained tar and its inventory, then extract only requested members."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True); p.add_argument("--archive-sha256", required=True)
    p.add_argument("--inventory", required=True); p.add_argument("--inventory-sha256", required=True)
    p.add_argument("--member", action="append", required=True); p.add_argument("--output-dir", required=True)
    a = p.parse_args(); archive, inventory = Path(a.archive), Path(a.inventory)
    require(sha(archive) == a.archive_sha256 and sha(inventory) == a.inventory_sha256,
            "archive or inventory SHA-256 differs")
    ledger = json.loads(inventory.read_text()); entries = ledger.get("entries")
    require(isinstance(entries, list), "inventory entries absent")
    indexed = {row.get("archive_member", row.get("path")): row for row in entries
               if row.get("disposition") in (None, "PRESERVE")}
    require(set(a.member) <= set(indexed), "requested member absent from preservation inventory")
    out = Path(a.output_dir); require(not out.exists() and out.parent.is_dir(), "fresh output required")
    out.mkdir(mode=0o700)
    with tarfile.open(archive, "r:*") as bundle:
        members = {row.name: row for row in bundle.getmembers() if row.isfile()}
        require(set(a.member) <= set(members), "requested member absent from tar")
        for number, name in enumerate(a.member):
            source = bundle.extractfile(members[name]); require(source is not None, "member stream absent")
            target = out / (f"{number:02d}-" + Path(name).name)
            with target.open("xb") as stream:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""): stream.write(block)
            row = indexed[name]
            require(target.stat().st_size == int(row["bytes"]) and sha(target) == row["sha256"],
                    "extracted member differs from preservation inventory")
    (out / "receipt.json").write_text(json.dumps({"schema":"r02_lai_archive_extract_receipt_v1",
        "status":"REQUESTED_MEMBERS_VERIFIED_AND_EXTRACTED", "archive_sha256":a.archive_sha256,
        "inventory_sha256":a.inventory_sha256,
        "members":[{"name":name,"output":f"{i:02d}-{Path(name).name}","sha256":indexed[name]["sha256"]}
                   for i,name in enumerate(a.member)]}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
