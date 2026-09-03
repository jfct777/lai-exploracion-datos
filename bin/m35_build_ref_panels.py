#!/usr/bin/env python3
"""Materialize explicit M35 REF_TRAIN FLARE panel and panel→macro files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


MACROS = {"African": "AFR", "European": "EUR", "Native_American": "NAM"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(roles_path: Path, output_dir: Path, granularity: str) -> dict[str, object]:
    require(granularity in {"coarse", "population"}, "granularity must be coarse or population")
    require(not output_dir.exists(), "refusing to overwrite panel output directory")
    with roles_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"sample_id", "ancestry", "population", "canonical_population", "role"}
    require(rows and required.issubset(rows[0]), "M27F role file lacks panel columns")
    selected = [row for row in rows if row["role"] == "REF_TRAIN"]
    require(selected, "M27F role file has no REF_TRAIN rows")
    seen = set()
    sample_panel: list[tuple[str, str]] = []
    panel_macro: dict[str, str] = {}
    for row in selected:
        sample, source_ancestry = row["sample_id"], row["ancestry"]
        require(sample and sample not in seen, "REF_TRAIN sample ID is empty or duplicated")
        seen.add(sample)
        require(source_ancestry in MACROS, "REF_TRAIN ancestry is unsupported")
        macro = MACROS[source_ancestry]
        panel = macro if granularity == "coarse" else row["population"]
        require(panel and panel.strip() == panel and not any(char.isspace() for char in panel),
                "fine panel label is empty or unsafe")
        previous = panel_macro.setdefault(panel, macro)
        require(previous == macro, "one fine panel maps to more than one macro-ancestry")
        sample_panel.append((sample, panel))
    require(set(panel_macro.values()) == {"AFR", "EUR", "NAM"}, "panel set lacks a macro-ancestry")
    output_dir.mkdir(parents=True)
    sample_path = output_dir / "m35_ref_train.sample_panel.tsv"
    macro_path = output_dir / "m35_panel_to_macro.tsv"
    sample_path.write_text("".join(f"{sample}\t{panel}\n" for sample, panel in sample_panel), encoding="utf-8")
    macro_path.write_text("".join(f"{panel}\t{macro}\n" for panel, macro in sorted(panel_macro.items())), encoding="utf-8")
    panel_counts = Counter(panel for _sample, panel in sample_panel)
    macro_counts = {macro: sum(panel_counts[panel] for panel, assigned in panel_macro.items() if assigned == macro)
                    for macro in ("AFR", "EUR", "NAM")}
    return {
        "schema_version": "1.0.0", "stage": "M35_REF_TRAIN_PANEL_MATERIALIZATION",
        "status": "PASS_METADATA_ONLY_PANEL_MAP", "granularity": granularity,
        "roles_sha256": sha256_file(roles_path), "sample_panel_sha256": sha256_file(sample_path),
        "panel_to_macro_sha256": sha256_file(macro_path), "panel_counts": dict(sorted(panel_counts.items())),
        "macro_counts": macro_counts, "panel_count": len(panel_counts), "sample_count": len(sample_panel),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--granularity", choices=("coarse", "population"), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.roles, args.outdir, args.granularity)
    (args.outdir / "m35_ref_train.panel_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
