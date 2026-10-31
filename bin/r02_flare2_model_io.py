"""Truth-blind validation and canonicalization of FLARE 0.6.0 model outputs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


class Flare2ModelError(ValueError):
    """Raised when a clustered FLARE model cannot be identified safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Flare2ModelError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _section(lines: list[str], heading: str, skip: int = 1) -> int:
    try:
        return lines.index(heading) + skip
    except ValueError as exc:
        raise Flare2ModelError(f"FLARE model lacks section: {heading}") from exc


def parse_model(path: Path, n_ancestries: int) -> dict[str, Any]:
    """Parse the fields that bind latent clusters to declared reference panels.

    Cluster names are arbitrary.  This function validates their probability
    simplex without consulting simulated or observed target ancestry labels.
    """
    require(path.is_file(), "FLARE model must resolve to a regular file")
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), "FLARE model must resolve to a regular file")
    lines = resolved.read_text(encoding="utf-8").splitlines()
    anc_at = _section(lines, "# list of ancestries")
    ancestries = lines[anc_at].split()
    panel_at = _section(lines, "# list of reference panels")
    panels = lines[panel_at].split()
    require(len(ancestries) == n_ancestries and len(set(ancestries)) == n_ancestries,
            "FLARE model ancestry/cluster count differs")
    require(panels and len(set(panels)) == len(panels),
            "FLARE model reference panels are empty or duplicated")
    p_heading = "# p[i][j]: probability that a model state haplotype is in reference panel j"
    p_at = _section(lines, p_heading, 2)
    matrix: list[list[float]] = []
    row_sums: list[float] = []
    for row in lines[p_at:p_at + n_ancestries]:
        values = [float(value) for value in row.split()]
        require(len(values) == len(panels), "FLARE model panel-probability width differs")
        require(all(math.isfinite(value) and value >= -1e-12 and value <= 1 + 1e-12
                    for value in values), "FLARE model has invalid panel probability")
        total = math.fsum(values)
        require(abs(total - 1.0) <= 1e-8,
                "FLARE model cluster center is not a normalized probability simplex")
        matrix.append(values)
        row_sums.append(total)
    require(len(matrix) == n_ancestries, "FLARE model panel-probability row count differs")
    return {
        "model_sha256": sha256_file(resolved),
        "clusters": ancestries,
        "panels": panels,
        "p": matrix,
        "p_row_sums": row_sums,
        "normalization_tolerance": 1e-8,
    }
