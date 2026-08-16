#!/usr/bin/env python3
"""Create a privacy-safe M27F transfer figure from frozen public/private receipts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-public", type=Path, required=True)
    parser.add_argument("--valid-public", type=Path, required=True)
    parser.add_argument("--valid-block-private", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_primary_nam_matrix(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[list[int]]]:
    selected = [
        row
        for row in rows
        if row["primary_for_local_transfer"] == "True"
        and row["ancestry"] == "Native_American"
    ]
    positions = sorted({int(row["pos"]) for row in selected})
    blocks = sorted({row["block_token"] for row in selected})
    if len(positions) != 3 or len(blocks) != 3 or len(selected) != 9:
        raise ValueError("Expected the frozen 3 x 3 primary NAM matrix")
    state_value = {
        "ABSENT": 0,
        "PRESENT": 1,
        "UNEVALUABLE_CALLABILITY": -1,
        "UNEVALUABLE_PHASE": -1,
    }
    by_cell = {(int(row["pos"]), row["block_token"]): state_value[row["state"]] for row in selected}
    matrix = [[by_cell[(position, block)] for block in blocks] for position in positions]
    return [f"Patrón {index}" for index in range(1, 4)], [f"Bloque {index}" for index in range(1, 4)], matrix


def render(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    ref = json.loads(args.ref_public.read_text(encoding="utf-8"))
    valid = json.loads(args.valid_public.read_text(encoding="utf-8"))
    pattern_labels, block_labels, matrix = build_primary_nam_matrix(read_rows(args.valid_block_private))

    stages = [
        ("Catálogo\nDISCOVERY", int(ref["n_frozen_target_sites"])),
        ("Algún portador\nNAM\nen REF", int(ref["n_sites_with_any_native_american_ref_carrier"])),
        ("Ambos bloques\nNAM\nde REF", int(ref["n_sites_with_carriers_in_both_native_american_ref_atomic_units"])),
        ("Fuera del\nbaseline\nhistórico", int(valid["n_primary_historical_baseline_disjoint_sites"])),
        ("Transferidos a\n3/3 bloques\nVALID", int(valid["n_primary_sites_transferred_all_nam_valid_blocks"])),
    ]

    fig = plt.figure(figsize=(13.4, 6.1), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.34)
    fig.subplots_adjust(top=0.78, bottom=0.18)
    ax_flow = fig.add_subplot(grid[0, 0])
    ax_matrix = fig.add_subplot(grid[0, 1])

    ink = "#25313C"
    blue = "#2D6A9F"
    pale_blue = "#DCEAF5"
    pale_gray = "#E9EDF0"
    border = "#7D8993"

    ax_flow.set_xlim(-0.5, len(stages) - 0.5)
    ax_flow.set_ylim(-0.45, 0.65)
    ax_flow.axis("off")
    for index, (label, value) in enumerate(stages):
        if index < len(stages) - 1:
            ax_flow.annotate(
                "",
                xy=(index + 0.72, 0.13),
                xytext=(index + 0.28, 0.13),
                arrowprops={"arrowstyle": "->", "color": border, "lw": 1.6},
            )
        face = pale_blue if value else pale_gray
        ax_flow.text(
            index,
            0.13,
            f"{value:,}".replace(",", "."),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=ink,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": face, "edgecolor": blue if value else border, "lw": 1.4},
        )
        ax_flow.text(index, -0.20, label, ha="center", va="top", fontsize=8.8, color=ink)
    ax_flow.set_title("Pérdida de soporte a través de los filtros independientes", loc="left", fontsize=12, color=ink, pad=12)

    array = np.asarray(matrix)
    display = np.where(array < 0, 0, array)
    ax_matrix.imshow(display, cmap=ListedColormap([pale_gray, blue]), vmin=0, vmax=1, aspect="equal")
    ax_matrix.set_xticks(range(3), block_labels, fontsize=9.5)
    ax_matrix.set_yticks(range(3), pattern_labels, fontsize=9.5)
    ax_matrix.tick_params(length=0)
    for row in range(3):
        for column in range(3):
            value = array[row, column]
            label = "Presente" if value == 1 else "Ausente" if value == 0 else "No evaluable"
            ax_matrix.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if value == 1 else "normal",
                color="white" if value == 1 else ink,
            )
    for edge in ax_matrix.spines.values():
        edge.set_color(border)
        edge.set_linewidth(0.8)
    ax_matrix.set_title("Transferencia en los tres bloques NAM de VALID", loc="left", fontsize=12, color=ink, pad=12)
    ax_matrix.set_xlabel("Bloques independientes (población + IBD)", fontsize=9.5, color=ink, labelpad=10)

    fig.suptitle("M27F: transferencia del catálogo raro de chr22", x=0.055, y=0.97, ha="left", fontsize=16, fontweight="bold", color=ink)
    fig.text(
        0.055,
        0.90,
        "Catálogo y orientación congelados antes de VALID; los tres patrones primarios estaban fuera del baseline histórico.",
        ha="left",
        fontsize=10.3,
        color="#52616D",
    )
    fig.text(
        0.055,
        0.025,
        "Resultado: 0/3 patrones alcanzaron los 3/3 bloques (regla preespecificada: al menos 2). TEST no fue analizado.",
        ha="left",
        fontsize=9.5,
        color=ink,
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render(parse_args())
