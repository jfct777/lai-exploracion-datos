#!/usr/bin/env python3
"""Variantes raras sobre el painting de ancestria local (LAI) — vista por-haplotipo.

Tres pistas alineadas en el eje genomico (proporcional en bp):

  (a) Karyograma: painting de ancestria local de TODOS los haplotipos de la cohorte
      (Gnomix, Nunes 2025), una fila por haplotipo, ordenados por fraccion de
      ancestria del cromosoma (gradiente continuo, sin etiqueta argmax). Hace
      visibles los tractos discretos AFR/EUR/NAM que el promedio poblacional borra.

  (b) Densidad de variantes raras estratificada por ancestria local: copias de
      alelo raro (MAC>=2) por Mb, separadas segun la ancestria local del tracto
      donde caen (solo copias con atribucion EXACTA; las fase-ambiguas se excluyen).

  (c) Zoom: pocos individuos representativos del gradiente, con sus dos haplotipos
      pintados y sus variantes raras portadas como ticks. Sin fase -> la marca es
      a nivel de individuo, no de haplotipo.

El painting tiene 3 ancestrias por diseno del panel de referencia (EAS no existe
en el LAI). Solo resultados: sin texto operativo ni referencias a modulos/scripts.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# Okabe-Ito, consistente con el resto del set. Codigos .msp: AFR=0, EUR=1, NAM=2.
COL = ["#D55E00", "#0072B2", "#009E73"]
ANC_EN = ["African", "European", "Native American"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
})


def load_crosswalk(metadata_path):
    """Retorna lai_to_id (Cohort_ID -> ID==muestra del VCF) para los individuos."""
    df = pd.read_csv(metadata_path, sep="\t", dtype=str).drop_duplicates(subset="ID")
    return {f"{r.Cohort}_{r.ID}": r.ID for r in df.itertuples()}


def parse_msp(msp_path, valid_lai):
    """Lee el .msp; retiene haplotipos cuyo LAI_id esta en valid_lai.

    Retorna spos, epos (1-based asc), codes [n_seg, n_hap_keep] int8,
    y lai_ids (lista paralela a las columnas de codes, .0 y .1 consecutivos).
    """
    with open(msp_path) as fh:
        line1 = fh.readline()
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
    assert "African=0" in line1 and "European=1" in line1 and "Native-American=2" in line1, \
        f"Codigos inesperados: {line1!r}"
    hap_start = next(i for i, c in enumerate(header) if c.endswith(".0"))
    hap_headers = header[hap_start:]

    col0, col1 = {}, {}
    for j, h in enumerate(hap_headers):
        lai_id, hap = h.rsplit(".", 1)
        if lai_id not in valid_lai:
            continue
        (col0 if hap == "0" else col1)[lai_id] = j
    common = sorted(set(col0) & set(col1))
    keep, lai_per_col = [], []
    for lid in common:
        keep.extend([col0[lid], col1[lid]])
        lai_per_col.extend([lid, lid])

    data = pd.read_csv(msp_path, sep="\t", skiprows=1)
    spos = data.iloc[:, 1].to_numpy(np.int64)
    epos = data.iloc[:, 2].to_numpy(np.int64)
    codes = data.iloc[:, hap_start:].to_numpy(np.int8)[:, keep]
    assert np.all(np.diff(spos) > 0), ".msp: segmentos no ordenados"
    print(f"[msp] {len(spos)} segmentos, {len(common)} individuos retenidos", file=sys.stderr)
    return spos, epos, codes, np.array(lai_per_col), common


def seg_index_at(spos, epos, positions):
    """Indice de segmento que contiene cada posicion (-1 si fuera de todo segmento)."""
    idx = np.searchsorted(spos, positions, side="right") - 1
    idx = np.clip(idx, 0, len(spos) - 1)
    bad = positions > epos[idx]
    idx[bad] = -1
    return idx


def indiv_ancestry_fraction(spos, epos, codes, lai_per_col, common):
    """Fraccion de bp del cromosoma en cada ancestria, por individuo (2 haplotipos)."""
    seg_bp = (epos - spos + 1).astype(np.float64)
    frac = {}
    col_of = {}
    for j, lid in enumerate(lai_per_col):
        col_of.setdefault(lid, []).append(j)
    for lid in common:
        c = col_of[lid]
        f = np.zeros(3)
        for a in range(3):
            mask = (codes[:, c] == a)  # [n_seg, 2]
            f[a] = (seg_bp[:, None] * mask).sum()
        # Normaliza por el bp efectivamente pintado del individuo (robusto si el
        # painting no cubre uniformemente todos los segmentos).
        tot = f.sum()
        frac[lid] = f / tot if tot > 0 else f
    return frac


def carrier_positions(bcftools, rare_vcf, vcf_chrom, sample, lo, hi):
    """Posiciones donde `sample` porta >=1 alelo raro, en [lo, hi]."""
    cmd = [bcftools, "query", "-s", sample, "-i", 'GT="alt"',
           "-r", f"{vcf_chrom}:{lo}-{hi}", "-f", "%POS\n", rare_vcf]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return np.array([int(x) for x in out.split()], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msp", required=True)
    ap.add_argument("--windows_tsv", required=True, help="<prefix>.windows.tsv de M17 extendido")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--rare_vcf", required=True)
    ap.add_argument("--vcf_chrom", default="chr22")
    ap.add_argument("--chrom", default="22")
    ap.add_argument("--bcftools", default="bcftools")
    ap.add_argument("--n_grid", type=int, default=1600, help="bins de muestreo del painting (panel a)")
    ap.add_argument("--zoom_mb", type=float, default=4.0, help="ancho de la region de zoom (Mb)")
    ap.add_argument("--n_zoom_per_group", type=int, default=2)
    ap.add_argument("--panel_c", choices=["zoom", "full"], default="zoom",
                    help="panel c: region de zoom o cromosoma completo")
    ap.add_argument("--panel_c_select", choices=["extremes", "mosaic"], default="extremes",
                    help="individuos del panel c: extremos del gradiente o los mas mosaico")
    ap.add_argument("--individuals", default="",
                    help="Lista CSV de IDs (LAI_id 'Cohort_ID' o sample ID) a mostrar en el "
                         "panel c, en ese orden. Si se da, ignora --panel_c_select.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out_prefix", default=None)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"fig_raras_karyograma_lai_chr{args.chrom}"

    lai_to_id = load_crosswalk(args.metadata)
    spos, epos, codes, lai_per_col, common = parse_msp(args.msp, set(lai_to_id))
    painted_min, painted_max = int(spos[0]), int(epos[-1])

    frac = indiv_ancestry_fraction(spos, epos, codes, lai_per_col, common)
    # Orden continuo: NAM desc, luego AFR desc (gradiente, NO etiqueta argmax).
    order = sorted(common, key=lambda l: (-frac[l][2], -frac[l][0]))

    # --- matriz del karyograma (panel a): [hap_row, bin] = codigo de ancestria ---
    grid = np.linspace(painted_min, painted_max, args.n_grid)
    seg_at = seg_index_at(spos, epos, grid)
    col_of = {}
    for j, lid in enumerate(lai_per_col):
        col_of.setdefault(lid, []).append(j)
    rows = []
    for lid in order:
        for j in col_of[lid]:
            rows.append(codes[seg_at, j])
    karyo = np.array(rows, dtype=np.int8)  # [2*n_indiv, n_grid]
    mask = np.broadcast_to(seg_at[None, :] < 0, karyo.shape)
    karyo_masked = np.ma.masked_where(mask, karyo)

    # --- densidad estratificada (panel b) ---
    win = pd.read_csv(args.windows_tsv, sep="\t")
    w_mid = (win["window_start"] + win["window_end"]) / 2.0 / 1e6
    w_width_mb = (win["window_end"] - win["window_start"] + 1) / 1e6
    dens = {a: win[f"copies_exact_{k}"] / w_width_mb
            for a, k in zip(range(3), ["African", "European", "Native_American"])}

    # --- seleccion de individuos para el panel c ---
    common_set = set(common)
    if args.individuals.strip():
        # Prioridad: lista explicita. Acepta LAI_id (Cohort_ID) o sample ID.
        id_to_lai = {v: k for k, v in lai_to_id.items()}
        zoom_ids = []
        for tok in (t.strip() for t in args.individuals.split(",") if t.strip()):
            if tok in common_set:
                zoom_ids.append(tok)
            elif tok in id_to_lai and id_to_lai[tok] in common_set:
                zoom_ids.append(id_to_lai[tok])
            else:
                print(f"[warn] individuo no encontrado, se omite: {tok}", file=sys.stderr)
        if not zoom_ids:
            raise ValueError("--individuals no resolvio ningun individuo valido")
    elif args.panel_c_select == "mosaic":
        # Mas mosaico = mas cambios de ancestria local entre segmentos contiguos
        # (sumados sobre los 2 haplotipos) -> resalta tractos minoritarios que quiebran.
        switches = {l: sum(int(np.sum(np.diff(codes[:, j].astype(int)) != 0)) for j in col_of[l])
                    for l in common}
        zoom_ids = sorted(common, key=lambda l: -switches[l])[:6]
    else:
        by_nam = sorted(common, key=lambda l: -frac[l][2])
        by_afr = sorted(common, key=lambda l: -frac[l][0])
        by_eur = sorted(common, key=lambda l: -frac[l][1])
        k = args.n_zoom_per_group
        zoom_ids, seen = [], set()
        for lst in (by_nam, by_afr, by_eur):
            added = 0
            for lid in lst:
                if lid not in seen:
                    zoom_ids.append(lid); seen.add(lid); added += 1
                if added >= k:
                    break

    # --- region del panel c ---
    if args.panel_c == "full":
        z_lo, z_hi = painted_min, painted_max
    else:
        nam_terr = win["territory_bp_Native_American"].to_numpy()
        center = (win["window_start"] + win["window_end"]).to_numpy()[int(np.argmax(nam_terr))] / 2.0
        half = args.zoom_mb * 1e6 / 2.0
        z_lo = max(painted_min, int(center - half)); z_hi = min(painted_max, int(center + half))

    lo_mb, hi_mb = painted_min / 1e6, painted_max / 1e6

    # ===================== FIGURA =====================
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 0.85, 1.25], hspace=0.32)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2])

    cmap = ListedColormap(COL)
    ax_a.imshow(karyo_masked, aspect="auto", interpolation="nearest", cmap=cmap,
                vmin=0, vmax=2, extent=[lo_mb, hi_mb, karyo.shape[0], 0])
    ax_a.set_ylabel(f"Haplotypes (n = {karyo.shape[0]})")
    ax_a.set_yticks([])
    ax_a.tick_params(labelbottom=False)
    ax_a.set_xlim(lo_mb, hi_mb)
    leg = [Patch(facecolor=COL[a], label=ANC_EN[a]) for a in range(3)]
    ax_a.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
                frameon=False, handlelength=1.1, columnspacing=1.4)
    ax_a.text(-0.075, 1.02, "a", transform=ax_a.transAxes, fontsize=11, fontweight="bold")

    ax_b.stackplot(w_mid, dens[0], dens[1], dens[2], colors=COL, edgecolor="none")
    ax_b.set_xlim(lo_mb, hi_mb)
    ax_b.set_ylabel("Rare-allele copies\nper Mb")
    ax_b.set_xlabel(f"Chromosome {args.chrom} — position (Mb)")
    ax_b.text(-0.075, 1.02, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold")
    # Banda del zoom marcada sobre el painting y la densidad (solo si es zoom).
    if args.panel_c == "zoom":
        for ax in (ax_a, ax_b):
            ax.axvspan(z_lo / 1e6, z_hi / 1e6, color="none", ec="black", lw=0.8, ls=(0, (4, 2)))
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)

    # --- Panel c: zoom por individuo ---
    bar_h, gap = 0.34, 0.55
    yticks, ylabels = [], []
    for r, lid in enumerate(zoom_ids):
        y_base = r * (2 * bar_h + gap)
        sample = lai_to_id[lid]
        # Posiciones raras portadas por el individuo (sin fase: nivel individuo).
        cp = carrier_positions(args.bcftools, args.rare_vcf, args.vcf_chrom, sample, z_lo, z_hi)
        for hi_, j in enumerate(col_of[lid]):  # dos haplotipos
            y = y_base + hi_ * bar_h
            # segmentos dentro del zoom: dibuja barras pintadas
            for i in range(len(spos)):
                if epos[i] < z_lo or spos[i] > z_hi:
                    continue
                s = max(spos[i], z_lo) / 1e6
                e = min(epos[i], z_hi) / 1e6
                ax_c.add_patch(plt.Rectangle((s, y), e - s, bar_h * 0.9,
                                             facecolor=COL[int(codes[i, j])], edgecolor="none"))
        # rug de raras del individuo, bajo sus dos barras
        if cp.size:
            ms = 3 if args.panel_c == "zoom" else 1.6
            al = 0.6 if args.panel_c == "zoom" else 0.4
            ax_c.plot(cp / 1e6, np.full(cp.size, y_base - 0.10), "|", color="black",
                      markersize=ms, markeredgewidth=0.4, alpha=al)
        yticks.append(y_base + bar_h)
        f = frac[lid]
        ylabels.append(f"{lid}\nAFR {f[0]:.0%}  EUR {f[1]:.0%}  NAM {f[2]:.0%}")
    ax_c.set_xlim(z_lo / 1e6, z_hi / 1e6)
    ax_c.set_ylim(-0.4, len(zoom_ids) * (2 * bar_h + gap))
    ax_c.invert_yaxis()
    ax_c.set_yticks(yticks); ax_c.set_yticklabels(ylabels, fontsize=6)
    ax_c.set_xlabel(f"Chromosome {args.chrom} — position (Mb)")
    ax_c.text(-0.075, 1.02, "c", transform=ax_c.transAxes, fontsize=11, fontweight="bold")
    for sp in ("top", "right", "left"):
        ax_c.spines[sp].set_visible(False)
    ax_c.tick_params(left=False)

    # eje X de a (arriba) sin labels; ponemos el de b al pie de b via panel c separado.
    ax_a.xaxis.set_major_locator(MultipleLocator(5))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{prefix}.{ext}", dpi=args.dpi)
    plt.close(fig)
    print(f"[OK] {prefix}: {karyo.shape[0]} haplotipos, painting {lo_mb:.1f}-{hi_mb:.1f} Mb; "
          f"zoom {z_lo/1e6:.1f}-{z_hi/1e6:.1f} Mb, {len(zoom_ids)} individuos")


if __name__ == "__main__":
    main()
