#!/usr/bin/env python3
"""Concatena las 22 matrices sparse por-cromosoma (individuo x variante rara) en UNA matriz
genoma-completa (CSC, filas=muestras TRAIN en el orden del manifiesto, columnas=variantes de los 22
autosomas en orden cromosómico). Aplica el prefiltro de missingness sobre todo TRAIN. La extracción
imputó faltantes a cero, así que la máscara por individuo no puede recuperarse y este filtro no
puede ajustarse por fold. En la práctica es casi inerte: ninguna variante supera ~0.087 frente al
umbral de 0.10. MAC, portadores y varianza se vuelven a derivar por fold en la CV.

Validaciones:
  - El orden de filas debe ser idéntico en los 22 cromosomas y coincidir con TRAIN en el split.
  - Nunca densifica: hstack de CSC -> CSC.

Salidas:
  genome.rare_matrix.npz   matriz CSC int16 (2091 x sum_variantes_retenidas)
  genome.samples.tsv       orden de filas (sample_id)
  genome.variant_index.npy  int32 [n_cols x 2] (chrom, pos) para trazabilidad (la CV no lo necesita)
  genome.concat_summary.json  dimensiones, nnz, sparsity, descartes de missingness por cromosoma
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main():
    """Concatena matrices cromosómicas conservando muestras y formato sparse."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", required=True, type=Path,
                    help="dir con {chrom}.rare_matrix.npz / .samples.tsv / .variants.tsv")
    ap.add_argument("--chromosomes", required=True, help="lista separada por comas, en orden (1..22)")
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--sample-id-col", default="sample_id")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--train-label", default="TRAIN")
    ap.add_argument("--max-missing-train", type=float, default=0.10)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    chroms = [c.strip() for c in args.chromosomes.split(",") if c.strip()]
    if not chroms:
        raise SystemExit("[concat_rare_matrix] lista de cromosomas vacia")

    man = pd.read_csv(args.split_manifest, sep="\t", dtype={args.sample_id_col: str})
    train_ids = man.loc[man[args.split_col] == args.train_label, args.sample_id_col].tolist()

    blocks = []
    ref_samples = None
    chrom_idx = []
    pos_idx = []
    per_chr = []
    for c in chroms:
        npz = args.extract_dir / f"{c}.rare_matrix.npz"
        stsv = args.extract_dir / f"{c}.samples.tsv"
        vtsv = args.extract_dir / f"{c}.variants.tsv"
        for p in (npz, stsv, vtsv):
            if not p.exists():
                raise SystemExit(f"[concat_rare_matrix] falta {p}")
        samples = pd.read_csv(stsv, sep="\t", dtype={args.sample_id_col: str})[args.sample_id_col].tolist()
        if ref_samples is None:
            ref_samples = samples
            if ref_samples != train_ids:
                nd = sum(1 for a, b in zip(ref_samples, train_ids) if a != b)
                raise SystemExit(f"[concat_rare_matrix] orden de muestras chr{c} != orden TRAIN del split "
                                 f"(len {len(ref_samples)} vs {len(train_ids)}, {nd} posiciones distintas)")
        elif samples != ref_samples:
            raise SystemExit(f"[concat_rare_matrix] chr{c} tiene orden/identidad de muestras distinto a chr{chroms[0]}")

        X = sp.load_npz(npz).tocsc()
        if X.shape[0] != len(ref_samples):
            raise SystemExit(f"[concat_rare_matrix] chr{c}: filas {X.shape[0]} != n_train {len(ref_samples)}")
        v = pd.read_csv(vtsv, sep="\t", usecols=["chrom", "pos", "missing_rate_train"])
        if v.shape[0] != X.shape[1]:
            raise SystemExit(f"[concat_rare_matrix] chr{c}: variants.tsv {v.shape[0]} != columnas {X.shape[1]}")
        keep = (v["missing_rate_train"].to_numpy() <= args.max_missing_train)
        n_drop = int((~keep).sum())
        Xk = X[:, keep.nonzero()[0]].tocsc()
        blocks.append(Xk)
        chrom_idx.append(np.full(Xk.shape[1], int(str(c).replace("chr", "")) if str(c).replace("chr", "").isdigit() else -1, dtype=np.int32))
        pos_idx.append(v.loc[keep, "pos"].to_numpy(dtype=np.int32))
        per_chr.append({"chrom": c, "n_variants_input": int(X.shape[1]),
                        "n_dropped_missing": n_drop, "n_retained": int(Xk.shape[1]), "nnz": int(Xk.nnz)})

    genome = sp.hstack(blocks, format="csc")
    genome = genome.astype(np.int16)
    genome.eliminate_zeros()

    if genome.shape[0] != len(train_ids):
        raise SystemExit(f"[concat_rare_matrix] matriz genoma filas {genome.shape[0]} != TRAIN {len(train_ids)}")

    variant_index = np.column_stack([np.concatenate(chrom_idx), np.concatenate(pos_idx)]).astype(np.int32)
    sp.save_npz(args.outdir / "genome.rare_matrix.npz", genome)
    pd.DataFrame({"sample_id": ref_samples}).to_csv(args.outdir / "genome.samples.tsv", sep="\t", index=False)
    np.save(args.outdir / "genome.variant_index.npy", variant_index)

    summary = {
        "n_train_samples": int(genome.shape[0]),
        "n_chromosomes": len(chroms),
        "total_variants_input": int(sum(p["n_variants_input"] for p in per_chr)),
        "total_dropped_missing": int(sum(p["n_dropped_missing"] for p in per_chr)),
        "total_variants_retained": int(genome.shape[1]),
        "nnz": int(genome.nnz),
        "sparsity": round(1.0 - genome.nnz / (genome.shape[0] * genome.shape[1]), 8),
        "matrix_format": "csc", "matrix_dtype": str(genome.dtype),
        "max_missing_train_prefilter": args.max_missing_train,
        "sample_order_asserted_against_train_ids": True,
        "per_chr": per_chr,
    }
    (args.outdir / "genome.concat_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in
                      ("n_train_samples", "total_variants_input", "total_dropped_missing",
                       "total_variants_retained", "nnz", "sparsity")}, indent=2))


if __name__ == "__main__":
    main()
