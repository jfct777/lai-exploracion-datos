#!/usr/bin/env python3
"""Extrae la matriz sparse individuo x variante rara de UN cromosoma, solo para las muestras de
TRAIN. El fold de TEST no se decodifica: bcftools query -S recibe únicamente los ID de TRAIN,
de modo que el genotipo de TEST no se lee del VCF en ningun momento (bloqueo del fold 3 a nivel de
lectura, no solo a nivel de filtro posterior).

Fuente del label (M14): results_modtest_mac2/lai_rare/*.rare.vcf.gz (raras MAC>=2, biallelic split,
FORMAT=GT). El alt-dosage (0/1/2) se acumula en una CSC (n_train x n_variants); los ceros (incluidos
los faltantes imputados a cero no se materializan, por lo que se conserva la matriz sparse.

Se comprobó con bcftools 1.16 que `bcftools query -S <file>` conserva el orden del archivo de
muestras: por eso el orden de filas de la matriz == orden de train_samples.txt, deterministico.

Salidas:
  <chrom>.rare_matrix.npz     matriz CSC int16 (filas=muestras TRAIN en orden del manifiesto)
  <chrom>.samples.tsv         orden exacto de filas (sample_id)
  <chrom>.variants.tsv        por variante: variant_id, chrom, pos, ref, alt, mac_train,
                              n_alt_carriers_train, n_missing_train, missing_rate_train
  <chrom>.extract_summary.json  dimensiones, sparsity, nnz, chequeos del bloqueo de TEST
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Genotipos homocigotos-ref, el caso abrumadoramente mayoritario en variantes raras: se saltan por
# igualdad de cadena (rapido) antes de cualquier parseo. Cubre no-fase y fase.
_HOMREF = {"0/0", "0|0"}


def _fail(msg):
    sys.stderr.write(f"[extract_rare_matrix_chr] ERROR: {msg}\n")
    raise SystemExit(1)


def _alt_dosage(gt):
    """Alt-dosage (0/1/2) de un GT biallelic; None si es faltante. Cuenta el alelo '1' porque el VCF
    esta biallelic-split (mac2 lai_rare) -> el unico alelo alternativo es el indice 1."""
    if "." in gt:
        return None
    return gt.count("1")


def _emitted_sample_order(vcf, samples_file):
    """Devuelve el orden de muestras que emite `bcftools query -S` y que usa el encabezado del subset.
    (`bcftools view -h -S`). Se usa para comprobar identidad y orden frente a train_ids antes de construir
    la matriz: si difiere, la asignacion columna->muestra seria incorrecta -> se aborta fail-closed."""
    proc = subprocess.run(["bcftools", "view", "-h", "-S", str(samples_file), str(vcf)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        _fail(f"bcftools view -h -S fallo al resolver el orden de muestras: {proc.stderr.strip()}")
    header = [ln for ln in proc.stdout.splitlines() if ln.startswith("#CHROM")]
    if not header:
        _fail("no se encontro la linea #CHROM en el header subseteado")
    return header[0].split("\t")[9:]


def main():
    """Extrae la matriz rara de un cromosoma para las muestras de entrenamiento."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, type=Path, help="VCF rare de un cromosoma (bgzip+tbi)")
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--chrom", required=True, help="etiqueta del cromosoma (p.ej. 22)")
    ap.add_argument("--sample-id-col", default="sample_id")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--train-label", default="TRAIN")
    ap.add_argument("--test-label", default="TEST")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sp_df = pd.read_csv(args.split_manifest, sep="\t", dtype={args.sample_id_col: str})
    for col in (args.sample_id_col, args.split_col):
        if col not in sp_df.columns:
            _fail(f"columna '{col}' ausente en {args.split_manifest}")

    train_ids = sp_df.loc[sp_df[args.split_col] == args.train_label, args.sample_id_col].tolist()
    test_ids = set(sp_df.loc[sp_df[args.split_col] == args.test_label, args.sample_id_col])
    if not train_ids:
        _fail(f"0 muestras con split=='{args.train_label}' en {args.split_manifest}")

    # Bloqueo del fold 3: ningún ID de TRAIN puede pertenecer también a TEST. La lista enviada a
    # bcftools -S contiene únicamente TRAIN; si esto falla, se aborta antes de leer el VCF.
    leak = sorted(set(train_ids) & test_ids)
    if leak:
        _fail(f"{len(leak)} IDs aparecen en TRAIN y TEST a la vez (fuga): {leak[:5]}...")

    samples_file = args.outdir / f"{args.chrom}.train_samples.txt"
    samples_file.write_text("\n".join(train_ids) + "\n")
    n_train = len(train_ids)

    # Se comprueban identidad y orden: las muestras que bcftools emite con -S deben coincidir con
    # train_ids, en el mismo orden. Sin esto, el indice de columna i del stream podria no corresponder
    # a train_ids[i] (asignacion muestra->fila incorrecta). Fail-closed antes de leer un solo genotipo.
    emitted = _emitted_sample_order(args.vcf, samples_file)
    if emitted != train_ids:
        n_diff = sum(1 for a, b in zip(emitted, train_ids) if a != b)
        _fail(f"orden/identidad de muestras emitidas por bcftools != train_ids "
              f"(emitidas={len(emitted)}, train={len(train_ids)}, posiciones distintas={n_diff}); "
              f"no se construye la matriz")

    # Se leen los genotipos de TRAIN. -S restringe las columnas en bcftools, por lo que el GT de
    # TEST no se emite ni se decodifica. El orden de columnas == orden de train_samples.txt (verificado).
    query_fmt = r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n"
    cmd = ["bcftools", "query", "-S", str(samples_file), "-f", query_fmt, str(args.vcf)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1 << 20)

    rows = []          # indices de fila (muestra) de las entradas no-cero
    cols = []          # indices de columna (variante)
    data = []          # alt-dosage (1 o 2)
    var_records = []   # metadatos por variante
    n_variants = 0
    try:
        for line in proc.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4 + n_train:
                proc.kill()
                _fail(f"columnas inesperadas: {len(fields)} (esperado {4 + n_train}) en variante {n_variants}")
            chrom, pos, ref, alt = fields[0], fields[1], fields[2], fields[3]
            gts = fields[4:]
            j = n_variants
            mac = 0
            n_carriers = 0
            n_missing = 0
            for i, g in enumerate(gts):
                if g in _HOMREF:
                    continue
                d = _alt_dosage(g)
                if d is None:
                    n_missing += 1        # faltante -> se imputa a 0 (no-portador), sparse-safe
                    continue
                if d > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(d)
                    mac += d
                    n_carriers += 1
            var_records.append({
                "variant_id": f"{chrom}:{pos}:{ref}:{alt}",
                "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                "mac_train": mac, "n_alt_carriers_train": n_carriers,
                "n_missing_train": n_missing,
                "missing_rate_train": round(n_missing / n_train, 6),
            })
            n_variants += 1
    finally:
        proc.stdout.close()
    ret = proc.wait()
    stderr = proc.stderr.read()
    proc.stderr.close()
    if ret != 0:
        _fail(f"bcftools query fallo (exit {ret}): {stderr.strip()}")
    if n_variants == 0:
        _fail(f"0 variantes leidas de {args.vcf}")

    matrix = sp.csc_matrix(
        (np.asarray(data, dtype=np.int16), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_train, n_variants),
    )
    matrix.eliminate_zeros()

    variants = pd.DataFrame(var_records)
    variants.to_csv(args.outdir / f"{args.chrom}.variants.tsv", sep="\t", index=False)
    pd.DataFrame({"sample_id": train_ids}).to_csv(args.outdir / f"{args.chrom}.samples.tsv", sep="\t", index=False)
    sp.save_npz(args.outdir / f"{args.chrom}.rare_matrix.npz", matrix)

    summary = {
        "chrom": args.chrom,
        "source_vcf": str(args.vcf),
        "n_train_samples": n_train,
        "n_variants_input": n_variants,
        "nnz": int(matrix.nnz),
        "sparsity": round(1.0 - matrix.nnz / (n_train * n_variants), 8),
        "matrix_format": "csc",
        "matrix_dtype": str(matrix.dtype),
        "sample_order_asserted_against_train_ids": True,
        "fold3_block": {
            "train_ids_requested": n_train,
            "test_ids_in_manifest": len(test_ids),
            "train_test_overlap": len(leak),
            "note": "bcftools query -S recibio solo IDs de TRAIN; TEST nunca se decodifico del VCF",
        },
    }
    (args.outdir / f"{args.chrom}.extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
