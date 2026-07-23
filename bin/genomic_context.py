#!/usr/bin/env python3
"""Funciones puras de contexto genómico usadas por el análisis de presencia externa.

El módulo no mantiene estado global:

  - load_bed_chrom : intervalos LCR/segdup de UN cromosoma -> (starts, ends) ordenados
  - in_bed         : ¿una posición 0-based cae en algún intervalo? (búsqueda binaria)
  - mut_context    : clase CpG/transición + contexto trinucleótido (hebra +) de un SNV
  - resolve_contig : nombre real del contig en un VCF (chr22 vs 22); falla ruidoso

Estas funciones aceptan indistintamente el estilo ``chr22`` o ``22`` para tolerar
headers de VCF/BED mezclados (hg38 UCSC vs Ensembl).
"""
from __future__ import annotations

import gzip
import logging

import numpy as np
import pysam
from cyvcf2 import VCF

LOG = logging.getLogger("genomic_context")

_PURINES, _PYRIMIDINES = {"A", "G"}, {"C", "T"}


def load_bed_chrom(path, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga intervalos BED (0-based, half-open) de UN cromosoma -> (starts, ends) ordenados.

    Acepta ``chr22`` o ``22`` en la 1.ª columna. Lee en streaming y retiene solo las filas
    del cromosoma pedido (memoria O(intervalos del cromosoma), no del archivo). Soporta ``.gz``.
    Devuelve arrays vacíos si no hay intervalos."""
    starts, ends = [], []
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            if f[0] != chrom and f[0] != alt:
                continue
            try:
                starts.append(int(f[1]))
                ends.append(int(f[2]))
            except ValueError:
                continue
    if not starts:
        LOG.warning("BED %s: 0 intervalos para %s", path, chrom)
        return np.empty(0, np.int64), np.empty(0, np.int64)
    s = np.array(starts, np.int64)
    e = np.array(ends, np.int64)
    order = np.argsort(s)
    return s[order], e[order]


def in_bed(pos0: int, starts: np.ndarray, ends: np.ndarray) -> bool:
    """¿pos0 (0-based) cae en algún intervalo [start, end)? ``starts`` ordenado -> búsqueda binaria.

    Maneja intervalos solapados/anidados: chequea el max ``end`` entre los que empiezan <= pos0."""
    if starts.size == 0:
        return False
    i = int(np.searchsorted(starts, pos0, side="right"))  # primer start > pos0
    if i == 0:
        return False
    return bool(np.any(ends[:i] > pos0))


def mut_context(fa: pysam.FastaFile, fa_chrom: str, pos0: int, ref: str, alt: str
                ) -> tuple[str, str]:
    """(cpg_class, trinuc_context). cpg_class ∈ {cpg_ti, noncpg_ti, transversion, non_snv}.

    trinuc = base previa + ref + base siguiente (hebra +). ``pos0`` es 0-based de la base del SNV."""
    prev_b = (fa.fetch(fa_chrom, max(pos0 - 1, 0), pos0).upper() or "N")
    next_b = (fa.fetch(fa_chrom, pos0 + 1, pos0 + 2).upper() or "N")
    trinuc = f"{prev_b}{ref}{next_b}"
    if ref not in _PURINES | _PYRIMIDINES or alt not in _PURINES | _PYRIMIDINES:
        return "non_snv", trinuc
    is_ti = (ref in _PURINES and alt in _PURINES) or (ref in _PYRIMIDINES and alt in _PYRIMIDINES)
    if not is_ti:
        return "transversion", trinuc
    if (ref == "C" and alt == "T" and next_b == "G") or (ref == "G" and alt == "A" and prev_b == "C"):
        return "cpg_ti", trinuc
    return "noncpg_ti", trinuc


def resolve_contig(vcf: VCF, chrom: str) -> str:
    """Resuelve el nombre real del contig en el VCF (chr22 vs 22). Falla ruidoso si no existe —
    evita que ``vcf('chr22')`` devuelva 0 registros en silencio cuando el header usa '22'."""
    names = set(vcf.seqnames)
    if chrom in names:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in names:
        LOG.warning("VCF usa contig '%s' (no '%s') -> usando '%s'", alt, chrom, alt)
        return alt
    raise ValueError(f"Ni '{chrom}' ni '{alt}' en el VCF (contigs: {sorted(names)[:6]}...)")
