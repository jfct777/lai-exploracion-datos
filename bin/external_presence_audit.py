#!/usr/bin/env python3
"""Módulo 21 — canal de presencia externa por cromosoma y panel.

La interpretación es asimétrica: si un alelo raro de DNABR aparece en un panel NAM externo, no es
privado de DNABR y descarta privacidad; su ausencia del panel no confirma founder porque intervienen
ausencia por muestreo y artefactos de calling). El canal defendible es binario: present/unknown.
No se aplica un modelo Beta-Binomial ni un tercer estado de callability.

Clasifica cada SNV raro bialélico de DNABR (target) de UN cromosoma contra UN panel en:
  - PRESENT_ALLELE   : el panel tiene fila en esa POS y el ALT target está entre sus alelos (AC>0).
  - PRESENT_POS_ONLY : el panel tiene una fila en esa posición y el mismo REF, pero no contiene el ALT
                       objetivo. Es un caso ambiguo y no cuenta como presencia del alelo.
  - REF_MISMATCH     : hay fila del panel en esa POS pero con REF distinto al target (no es la misma
                       variante; en hg38 left-aligned debe ser ~0 -> aserción de QC).
  - ABSENT_FROM_PANEL: no hay fila del panel en esa POS (variante no segrega / no interrogada).

Validaciones aplicadas:
  (1) PROVENANCE incluye el sha256 del propio script (reproducible aunque el commit no lo fije aún).
  (4) contador REF_MISMATCH (esperado ~0 en VCFs hg38 normalizados; >0 = normalización aguas arriba).
  (5) conteo PASS/non-PASS del panel + ``--panel-pass-only`` para que la comparación entre paneles de
      distinto régimen de filtrado (VQSR-PASS frente a .raw) sea comparable y no quede confundida.
  (EC57) sensibilidad CON/SIN una muestra (p.ej. EC57, 1 DNABR dentro del panel NAMBR): atribución
      exacta por genotipo en UNA pasada; reporta cuántos PRESENT_ALLELE se pierden al excluirla.

No convierte bp a cM, no infiere edad, no fasea, no usa métodos espectrales, no calcula AUC ni estima DAF con
un modelo. Solo observables + comparación honesta.

Salidas en outdir:
  <prefix>.external_presence.audit.tsv.gz   tabla por target (observables + flags EC57)
  <prefix>.external_presence.summary.json   conteos + estratificación + delta EC57 + filtro del panel
  <prefix>.manifest.json                    panel_id/version, schema, sha256, drop_sample, pass_only
  <prefix>.PROVENANCE.md
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pysam
from cyvcf2 import VCF

# Reutiliza funciones de contexto genómico. En Nextflow ambos scripts se preparan en el
# mismo work dir (path inputs) -> ``__file__``.parent los encuentra; standalone, viven juntos en bin/.
sys.path.insert(0, str(Path(__file__).parent))  # .parent apunta al directorio de trabajo donde
# Nextflow prepara ambos scripts como enlaces simbólicos; resolve() seguiría el enlace al bin/ del repo, que
# puede no estar montado dentro del contenedor Singularity.
from genomic_context import (  # noqa: E402
    load_bed_chrom,
    in_bed,
    mut_context,
    resolve_contig,
)

LOG = logging.getLogger("external_presence")

SCHEMA_VERSION = "external_presence_v2"

STATES = ("PRESENT_ALLELE", "PRESENT_POS_ONLY", "REF_MISMATCH", "ABSENT_FROM_PANEL")


# --------------------------------------------------------------------- preindexado del panel
def _accum_slot(alts: dict, a: str, ac_full: int, ac_drop: int) -> None:
    """Inserta o actualiza el alelo ``a`` tomando el máximo, sin sumar. Si una posición aparece en
    varias filas del panel (multialélico split por ``bcftools norm -m-any``), sumar haría doble
    conteo de AC y doble descuento del drop (ac_drop negativo). MAX es coherente con el an por-POS."""
    slot = alts.get(a)
    if slot is None:
        alts[a] = {"ac_full": ac_full, "ac_drop": ac_drop}
    else:
        slot["ac_full"] = max(slot["ac_full"], ac_full)
        slot["ac_drop"] = max(slot["ac_drop"], ac_drop)


def index_panel(panel_vcf: Path, chrom: str, drop_sample: str | None, pass_only: bool
                ) -> tuple[dict[int, dict], dict]:
    """pos(1-based) -> {an_full, an_drop, ref, alts:{ALT:{ac_full,ac_drop}}}.

    AC/AN de los genotipos, no del campo INFO. Solo usa SNV con REF y ALT de una base. Para
    multialélicos guarda AC por ALT. ``drop_sample`` (p.ej. EC57) se atribuye por genotipo en la
    misma pasada: ac_drop = ac_full - contribución de la muestra; an_drop = an_full - 2 si fue llamada.

    ``pass_only``: si True, omite filas cuyo FILTER no sea PASS (cyvcf2: ``var.FILTER is None`` == PASS).
    Siempre cuenta n_pass / n_nonpass del panel para separar el efecto del filtro del tamaño muestral."""
    vcf = VCF(str(panel_vcf), gts012=True)
    samples = list(vcf.samples)
    n_samples = len(samples)
    drop_idx = samples.index(drop_sample) if (drop_sample and drop_sample in samples) else None
    LOG.info("panel: %d muestras; drop_sample=%s (idx=%s); pass_only=%s",
             n_samples, drop_sample, drop_idx, pass_only)

    it_chrom = resolve_contig(vcf, chrom)
    idx: dict[int, dict] = {}
    n_rows = n_snv = n_pass = n_nonpass = 0
    for var in vcf(it_chrom):
        n_rows += 1
        # cyvcf2 devuelve None para PASS y para '.', por lo que no permite distinguirlos.
        # var.FILTERS sí (verificado en datos reales 2026-06-22): PASS->['PASS']; '.' (raw)->[];
        # rechazo->['<etiqueta>']. Para comparar paneles, PASS debe aparecer de forma explícita;
        # así el panel .raw (todo '.') cuenta como non-PASS y frac_nonpass documenta el régimen.
        is_pass = "PASS" in var.FILTERS
        if is_pass:
            n_pass += 1
        else:
            n_nonpass += 1
        if pass_only and not is_pass:
            continue
        if var.REF is None or len(var.REF) != 1:
            continue
        alts_raw = var.ALT
        snv_alts = [(k, a.upper()) for k, a in enumerate(alts_raw, start=1) if len(a) == 1]
        if not snv_alts:
            continue

        gt = var.gt_types  # 0 HOMREF, 1 HET, 2 HOMALT, 3 UNKNOWN
        n_called = int(np.count_nonzero(gt != 3))
        an_full = 2 * n_called
        if an_full == 0:
            continue

        an_drop = an_full
        drop_gt = None
        if drop_idx is not None:
            drop_gt = var.genotypes[drop_idx]  # [a1, a2, phase]
            if drop_gt[0] >= 0 and drop_gt[1] >= 0:
                an_drop = an_full - 2

        ref = var.REF.upper()
        rec = idx.setdefault(var.POS, {"an_full": an_full, "an_drop": an_drop,
                                       "ref": ref, "alts": {}})
        rec["an_full"] = max(rec["an_full"], an_full)
        rec["an_drop"] = max(rec["an_drop"], an_drop)

        if len(alts_raw) == 1 and len(snv_alts) == 1:
            a = snv_alts[0][1]
            ac_full = int(np.count_nonzero(gt == 1) + 2 * np.count_nonzero(gt == 2))
            ac_drop = ac_full
            if drop_gt is not None:
                ac_drop = max(0, ac_full - int((drop_gt[0] == 1) + (drop_gt[1] == 1)))
            _accum_slot(rec["alts"], a, ac_full, ac_drop)
            n_snv += 1
        else:
            geno = var.genotypes
            for k, a in snv_alts:
                ac_full = 0
                for g in geno:
                    ac_full += (g[0] == k) + (g[1] == k)
                ac_full = int(ac_full)
                ac_drop = ac_full
                if drop_gt is not None:
                    ac_drop = max(0, ac_full - int((drop_gt[0] == k) + (drop_gt[1] == k)))
                _accum_slot(rec["alts"], a, ac_full, ac_drop)
            n_snv += 1

    LOG.info("panel %s: %d filas (%d PASS / %d non-PASS), %d sitios SNV (%d posiciones)",
             chrom, n_rows, n_pass, n_nonpass, n_snv, len(idx))
    meta = {"n_samples": n_samples, "drop_sample": drop_sample,
            "drop_sample_found": drop_idx is not None, "drop_idx": drop_idx,
            "pass_only": pass_only, "n_rows": n_rows, "n_pass": n_pass, "n_nonpass": n_nonpass}
    return idx, meta


def classify(panel_rec, ref_target: str, alt_target: str, use_drop: bool) -> tuple[str, int, int]:
    """(state, ac, an) del target contra una fila del panel. use_drop -> usa conteos sin el sample.

    state ∈ STATES. REF_MISMATCH si el panel tiene la POS con un REF distinto (no es la misma
    variante; aserción de QC, esperado ~0 en hg38 left-aligned)."""
    if panel_rec is None:
        return "ABSENT_FROM_PANEL", 0, 0
    an = panel_rec["an_drop"] if use_drop else panel_rec["an_full"]
    if panel_rec["ref"] != ref_target:
        return "REF_MISMATCH", 0, an
    slot = panel_rec["alts"].get(alt_target)
    if slot is None:
        return "PRESENT_POS_ONLY", 0, an
    ac = slot["ac_drop"] if use_drop else slot["ac_full"]
    if ac > 0:
        return "PRESENT_ALLELE", ac, an
    # El alelo existía en el panel completo, pero todo su soporte venía de la muestra excluida.
    return "PRESENT_POS_ONLY", 0, an


def file_sha256(path: Path) -> str:
    """Calcula sha256 de un archivo por bloques."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    """Devuelve el commit actual del repositorio o una cadena vacía."""
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    """Audita presencia externa por panel y escribe resultados y procedencia."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dnabr-rare-vcf", required=True, type=Path)
    p.add_argument("--panel-vcf", required=True, type=Path)
    p.add_argument("--panel-id", required=True, help="p.ej. NAMBR_128_hg38_vqsr o NAM_native_71_raw")
    p.add_argument("--panel-version", required=True)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--lcr-segdup-bed", required=True, type=Path)
    p.add_argument("--chrom", default="chr22")
    p.add_argument("--drop-sample", default=None,
                   help="muestra a excluir para la sensibilidad (p.ej. EC57_EC57); None = sin drop")
    p.add_argument("--panel-pass-only", action="store_true",
                   help="indexa solo las filas PASS del panel para comparar regímenes de filtrado")
    p.add_argument("--out-prefix", required=True, help="prefijo de los archivos de salida")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    fa = pysam.FastaFile(str(args.fasta))
    fa_chrom = args.chrom if args.chrom in fa.references else args.chrom.replace("chr", "")
    if fa_chrom not in fa.references:
        raise ValueError(f"{args.chrom} no está en el FASTA {args.fasta}")

    lcr_s, lcr_e = load_bed_chrom(args.lcr_segdup_bed, args.chrom)
    panel, panel_meta = index_panel(args.panel_vcf, args.chrom, args.drop_sample, args.panel_pass_only)
    has_drop = panel_meta["drop_sample_found"]

    out_path = args.outdir / f"{args.out_prefix}.external_presence.audit.tsv.gz"
    cols = ["panel_id", "chrom", "pos", "ref", "alt_target",
            "state_full", "AC_full", "AN_full",
            "state_drop", "AC_drop", "AN_drop",
            "lost_when_drop", "cpg_class", "trinuc_context", "lcr_segdup_flag"]

    counts_full = {s: 0 for s in STATES}
    counts_drop = {s: 0 for s in STATES}
    present_in_lcr = present_out_lcr = present_cpg = present_noncpg_ti = present_tv = 0
    lost_when_drop = 0
    n_target = n_in_lcr = n_cpg = 0

    vcf = VCF(str(args.dnabr_rare_vcf), gts012=True)
    dn_chrom = resolve_contig(vcf, args.chrom)
    with gzip.open(out_path, "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for var in vcf(dn_chrom):
            if var.REF is None or len(var.REF) != 1 or len(var.ALT) != 1 or len(var.ALT[0]) != 1:
                continue  # solo SNVs bialélicos raros (target)
            ref, alt = var.REF.upper(), var.ALT[0].upper()
            pos, pos0 = var.POS, var.start
            n_target += 1

            rec = panel.get(pos)
            state_full, ac_full, an_full = classify(rec, ref, alt, use_drop=False)
            if has_drop:
                state_drop, ac_drop, an_drop = classify(rec, ref, alt, use_drop=True)
            else:
                state_drop, ac_drop, an_drop = state_full, ac_full, an_full
            counts_full[state_full] += 1
            counts_drop[state_drop] += 1
            lost = int(state_full == "PRESENT_ALLELE" and state_drop != "PRESENT_ALLELE")
            lost_when_drop += lost

            cpg_class, trinuc = mut_context(fa, fa_chrom, pos0, ref, alt)
            lcr_flag = int(in_bed(pos0, lcr_s, lcr_e))
            n_in_lcr += lcr_flag
            n_cpg += int(cpg_class == "cpg_ti")

            if state_full == "PRESENT_ALLELE":
                if lcr_flag:
                    present_in_lcr += 1
                else:
                    present_out_lcr += 1
                if cpg_class == "cpg_ti":
                    present_cpg += 1
                elif cpg_class == "noncpg_ti":
                    present_noncpg_ti += 1
                elif cpg_class == "transversion":
                    present_tv += 1

            fh.write("\t".join(map(str, [
                args.panel_id, args.chrom, pos, ref, alt,
                state_full, ac_full, an_full,
                state_drop, ac_drop, an_drop,
                lost, cpg_class, trinuc, lcr_flag])) + "\n")
    fa.close()

    n_present = counts_full["PRESENT_ALLELE"]
    frac_present_in_lcr = present_in_lcr / max(n_present, 1)
    frac_target_in_lcr = n_in_lcr / max(n_target, 1)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "chrom": args.chrom,
        "panel_id": args.panel_id,
        "panel_version": args.panel_version,
        "panel_n_samples": panel_meta["n_samples"],
        "panel_filter": {
            "pass_only_applied": panel_meta["pass_only"],
            "n_rows": panel_meta["n_rows"],
            "n_pass": panel_meta["n_pass"],
            "n_nonpass": panel_meta["n_nonpass"],
            "frac_nonpass": panel_meta["n_nonpass"] / max(panel_meta["n_rows"], 1),
        },
        "n_target_rare_snv": n_target,
        "n_target_in_lcr_segdup": n_in_lcr,
        "n_target_cpg_ti": n_cpg,
        "counts_full": counts_full,
        "ref_mismatch": counts_full["REF_MISMATCH"],
        "fraction_present_allele": n_present / max(n_target, 1),
        "fraction_present_pos_only": counts_full["PRESENT_POS_ONLY"] / max(n_target, 1),
        "present_allele_stratification": {
            "in_lcr": present_in_lcr, "out_lcr": present_out_lcr,
            "frac_in_lcr": frac_present_in_lcr, "frac_target_in_lcr": frac_target_in_lcr,
            "lcr_enrichment_vs_target": frac_present_in_lcr / max(frac_target_in_lcr, 1e-12),
            "cpg_ti": present_cpg, "noncpg_ti": present_noncpg_ti, "transversion": present_tv,
        },
        "drop_sample": args.drop_sample,
        "drop_sample_found": has_drop,
        "ec57_sensitivity": {
            "counts_drop": counts_drop,
            "present_allele_drop": counts_drop["PRESENT_ALLELE"],
            "lost_when_drop": lost_when_drop,
            "frac_present_lost_when_drop": lost_when_drop / max(n_present, 1),
        } if has_drop else None,
        "interpretation": ("Canal de presencia externa. PRESENT_ALLELE descarta "
                           "privacidad del alelo; ABSENT_FROM_PANEL no confirma founder. Sin "
                           "Beta-Binomial ni un tercer estado de callability."),
    }
    (args.outdir / f"{args.out_prefix}.external_presence.summary.json").write_text(
        json.dumps(summary, indent=2))

    script_path = Path(__file__).resolve()
    try:
        script_sha = file_sha256(script_path)
    except Exception:  # El destino del enlace puede quedar fuera del montaje.
        script_sha = "unavailable"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "panel_id": args.panel_id,
        "panel_version": args.panel_version,
        "panel_n_samples": panel_meta["n_samples"],
        "drop_sample": args.drop_sample,
        "panel_pass_only": args.panel_pass_only,
        "script_sha256": script_sha,
        "reused_lib": "bin/genomic_context.py (load_bed_chrom, in_bed, mut_context, resolve_contig)",
        "columns": cols,
        "ac_an_source": "genotypes (build-agnostic, not INFO)",
        "lcr_segdup_bed": str(args.lcr_segdup_bed),
    }
    (args.outdir / f"{args.out_prefix}.manifest.json").write_text(json.dumps(manifest, indent=2))

    repo = script_path.parents[1]
    (args.outdir / f"{args.out_prefix}.PROVENANCE.md").write_text("\n".join([
        "# PROVENANCE — Módulo 21 canal de presencia externa / no-privacidad",
        f"- Generado: {datetime.now(timezone.utc).isoformat()}",
        f"- Script: bin/external_presence_audit.py @ commit {git_commit(repo)} (sha256 {manifest['script_sha256'][:16]}…)",
        f"- dnabr_rare_vcf: {args.dnabr_rare_vcf}",
        f"- panel_vcf: {args.panel_vcf} (id={args.panel_id}, v={args.panel_version}, pass_only={args.panel_pass_only})",
        f"- fasta: {args.fasta}", f"- lcr_segdup_bed: {args.lcr_segdup_bed}",
        f"- drop_sample: {args.drop_sample}",
        f"- comando: {' '.join(sys.argv)}",
    ]) + "\n")

    LOG.info("PRESENT_ALLELE=%d / %d target (%.2f%%); REF_MISMATCH=%d; lost_when_drop=%d",
             n_present, n_target, 100 * n_present / max(n_target, 1),
             counts_full["REF_MISMATCH"], lost_when_drop)
    LOG.info("OK -> %s", args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
