#!/usr/bin/env python3
"""
Mapeo de variantes raras (MAF<1%) sobre el painting de ancestría local (LAI Gnomix).

Pregunta del análisis: ¿dónde caen las variantes raras respecto a los tractos de
ancestría local (Native_American / European / African) del paper de Nunes 2025?

Enfoque y salvaguardas (ver registro metodológico: crosswalk-ids-rare-lai-metadata,
2026-06-01-painting-gnomix-local-desbloqueo):

1. Crosswalk comprobado (N=2619): rare_id == metadata.ID → LAI_id = {Cohort}_{ID}.
   metadata_cleaned.txt contiene dos filas de BB-COVL-397 y requiere deduplicación.

2. Fase: el VCF raro contiene fase parcial o inconsistente (mezcla 0/1, 1/0, 0|1, 1|1).
   Esa fase no se usa para asignar haplotipo porque la correspondencia de orden (.0/.1)
   con el scaffold SHAPEIT4 de LAI (.msp) no está certificada.
   El conteo n_alt=count("1") es robusto al separador (|/) y al orden, así que es
   exacto igual. El .msp da la ancestría local de ambos haplotipos. Atribución:
     - homocigoto-alt (1/1): cada haplotipo porta una copia → exacto sin fase.
     - heterocigoto (0/1) en tracto homocigoto-ancestral (hap0==hap1): exacto sin fase.
     - heterocigoto (0/1) en tracto het-ancestral (hap0!=hap1): ambiguo porque no sabemos
       qué haplotipo porta el alelo. Se reporta aparte; en el resultado primario solo
       cuentan las atribuciones exactas. Sensibilidad: reparto fraccional 0.5/0.5.

3. Baseline posicional: "X% de raras en tractos NAM" no se interpreta por sí solo. Se compara
   contra la fracción de ancestría local de la cohorte en esas mismas posiciones
   (expected = suma de baseline_seg sobre los mismos eventos). Enrichment = obs/exp.

   Limitación: el baseline posicional no elimina la tautología a
   nivel-individuo (burden de raras ∝ ancestría NAM, Nunes Fig 1A). Eso requiere la
   residualización y un null condicional. Este script es descriptivo.

Códigos de ancestría en el .msp: African=0, European=1, Native-American=2.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANC_NAMES = {0: "African", 1: "European", 2: "Native_American"}


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--msp", required=True, help="Painting Gnomix .msp (chr-específico)")
    p.add_argument("--rare_vcf", required=True, help="VCF de variantes raras (chr-específico, bgzip+tbi)")
    p.add_argument("--metadata", required=True, help="metadata_cleaned.txt (cols ID, Cohort)")
    p.add_argument("--vcf_chrom", required=True, help="Nombre del cromosoma en el VCF (ej. chr22)")
    p.add_argument("--out_prefix", required=True, help="Prefijo de salida")
    p.add_argument("--bcftools", default="bcftools", help="Ruta al binario bcftools")
    p.add_argument("--emit_windows_bp", type=int, default=0,
                   help="Si >0, emite <prefix>.windows.tsv con copias exactas por "
                        "ancestria y territorio pintado por ancestria, por ventana de "
                        "este tamano (bp). Para densidad posicional estratificada. "
                        "Default 0 = no emite (modulo inalterado).")
    return p.parse_args()


def load_crosswalk(metadata_path):
    """rare_id (==metadata.ID) -> LAI_id (=={Cohort}_{ID}). Dedup obligatorio."""
    df = pd.read_csv(metadata_path, sep="\t", dtype=str)
    n_raw = len(df)
    df = df.drop_duplicates(subset="ID")
    n_dedup = len(df)
    if n_raw != n_dedup:
        print(f"[crosswalk] dedup metadata: {n_raw} filas -> {n_dedup} IDs únicos "
              f"(removidas {n_raw - n_dedup})", file=sys.stderr)
    rare_to_lai = {row.ID: f"{row.Cohort}_{row.ID}" for row in df.itertuples()}
    return rare_to_lai


def parse_msp(msp_path, rare_to_lai):
    """
    Parsea el .msp. Retorna:
      spos, epos: arrays de límites de segmento (1-based, ascendentes)
      codes: matriz int8 [n_seg, n_hap_keep] con el código de ancestría
      lai_to_cols: dict LAI_id -> (idx_hap0, idx_hap1) en las columnas de `codes`
    Solo se retienen las columnas-haplotipo de individuos cuyo LAI_id está en el
    conjunto de valores del crosswalk (los 2619 reconciliables).
    """
    with open(msp_path) as fh:
        line1 = fh.readline()  # "#Subpopulation order/codes: African=0 European=1 Native-American=2"
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
    # Validar códigos declarados == los asumidos
    assert "African=0" in line1 and "European=1" in line1 and "Native-American=2" in line1, \
        f"Códigos de ancestría inesperados en .msp: {line1!r}"

    # Localizar inicio de columnas-haplotipo: primera columna que termina en '.0'
    hap_start = next(i for i, c in enumerate(header) if c.endswith(".0"))
    fixed_cols = header[:hap_start]
    assert fixed_cols[:3] == ["chm", "spos", "epos"], f"Columnas fijas inesperadas: {fixed_cols}"

    hap_headers = header[hap_start:]
    lai_ids_in_crosswalk = set(rare_to_lai.values())

    # Se mapea LAI_id a los índices de columna .0 y .1 dentro del bloque de haplotipos,
    # reteniendo solo individuos del crosswalk.
    col0, col1 = {}, {}
    for j, h in enumerate(hap_headers):
        lai_id, hap = h.rsplit(".", 1)
        if lai_id not in lai_ids_in_crosswalk:
            continue
        (col0 if hap == "0" else col1)[lai_id] = j

    common = sorted(set(col0) & set(col1))
    keep_cols = []          # índices (en hap_headers) a conservar, en orden
    lai_to_cols = {}        # LAI_id -> (pos en keep_cols para .0, pos para .1)
    for lid in common:
        lai_to_cols[lid] = (len(keep_cols), len(keep_cols) + 1)
        keep_cols.extend([col0[lid], col1[lid]])

    # Leer la matriz de datos (solo columnas fijas spos/epos + las hap retenidas)
    data = pd.read_csv(msp_path, sep="\t", skiprows=1)  # header = línea 2
    spos = data.iloc[:, 1].to_numpy(dtype=np.int64)
    epos = data.iloc[:, 2].to_numpy(dtype=np.int64)
    hap_block = data.iloc[:, hap_start:].to_numpy(dtype=np.int8)
    codes = hap_block[:, keep_cols]

    # Segmentos deben venir ordenados y no solaparse
    assert np.all(np.diff(spos) > 0), ".msp: segmentos no ordenados por spos"
    print(f"[msp] {len(spos)} segmentos, {len(common)} individuos retenidos "
          f"(de {len(hap_headers)//2} en el .msp)", file=sys.stderr)
    return spos, epos, codes, lai_to_cols, len(common)


def segment_baseline(codes):
    """baseline[seg, anc] = fracción de haplotipos de la cohorte con esa ancestría."""
    n_hap = codes.shape[1]
    base = np.zeros((codes.shape[0], 3), dtype=np.float64)
    for anc in (0, 1, 2):
        base[:, anc] = (codes == anc).sum(axis=1) / n_hap
    return base


def main():
    """Cruza variantes raras con segmentos de ancestría local y agrega resultados."""
    args = parse_args()
    rare_to_lai = load_crosswalk(args.metadata)
    spos, epos, codes, lai_to_cols, n_indiv = parse_msp(args.msp, rare_to_lai)
    baseline = segment_baseline(codes)
    painted_min, painted_max = int(spos[0]), int(epos[-1])

    # Acumuladores
    obs_exact = np.zeros(3)        # Copias del alelo raro con ancestría exacta.
    obs_frac = np.zeros(3)         # incluye ambiguos repartidos 0.5/0.5 (sensibilidad)
    exp_copies = np.zeros(3)       # Baseline de la métrica fraccional sobre todos los eventos.
    exp_exact = np.zeros(3)        # Baseline de los eventos con atribución exacta.
    n_events = n_copies = 0
    n_skip_not2619 = n_unassignable = n_ambiguous_copies = 0

    # Acumuladores por ventana (opcional): copias exactas y territorio pintado.
    win_bp = int(args.emit_windows_bp)
    if win_bp > 0:
        n_win = (painted_max - painted_min) // win_bp + 1
        obs_exact_win = np.zeros((n_win, 3))
        territory_win = np.zeros((n_win, 3))  # bp pintado por ancestria (cohorte) por ventana
        # Territorio: reparte bp de cada segmento (x composicion de cohorte) entre ventanas.
        for i in range(len(spos)):
            s, e = int(spos[i]), int(epos[i])
            w0 = (s - painted_min) // win_bp
            w1 = (e - painted_min) // win_bp
            for w in range(w0, w1 + 1):
                win_s = painted_min + w * win_bp
                win_e = win_s + win_bp - 1
                ov = min(e, win_e) - max(s, win_s) + 1
                if ov > 0:
                    territory_win[w] += baseline[i] * ov

    cmd = [args.bcftools, "query", "-i", 'GT="alt"',
           "-f", "[%POS\t%SAMPLE\t%GT\n]",
           "-r", f"{args.vcf_chrom}:{painted_min}-{painted_max}", args.rare_vcf]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)

    for line in proc.stdout:
        pos_s, rare_id, gt = line.rstrip("\n").split("\t")
        lai_id = rare_to_lai.get(rare_id)
        if lai_id is None or lai_id not in lai_to_cols:
            n_skip_not2619 += 1
            continue
        pos = int(pos_s)
        # Segmento que contiene pos
        i = int(np.searchsorted(spos, pos, side="right")) - 1
        if i < 0 or pos > epos[i]:
            n_unassignable += 1
            continue
        c0idx, c1idx = lai_to_cols[lai_id]
        a0, a1 = int(codes[i, c0idx]), int(codes[i, c1idx])
        n_alt = gt.count("1")  # VCF bialélico → alt index 1
        if n_alt == 0:
            continue
        n_events += 1
        n_copies += n_alt
        # Esperado posicional para la métrica fraccional: cada copia aporta la
        # composición de la cohorte en el seg (todas las copias, incl. ambiguas).
        exp_copies += n_alt * baseline[i]

        ex = np.zeros(3)                    # Copias exactas atribuidas en este evento.
        if n_alt == 2:                      # homocigoto-alt: una copia por haplotipo → exacto
            ex[a0] += 1; ex[a1] += 1
            obs_frac[a0] += 1; obs_frac[a1] += 1
            exp_exact += 2 * baseline[i]    # Dos copias exactas aportan dos veces el baseline.
        else:                               # heterocigoto: 1 copia
            if a0 == a1:                    # tracto homocigoto-ancestral → exacto
                ex[a0] += 1
                obs_frac[a0] += 1
                exp_exact += baseline[i]    # Una copia exacta aporta una vez el valor esperado.
            else:                           # het-ancestral → ambiguo (sin fase)
                n_ambiguous_copies += 1
                obs_frac[a0] += 0.5; obs_frac[a1] += 0.5
                # exp_exact no acumula porque aquí no existe una copia con atribución exacta.
        obs_exact += ex
        if win_bp > 0 and ex.any():
            obs_exact_win[(pos - painted_min) // win_bp] += ex

    ret = proc.wait()
    if ret != 0:
        sys.exit(f"bcftools query falló (exit {ret})")

    # Composición global (chr) de la cohorte = media de baseline ponderada por bp de segmento
    seg_bp = (epos - spos + 1).astype(np.float64)
    chr_comp = (baseline * seg_bp[:, None]).sum(axis=0) / seg_bp.sum()

    def ratios(obs, exp):
        """Calcula proporciones observadas y enriquecimientos frente al baseline."""
        tot = obs.sum()
        frac = obs / tot if tot > 0 else np.zeros(3)
        exp_frac = exp / exp.sum() if exp.sum() > 0 else np.zeros(3)
        enr = np.divide(frac, exp_frac, out=np.zeros(3), where=exp_frac > 0)
        return frac, exp_frac, enr

    # exact usa el baseline de eventos exactos; fractional usa el conjunto completo.
    frac_e, exp_frac, enr_e = ratios(obs_exact, exp_exact)
    frac_f, _, enr_f = ratios(obs_frac, exp_copies)

    summary = {
        "n_individuals": n_indiv,
        "painted_range": [painted_min, painted_max],
        "n_carrier_events": n_events,
        "n_rare_allele_copies": n_copies,
        "n_skipped_not_in_2619": n_skip_not2619,
        "n_unassignable_outside_painting": n_unassignable,
        "n_ambiguous_copies_phase_limited": n_ambiguous_copies,
        "pct_ambiguous_of_copies": round(100 * n_ambiguous_copies / n_copies, 2) if n_copies else None,
        "chr_cohort_ancestry_composition": {ANC_NAMES[a]: round(chr_comp[a], 4) for a in (0, 1, 2)},
        "observed_exact": {ANC_NAMES[a]: float(obs_exact[a]) for a in (0, 1, 2)},
        "observed_fractional_raw": {ANC_NAMES[a]: float(obs_frac[a]) for a in (0, 1, 2)},
        "expected_copies_raw": {ANC_NAMES[a]: float(exp_copies[a]) for a in (0, 1, 2)},
        "expected_copies_exact_raw": {ANC_NAMES[a]: float(exp_exact[a]) for a in (0, 1, 2)},
        "observed_fraction_exact": {ANC_NAMES[a]: round(frac_e[a], 4) for a in (0, 1, 2)},
        "expected_fraction_positional_baseline": {ANC_NAMES[a]: round(exp_frac[a], 4) for a in (0, 1, 2)},
        "enrichment_exact_obs_over_exp": {ANC_NAMES[a]: round(enr_e[a], 4) for a in (0, 1, 2)},
        "enrichment_with_ambiguous_fractional": {ANC_NAMES[a]: round(enr_f[a], 4) for a in (0, 1, 2)},
        "CAVEAT": "Descriptivo. El baseline posicional no elimina la tautología "
                  "burden-raras ∝ NAM a nivel-individuo (requiere residualización, Paso 2).",
    }

    out_json = Path(f"{args.out_prefix}.summary.json")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    rows = []
    for a in (0, 1, 2):
        rows.append({
            "ancestry": ANC_NAMES[a],
            "observed_copies_exact": obs_exact[a],
            "observed_fraction_exact": frac_e[a],
            "expected_fraction_baseline": exp_frac[a],
            "enrichment_exact": enr_e[a],
            "enrichment_fractional": enr_f[a],
            "chr_cohort_composition": chr_comp[a],
        })
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}.by_ancestry.tsv", sep="\t", index=False)

    if win_bp > 0:
        win_rows = []
        for w in range(n_win):
            win_s = painted_min + w * win_bp
            win_e = min(win_s + win_bp - 1, painted_max)
            win_rows.append({
                "window_start": win_s,
                "window_end": win_e,
                "copies_exact_African": obs_exact_win[w, 0],
                "copies_exact_European": obs_exact_win[w, 1],
                "copies_exact_Native_American": obs_exact_win[w, 2],
                "territory_bp_African": territory_win[w, 0],
                "territory_bp_European": territory_win[w, 1],
                "territory_bp_Native_American": territory_win[w, 2],
            })
        pd.DataFrame(win_rows).to_csv(f"{args.out_prefix}.windows.tsv", sep="\t", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
