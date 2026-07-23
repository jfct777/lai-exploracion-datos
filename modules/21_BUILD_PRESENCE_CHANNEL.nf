nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 21 — Build presence channel (CANAL DE PRESENCIA EXTERNA / NO-PRIVACIDAD)
// ---------------------------------------------------------------------------
// Para cada SNV raro bialélico de DNABR (upstream lai_rare) y cada panel NAM externo
// (NAMBR-128 VQSR, 71-native .raw, …) clasifica si el ALELO aparece afuera:
//   PRESENT_ALLELE (refuta privacidad) / PRESENT_POS_ONLY / REF_MISMATCH / ABSENT.
// Canal binario present/unknown: la ausencia no confirma un efecto fundador,
// por eso NO hay Beta-Binomial ni 3.er estado de callability.
//
// El panel es un INPUT DE REFERENCIA pre-staged (``tools/stage_presence_panels.sh`` lo
// baja del bucket y lo tabixea a ``presence_panel_dir/<panel_id>/chr<C>.vcf.gz``). El
// pipeline NO baja de gs:// dentro de un proceso (igual que no baja el FASTA hg38): el
// staging es provisioning, separado del cómputo (principio: sin red dentro de procesos).
//
// Topología (mismo molde que M17):
//   BUILD_PRESENCE_LCR_MASK   1 vez  -> bed genome-wide segdup+simpleRepeat+blacklist
//   ANALYZE_PRESENCE_CHANNEL  panel × cromosoma -> summary.json + audit.tsv.gz
//   AGGREGATE_PRESENCE_CHANNEL  por panel -> genomewide.json + per_chr.tsv (suma cruda)
//
// El bin valida REF_MISMATCH, cuenta PASS/non-PASS, permite ``--panel-pass-only``
// y registra el sha256 del script.

process BUILD_PRESENCE_LCR_MASK {
    tag "lcr_mask"

    publishDir "${params.presence_channel_results_dir}/_reference", mode: 'copy'

    cpus   params.resources.presence_lcr_mask.cpus
    memory params.resources.presence_lcr_mask.memory
    time   params.time

    input:
    tuple path(segdup), path(simplerep), path(blacklist)

    output:
    path "lcr_segdup_blacklist.genomewide.bed.gz", emit: bed

    script:
    """
    set -euo pipefail
    # genomicSuperDups/simpleRepeat: UCSC txt con col1=bin -> chrom/start/end = \$2,\$3,\$4.
    # hg38-blacklist: BED nativo -> chrom/start/end = \$1,\$2,\$3.  El bin python filtra por
    # cromosoma en lectura (load_bed_chrom), así que UN bed genome-wide alcanza (DRY).
    {
      zcat ${segdup}    | awk -F'\\t' 'NF>=4 && \$2 ~ /^chr/ {print \$2"\\t"\$3"\\t"\$4}'
      zcat ${simplerep} | awk -F'\\t' 'NF>=4 && \$2 ~ /^chr/ {print \$2"\\t"\$3"\\t"\$4}'
      zcat ${blacklist} | awk -F'\\t' 'NF>=3 && \$1 ~ /^chr/ {print \$1"\\t"\$2"\\t"\$3}'
    } | sort -k1,1 -k2,2n | gzip -c > lcr_segdup_blacklist.genomewide.bed.gz

    if [ ! -s lcr_segdup_blacklist.genomewide.bed.gz ]; then
      echo "ERROR: bed LCR/segdup genome-wide vacío" >&2; exit 1
    fi
    """
}

process ANALYZE_PRESENCE_CHANNEL {
    tag "${panel_id}:chr${chr}"

    publishDir "${params.presence_channel_results_dir}/${panel_id}/per_chr", mode: 'copy'

    cpus   params.resources.presence_channel_scan.cpus
    memory params.resources.presence_channel_scan.memory
    time   params.time

    input:
    tuple val(panel_id), val(panel_version), val(drop_sample), val(pass_only), val(chr),
          path(rare_vcf), path(rare_tbi), path(panel_vcf), path(panel_tbi),
          path(lcr_bed), path(fasta), path(fasta_fai), path(audit_py), path(lib_py)

    output:
    tuple val(panel_id), val(chr), path("${panel_id}.chr${chr}.external_presence.summary.json"), emit: summary
    path "${panel_id}.chr${chr}.external_presence.audit.tsv.gz", emit: audit
    path "${panel_id}.chr${chr}.manifest.json"
    path "${panel_id}.chr${chr}.PROVENANCE.md"

    script:
    def drop_arg = drop_sample?.trim() ? "--drop-sample ${drop_sample}" : ""
    def pass_arg = pass_only ? "--panel-pass-only" : ""
    """
    set -euo pipefail
    python3 ${audit_py} \
      --dnabr-rare-vcf ${rare_vcf} \
      --panel-vcf ${panel_vcf} \
      --panel-id ${panel_id} \
      --panel-version '${panel_version}' \
      --fasta ${fasta} \
      --lcr-segdup-bed ${lcr_bed} \
      --chrom chr${chr} \
      --out-prefix ${panel_id}.chr${chr} \
      --outdir . \
      ${drop_arg} ${pass_arg}
    """
}

process AGGREGATE_PRESENCE_CHANNEL {
    tag "${panel_id}"

    publishDir "${params.presence_channel_results_dir}/${panel_id}", mode: 'copy'

    cpus   params.resources.presence_channel_aggregate.cpus
    memory params.resources.presence_channel_aggregate.memory
    time   params.time

    input:
    tuple val(panel_id), path(summary_files), path(aggregate_py)

    output:
    path "${panel_id}.external_presence.genomewide.json", emit: genomewide
    path "${panel_id}.external_presence.per_chr.tsv",     emit: per_chr

    script:
    """
    set -euo pipefail
    python3 ${aggregate_py} \
      --glob '*.external_presence.summary.json' \
      --out-prefix ${panel_id}
    """
}
