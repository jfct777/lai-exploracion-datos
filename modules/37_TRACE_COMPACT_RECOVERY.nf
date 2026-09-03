nextflow.enable.dsl=2

process M37_TRACE_RECOVERY_GATE {
    tag { "${params.m37_run_id}_orphan-gate" }
    publishDir "${params.m37_results_dir}/${params.m37_run_id}/audit/recovery",
        mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('m37.orphan_recovery.audit') ? name : null }
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    tuple path(recovery_contract), path(source_archive),
          path(positive_control), path(positive_control_receipt),
          path(metric_files), path(metric_receipts),
          path(family_audits), path(family_audit_receipts),
          path(equivalence_files), path(equivalence_receipts)
    path gate_source

    output:
    tuple path(metric_files), path(metric_receipts),
          path(family_audits), path(family_audit_receipts),
          path(source_archive), path('m37.orphan_recovery.audit.json'),
          path('m37.orphan_recovery.audit.receipt.json'), emit: verified

    script:
    def metricFlags = metric_files.collect { path -> "--metric '${path}'" }.join(' ')
    def receiptFlags = metric_receipts.collect { path -> "--metric-receipt '${path}'" }.join(' ')
    def auditFlags = family_audits.collect { path -> "--family-audit '${path}'" }.join(' ')
    def auditReceiptFlags = family_audit_receipts.collect { path -> "--family-audit-receipt '${path}'" }.join(' ')
    def equivalenceFlags = equivalence_files.collect { path -> "--equivalence '${path}'" }.join(' ')
    def equivalenceReceiptFlags = equivalence_receipts.collect { path -> "--equivalence-receipt '${path}'" }.join(' ')
    """
    set -euo pipefail
    python3 ${gate_source} \
      --contract ${recovery_contract} --source-archive ${source_archive} \
      --positive-control ${positive_control} \
      --positive-control-receipt ${positive_control_receipt} \
      ${metricFlags} ${receiptFlags} ${auditFlags} ${auditReceiptFlags} \
      ${equivalenceFlags} ${equivalenceReceiptFlags} \
      --output m37.orphan_recovery.audit.json
    """
}


process M37_TRACE_RECOVERY_COLLECT_METRICS {
    tag { "${root}_recovered-paired-metrics" }
    publishDir "${params.m37_results_dir}/${params.m37_run_id}/promotion",
        mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith("m37.${params.m37_root}.paired_metrics") ? name : null }
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    tuple val(root), path(metric_files), path(metric_receipts),
          path(family_audits), path(family_audit_receipts),
          path(source_archive), path(recovery_audit), path(recovery_audit_receipt)

    output:
    tuple val(root), path("m37.${root}.paired_metrics.json"),
          path("m37.${root}.paired_metrics.receipt.json"),
          path(family_audits), path(family_audit_receipts),
          path(source_archive), path(recovery_audit), path(recovery_audit_receipt), emit: bundle

    script:
    def metricFlags = metric_files.collect { path -> "--metric '${path}'" }.join(' ')
    def receiptFlags = metric_receipts.collect { path -> "--receipt '${path}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p frozen
    tar --extract --file ${source_archive} --directory frozen
    PYTHONPATH=frozen/bin python3 frozen/bin/m37_trace_collect_metrics.py \
      --root '${root}' --expected-evaluation-split FIT_TUNE \
      ${metricFlags} ${receiptFlags} --output 'm37.${root}.paired_metrics.json'
    """
}


process M37_TRACE_RECOVERY_COMPACT_DECISION {
    tag { "${root}_recovered-compact-triage" }
    publishDir "${params.m37_results_dir}/${params.m37_run_id}/promotion",
        mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('m37.compact_triage') ? name : null }
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    tuple val(root), path(metrics_json), path(metrics_receipt),
          path(family_audits), path(family_audit_receipts),
          path(source_archive), path(recovery_audit), path(recovery_audit_receipt)

    output:
    tuple val(root), path('m37.compact_triage.json'),
          path('m37.compact_triage.receipt.json'), emit: decision

    script:
    def auditFlags = family_audits.collect { path -> "--family-audit '${path}'" }.join(' ')
    def auditReceiptFlags = family_audit_receipts.collect { path -> "--family-audit-receipt '${path}'" }.join(' ')
    def decisionSources = [
        'bin/m37_trace_core.py',
        'bin/m37_trace_collect_metrics.py',
        'bin/m37_trace_successive_halving.py',
        'bin/m37_trace_compact_decision.py',
        'modules/37_TRACE_COMPACT_SWEEP.nf',
        'workflows/m37_trace_compact_sweep.nf',
        'conf/m37_trace_compact_sweep.config',
        'conf/m37_trace_gcp.config',
    ]
    def authFlags = decisionSources.collect { name -> "--auth-file 'frozen/${name}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p frozen
    tar --extract --file ${source_archive} --directory frozen
    PYTHONPATH=frozen/bin python3 frozen/bin/m37_trace_compact_decision.py \
      --metrics-json ${metrics_json} --metrics-receipt ${metrics_receipt} \
      --root '${root}' --run-id '${params.m37_run_id}' \
      ${auditFlags} ${auditReceiptFlags} ${authFlags} \
      --output m37.compact_triage.json
    """
}
