nextflow.enable.dsl=2

process M37_TRACE_RECOVERY_GATE {
    tag { "${params.m37_recovery_id}_nonconsumable-audit" }
    publishDir "${params.m37_results_dir}/${params.m37_run_id}/audit/recovery",
        mode: 'copy', overwrite: false,
        saveAs: { name -> [
            'm37.recovered.nonconsumable.audit.json',
            'm37.recovered.nonconsumable.audit.receipt.json',
            'm37.recovered.nonconsumable.summary.json',
            'm37.recovered.nonconsumable.summary.receipt.json',
        ].contains(name) ? name : null }
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
    tuple path('m37.recovered.nonconsumable.audit.json'),
          path('m37.recovered.nonconsumable.audit.receipt.json'),
          path('m37.recovered.nonconsumable.summary.json'),
          path('m37.recovered.nonconsumable.summary.receipt.json'), emit: evidence

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
      --output-dir .
    """
}
