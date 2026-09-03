nextflow.enable.dsl=2

include { M37_TRACE_RECOVERY_GATE } from '../modules/37_TRACE_COMPACT_RECOVERY'

workflow {
    ['m37_run_id', 'm37_root', 'm37_recovery_id', 'm37_recovery_contract',
     'm37_recovery_source_archive', 'm37_recovery_hmm_work_dir',
     'm37_recovery_tcn_work_dir', 'm37_recovery_positive_control_work_dir'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (params.m37_run_id != 'm37-r0-compact-sweep-20260903a' ||
        params.m37_recovery_id != 'm37-r0-compact-recovery-20260903a' ||
        params.m37_root != 'R0') {
        error 'the recovery workflow is sealed to the orphaned M37 R0 run'
    }
    ['m37_valid_selected', 'm37_valid_target', 'm37_valid_reference_folds',
     'm37_valid_f0', 'm37_valid_marker_cm', 'm37_valid_f0_receipt',
     'm37_valid_truth'].each { key ->
        if (params.containsKey(key) && params[key]) {
            error "--${key} is forbidden in the recovery workflow"
        }
    }

    def repoDir = projectDir.resolve('..')
    def hmmDir = params.m37_recovery_hmm_work_dir as String
    def tcnDir = params.m37_recovery_tcn_work_dir as String
    def controlDir = params.m37_recovery_positive_control_work_dir as String
    def metrics = files("${hmmDir}/*.hmm.*.metrics.json", checkIfExists: true) +
                  files("${tcnDir}/*.tcn.*.metrics.json", checkIfExists: true)
    def receipts = files("${hmmDir}/*.hmm.*.metrics.receipt.json", checkIfExists: true) +
                   files("${tcnDir}/*.tcn.*.metrics.receipt.json", checkIfExists: true)
    def audits = [file("${hmmDir}/hmm.compact_sweep.audit.json", checkIfExists: true),
                  file("${tcnDir}/tcn.compact_sweep.audit.json", checkIfExists: true)]
    def auditReceipts = [file("${hmmDir}/hmm.compact_sweep.audit.receipt.json", checkIfExists: true),
                         file("${tcnDir}/tcn.compact_sweep.audit.receipt.json", checkIfExists: true)]
    def equivalences = [file("${hmmDir}/hmm.equivalence.json", checkIfExists: true),
                        file("${tcnDir}/tcn.equivalence.json", checkIfExists: true)]
    def equivalenceReceipts = [file("${hmmDir}/hmm.equivalence.receipt.json", checkIfExists: true),
                               file("${tcnDir}/tcn.equivalence.receipt.json", checkIfExists: true)]

    def gateInput = channel.of(tuple(
        file(params.m37_recovery_contract, checkIfExists: true),
        file(params.m37_recovery_source_archive, checkIfExists: true),
        file("${controlDir}/m37.compact_positive_control.json", checkIfExists: true),
        file("${controlDir}/m37.compact_positive_control.receipt.json", checkIfExists: true),
        metrics, receipts, audits, auditReceipts, equivalences, equivalenceReceipts,
    ))
    M37_TRACE_RECOVERY_GATE(
        gateInput,
        file("${repoDir}/bin/m37_trace_recovery_gate.py", checkIfExists: true),
    )
}
