nextflow.enable.dsl=2


def m37_compact_capacity_families(positive_control_path) {
    // Use the NIO provider directly: GCS-backed Nextflow Paths cannot be
    // converted safely with ``toFile()``.
    def capacity = java.nio.file.Files.newBufferedReader(positive_control_path).withCloseable {
        reader -> new groovy.json.JsonSlurper().parse(reader)
    }
    def eligible = capacity?.controls?.tcn?.eligible_candidate_ids
    if (!(eligible instanceof List)) {
        throw new IllegalArgumentException('M37 TCN capacity roster is missing')
    }
    return eligible.isEmpty() ? ['hmm'] : ['hmm', 'tcn']
}


def m37_compact_decision_parts(combined) {
    if (!(combined instanceof List) || combined.size() < 4) {
        def observedSize = combined instanceof List ? combined.size() : 'not-a-list'
        throw new IllegalArgumentException(
            "M37 compact decision channel shape differs: ${observedSize} fields"
        )
    }
    // ``combine`` flattens the collected audit rows into the left tuple.  A
    // one-argument closure is therefore required for both the HMM-only
    // (4 fields) and HMM+TCN (5 fields) branches.
    def auditRows = combined.subList(3, combined.size())
    if (!(auditRows instanceof List) || auditRows.isEmpty() ||
        !auditRows.every { row -> row instanceof List && row.size() == 2 }) {
        throw new IllegalArgumentException('M37 compact family-audit channel shape differs')
    }
    return [combined[0], combined[1], combined[2],
            auditRows.collect { row -> row[0] },
            auditRows.collect { row -> row[1] }]
}


process M37_TRACE_COMPACT_CAPACITY_SCREEN {
    tag { "${params.m37_run_id}_capacity-screen" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/audit" }, mode: 'copy', overwrite: false
    cpus 2
    memory '4 GB'
    // The adaptive ladder may execute four deterministic restarts per
    // candidate.  This is a ceiling, not reserved billable runtime.
    time '4h'

    input:
    tuple path(candidate_manifest), path(parent_contract), path(contract_amendment)
    path source_files

    output:
    tuple path('m37.capacity_screen.json'),
          path('m37.capacity_screen.receipt.json'),
          path('m37.capacity_selection.json'),
          path('m37.capacity_selection.receipt.json'), emit: evidence

    script:
    def authFlags = source_files.collect { path -> "--auth-file 'staged/bin/${path.name}'" }.join(' ')
    """
    set -euo pipefail
    export USER=m37-runner
    export LOGNAME=m37-runner
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_compact_positive_control.py \
      --phase screen \
      --run-id '${params.m37_run_id}' --candidate-manifest ${candidate_manifest} \
      --parent-contract ${parent_contract} --contract-amendment ${contract_amendment} \
      --container-digest '${params.m37_container_digest}' \
      ${authFlags} \
      --output m37.capacity_screen.json \
      --selection-output m37.capacity_selection.json
    """
}


process M37_TRACE_COMPACT_CAPACITY_REPLICATION {
    tag { "${params.m37_run_id}_capacity-replication" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/audit" }, mode: 'copy', overwrite: false
    cpus 2
    memory '4 GB'
    time '4h'

    input:
    tuple path(candidate_manifest), path(parent_contract), path(contract_amendment),
          path(capacity_screen), path(capacity_screen_receipt),
          path(capacity_selection), path(capacity_selection_receipt)
    path source_files

    output:
    tuple path('m37.compact_positive_control.json'),
          path('m37.compact_positive_control.receipt.json'), emit: evidence

    script:
    def authFlags = source_files.collect { path -> "--auth-file 'staged/bin/${path.name}'" }.join(' ')
    """
    set -euo pipefail
    export USER=m37-runner
    export LOGNAME=m37-runner
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_compact_positive_control.py \
      --phase replicate \
      --run-id '${params.m37_run_id}' --candidate-manifest ${candidate_manifest} \
      --parent-contract ${parent_contract} --contract-amendment ${contract_amendment} \
      --container-digest '${params.m37_container_digest}' \
      --screen ${capacity_screen} --screen-receipt ${capacity_screen_receipt} \
      --selection ${capacity_selection} --selection-receipt ${capacity_selection_receipt} \
      ${authFlags} \
      --output m37.compact_positive_control.json
    """
}

process M37_TRACE_COMPACT_SWEEP {
    tag { "${params.m37_run_id}_${family}" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/compact/${family}" }, mode: 'copy', overwrite: false
    cpus params.m37_compact_cpus
    memory params.m37_compact_memory
    time params.m37_compact_time
    maxForks params.m37_compact_max_forks

    input:
    tuple val(family), path(candidate_manifest), path(parent_contract), path(contract_amendment),
          path(canonical_metrics), path(canonical_metrics_receipt), path(fit_truth),
          path(fit_f0_receipt), path(feature_files), path(feature_receipts), path(run_overlay),
          path(positive_control), path(positive_control_receipt)
    path source_files

    output:
    tuple val(family), path('*.metrics.json'), path('*.metrics.receipt.json'),
          path("${family}.equivalence.json"), path("${family}.equivalence.receipt.json"),
          path("${family}.compact_sweep.audit.json"),
          path("${family}.compact_sweep.audit.receipt.json"), emit: bundle

    script:
    def featureFlags = feature_files.collect { path -> "--feature '${path}'" }.join(' ')
    def featureReceiptFlags = feature_receipts.collect { path -> "--feature-receipt '${path}'" }.join(' ')
    def authFlags = source_files.collect { path -> "--auth-file 'staged/bin/${path.name}'" }.join(' ')
    """
    set -euo pipefail
    export USER=m37-runner
    export LOGNAME=m37-runner
    export TORCHINDUCTOR_CACHE_DIR="\$PWD/.torch-cache"
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_compact_sweep.py \
      --run-id '${params.m37_run_id}' --root '${params.m37_root}' --family '${family}' \
      --candidate-manifest ${candidate_manifest} \
      --parent-contract ${parent_contract} --contract-amendment ${contract_amendment} \
      --positive-control ${positive_control} \
      --positive-control-receipt ${positive_control_receipt} \
      --canonical-metrics ${canonical_metrics} \
      --canonical-metrics-receipt ${canonical_metrics_receipt} \
      --truth ${fit_truth} --f0-receipt ${fit_f0_receipt} --run-overlay ${run_overlay} \
      --run-overlay-uri '${params.m37_run_overlay_uri}' \
      --container-digest '${params.m37_container_digest}' \
      ${featureFlags} ${featureReceiptFlags} ${authFlags} --output-dir .
    """
}


process M37_TRACE_COMPACT_DECISION {
    tag { "${root}_compact-triage" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/promotion" }, mode: 'copy', overwrite: false
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    tuple val(root), path(metrics_json), path(metrics_receipt), path(family_audits),
          path(family_audit_receipts)
    path source_files

    output:
    tuple val(root), path('m37.compact_triage.json'),
          path('m37.compact_triage.receipt.json'), emit: decision

    script:
    def auditFlags = family_audits.collect { path -> "--family-audit '${path}'" }.join(' ')
    def auditReceiptFlags = family_audit_receipts.collect { path -> "--family-audit-receipt '${path}'" }.join(' ')
    def authFlags = source_files.collect { path -> "--auth-file 'staged/bin/${path.name}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_compact_decision.py \
      --metrics-json ${metrics_json} --metrics-receipt ${metrics_receipt} \
      --root '${root}' --run-id '${params.m37_run_id}' \
      ${auditFlags} ${auditReceiptFlags} \
      ${authFlags} \
      --output m37.compact_triage.json
    """
}
