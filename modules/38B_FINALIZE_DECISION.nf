nextflow.enable.dsl=2

process M38B_FINALIZE_DECISION {
    tag 'M38B_finalize_immutable_scores'
    publishDir { "${params.m38b_finalizer_results_dir}/${params.m38b_finalizer_source_run_id}/decision" },
        mode: 'copy', overwrite: false, pattern: 'm38b.final_decision*.json'
    publishDir { "${params.m38b_finalizer_results_dir}/${params.m38b_finalizer_run_id}/audit" },
        mode: 'copy', overwrite: false, pattern: 'm38b.finalizer.provenance.json'
    cpus 1
    memory '2 GB'
    time '15m'
    maxForks 1

    input:
    tuple path(analytic), path(analyticReceipt), path(tcn), path(tcnReceipt),
          path(positive), path(positiveReceipt)
    path inputManifest
    val inputManifestSha256
    path decisionScript, stageAs: 'staged/bin/m38b_decide.py'
    path finalizerScript, stageAs: 'staged/bin/m38b_finalize_decision.py'
    path provenanceSources, stageAs: 'staged/provenance/*'
    val codeCommit
    val runtimeImage

    output:
    tuple path('m38b.final_decision.json'), path('m38b.final_decision.receipt.json'), emit: decision
    path 'm38b.finalizer.provenance.json', emit: provenance

    script:
    def provenanceFlags = provenanceSources.collect { "--provenance-source '${it}'" }.join(' ')
    """
    set -euo pipefail
    python3 staged/bin/m38b_finalize_decision.py \
      --manifest '${inputManifest}' --manifest-sha256 '${inputManifestSha256}' \
      --artifact 'analytic_metrics=${analytic}' \
      --artifact 'analytic_receipt=${analyticReceipt}' \
      --artifact 'tcn_metrics=${tcn}' --artifact 'tcn_receipt=${tcnReceipt}' \
      --artifact 'positive_metrics=${positive}' \
      --artifact 'positive_receipt=${positiveReceipt}' \
      --decision-script staged/bin/m38b_decide.py \
      --code-commit '${codeCommit}' --runtime-image '${runtimeImage}' \
      ${provenanceFlags} --output m38b.final_decision.json \
      --provenance-output m38b.finalizer.provenance.json
    """

    stub:
    """
    touch m38b.final_decision.json m38b.final_decision.receipt.json \
      m38b.finalizer.provenance.json
    """
}
