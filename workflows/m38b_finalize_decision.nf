nextflow.enable.dsl=2

include { M38B_FINALIZE_DECISION } from '../modules/38B_FINALIZE_DECISION'

workflow {
    def required = [
        'm38b_finalizer_run_id', 'm38b_finalizer_source_run_id',
        'm38b_finalizer_results_dir', 'm38b_finalizer_input_manifest',
        'm38b_finalizer_input_manifest_sha256', 'm38b_finalizer_code_commit',
        'm38b_finalizer_python_image', 'm38b_finalizer_container_user',
    ]
    required.each { key -> if (!params[key]) error "--${key} is required" }
    if (params.m38b_finalizer_results_dir != 'gs://teams-usp/frank/lai-exploracion-datos/runs')
        error 'M38B finalizer outputs must remain in the personal project bucket'
    if (params.m38b_finalizer_source_run_id != 'm38b-r0-oof-models-20260903c')
        error 'M38B finalizer source must be immutable run c'
    if (params.m38b_finalizer_run_id == params.m38b_finalizer_source_run_id)
        error 'M38B finalizer audit run must be separate from source run c'
    if (!(params.m38b_finalizer_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'M38B finalizer run ID is unsafe'
    if (!(params.m38b_finalizer_input_manifest_sha256 ==~ /[0-9a-f]{64}/))
        error 'M38B finalizer manifest SHA-256 is malformed'
    if (!(params.m38b_finalizer_code_commit ==~ /[0-9a-f]{40}/))
        error 'M38B finalizer requires the full implementation commit'
    def expectedImage = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99'
    if (params.m38b_finalizer_python_image != expectedImage ||
        params.m38b_finalizer_container_user != '0:0')
        error 'M38B finalizer runtime image or user differs from the pinned runtime'

    def repoDir = projectDir.resolve('..')
    def inputManifest = file(params.m38b_finalizer_input_manifest, checkIfExists: true)
    def manifest = new groovy.json.JsonSlurper().parseText(inputManifest.text)
    if (manifest.stage != 'M38B_FINALIZER_INPUTS' || manifest.artifacts.size() != 6)
        error 'M38B finalizer input manifest structure differs'
    def artifacts = manifest.artifacts.collectEntries { row -> [(row.logical_id): row] }
    def expectedIds = ['analytic_metrics', 'analytic_receipt', 'tcn_metrics',
                       'tcn_receipt', 'positive_metrics', 'positive_receipt'] as Set
    if (artifacts.keySet() != expectedIds)
        error 'M38B finalizer input manifest IDs differ'

    scoreInputs = Channel.value(tuple(
        file(artifacts.analytic_metrics.uri, checkIfExists: true),
        file(artifacts.analytic_receipt.uri, checkIfExists: true),
        file(artifacts.tcn_metrics.uri, checkIfExists: true),
        file(artifacts.tcn_receipt.uri, checkIfExists: true),
        file(artifacts.positive_metrics.uri, checkIfExists: true),
        file(artifacts.positive_receipt.uri, checkIfExists: true),
    ))
    def decisionScript = file("${repoDir}/bin/m38b_decide.py", checkIfExists: true)
    def finalizerScript = file("${repoDir}/bin/m38b_finalize_decision.py", checkIfExists: true)
    def provenanceSources = Channel.value([
        file("${repoDir}/conf/m38b_r0_oof_finalizer_inputs.json", checkIfExists: true),
        file("${repoDir}/conf/m38b_r0_oof_finalizer.config", checkIfExists: true),
        file("${repoDir}/modules/38B_FINALIZE_DECISION.nf", checkIfExists: true),
        file("${repoDir}/workflows/m38b_finalize_decision.nf", checkIfExists: true),
    ])
    M38B_FINALIZE_DECISION(
        scoreInputs,
        inputManifest,
        params.m38b_finalizer_input_manifest_sha256,
        decisionScript,
        finalizerScript,
        provenanceSources,
        params.m38b_finalizer_code_commit,
        params.m38b_finalizer_python_image,
    )
}
