nextflow.enable.dsl=2

include { WRITE_M29_ROOT_GNOMIX_PROVENANCE; BIND_M29_ROOT_GNOMIX_CONTRACT; VALIDATE_M29_ROOT_GNOMIX_B0; TRAIN_M29_ROOT_GNOMIX_B0; INFER_M29_ROOT_GNOMIX_B0; BIND_M29_ROOT_PREDICTIONS } from '../modules/29_ROOT_GNOMIX_B0'

workflow {
    if (!(params.m29_gnomix_max_parallel_roots in [1, 2])) {
        error '--m29_gnomix_max_parallel_roots must be 1 or 2'
    }
    def hostMemLine = new File('/proc/meminfo').readLines().find { it.startsWith('MemTotal:') }
    def hostMemGiB = hostMemLine ? hostMemLine.split()[1].toDouble() / (1024.0 * 1024.0) : 0.0
    if (params.m29_gnomix_max_parallel_roots == 2 && hostMemGiB < params.m29_gnomix_min_host_memory_gib.toDouble()) {
        error "Two parallel roots require at least ${params.m29_gnomix_min_host_memory_gib} GiB host RAM; observed ${String.format('%.2f', hostMemGiB)} GiB. Set --m29_gnomix_max_parallel_roots 1 or use the approved VM."
    }
    def required = [
        params.m29_gnomix_root17_reference_vcf, params.m29_gnomix_root17_reference_tbi,
        params.m29_gnomix_root17_target_vcf, params.m29_gnomix_root17_target_tbi,
        params.m29_gnomix_root17_sample_map, params.m29_gnomix_root17_b0_markers,
        params.m29_gnomix_root17_selection_report, params.m29_gnomix_root17_selection_manifest,
        params.m29_gnomix_root17_materialization_report, params.m29_gnomix_root17_materialization_manifest,
        params.m29_gnomix_root17_ingest_report, params.m29_gnomix_root17_ingest_manifest,
        params.m29_gnomix_root18_reference_vcf, params.m29_gnomix_root18_reference_tbi,
        params.m29_gnomix_root18_target_vcf, params.m29_gnomix_root18_target_tbi,
        params.m29_gnomix_root18_sample_map, params.m29_gnomix_root18_b0_markers,
        params.m29_gnomix_root18_selection_report, params.m29_gnomix_root18_selection_manifest,
        params.m29_gnomix_root18_materialization_report, params.m29_gnomix_root18_materialization_manifest,
        params.m29_gnomix_root18_ingest_report, params.m29_gnomix_root18_ingest_manifest,
        params.m29_gnomix_genetic_map, params.m29_gnomix_config, params.m29_gnomix_preregistration,
        params.m29_gnomix_production_contract, params.m29_gnomix_template_contract
    ]
    if (required.any { value -> !value }) error 'All persistent root17/root18 B0 artifacts and manifests are required'
    def repoDir = projectDir.resolve('..')
    roots = Channel.of(
        tuple('root17', 20260817, file(params.m29_gnomix_root17_reference_vcf, checkIfExists: true), file(params.m29_gnomix_root17_reference_tbi, checkIfExists: true), file(params.m29_gnomix_root17_target_vcf, checkIfExists: true), file(params.m29_gnomix_root17_target_tbi, checkIfExists: true), file(params.m29_gnomix_root17_sample_map, checkIfExists: true), file(params.m29_gnomix_root17_b0_markers, checkIfExists: true), file(params.m29_gnomix_root17_selection_report, checkIfExists: true), file(params.m29_gnomix_root17_selection_manifest, checkIfExists: true), file(params.m29_gnomix_root17_materialization_report, checkIfExists: true), file(params.m29_gnomix_root17_materialization_manifest, checkIfExists: true), file(params.m29_gnomix_root17_ingest_report, checkIfExists: true), file(params.m29_gnomix_root17_ingest_manifest, checkIfExists: true)),
        tuple('root18', 20260818, file(params.m29_gnomix_root18_reference_vcf, checkIfExists: true), file(params.m29_gnomix_root18_reference_tbi, checkIfExists: true), file(params.m29_gnomix_root18_target_vcf, checkIfExists: true), file(params.m29_gnomix_root18_target_tbi, checkIfExists: true), file(params.m29_gnomix_root18_sample_map, checkIfExists: true), file(params.m29_gnomix_root18_b0_markers, checkIfExists: true), file(params.m29_gnomix_root18_selection_report, checkIfExists: true), file(params.m29_gnomix_root18_selection_manifest, checkIfExists: true), file(params.m29_gnomix_root18_materialization_report, checkIfExists: true), file(params.m29_gnomix_root18_materialization_manifest, checkIfExists: true), file(params.m29_gnomix_root18_ingest_report, checkIfExists: true), file(params.m29_gnomix_root18_ingest_manifest, checkIfExists: true))
    )
    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: workflow.commitId
    if (!gitCommit) error 'Set DNABR_GIT_COMMIT to the exact source commit'
    provenance = [
        git_commit: gitCommit,
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        container_path: params.m29_gnomix_container,
        container_sha256: params.m29_gnomix_container_digest,
        container_options: params.m29_gnomix_container_options,
        scientific_scope: 'M29 root-specific B0 training and inference; no truth or effect estimation',
        roots: [20260817, 20260818],
        max_parallel_roots: params.m29_gnomix_max_parallel_roots,
        memory_per_root: params.m29_gnomix_memory,
        peak_rss_stop_gib: params.m29_gnomix_peak_rss_stop_gib,
        nextflow_command: workflow.commandLine
    ]
    provenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(provenance)).bytes.encodeBase64().toString()
    geneticMap = file(params.m29_gnomix_genetic_map, checkIfExists: true)
    gnomixConfig = file(params.m29_gnomix_config, checkIfExists: true)
    preregistration = file(params.m29_gnomix_preregistration, checkIfExists: true)
    productionContract = file(params.m29_gnomix_production_contract, checkIfExists: true)
    templateContract = file(params.m29_gnomix_template_contract, checkIfExists: true)
    builderPy = file("${repoDir}/bin/build_m29_root_gnomix_contract.py", checkIfExists: true)
    runnerPy = file("${repoDir}/bin/m28c_gnomix_training_smoke.py", checkIfExists: true)
    rssGuardPy = file("${repoDir}/bin/run_with_rss_guard.py", checkIfExists: true)
    manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)
    binderPy = file("${repoDir}/bin/bind_m29_b0_inputs.py", checkIfExists: true)

    WRITE_M29_ROOT_GNOMIX_PROVENANCE(channel.value(provenanceB64))
    BIND_M29_ROOT_GNOMIX_CONTRACT(roots, geneticMap, gnomixConfig, preregistration, productionContract, templateContract, builderPy, runnerPy)
    VALIDATE_M29_ROOT_GNOMIX_B0(BIND_M29_ROOT_GNOMIX_CONTRACT.out.bound, geneticMap, gnomixConfig, runnerPy, manifestPy, WRITE_M29_ROOT_GNOMIX_PROVENANCE.out.provenance, channel.value(provenanceB64))
    TRAIN_M29_ROOT_GNOMIX_B0(VALIDATE_M29_ROOT_GNOMIX_B0.out.training_ready, geneticMap, gnomixConfig, runnerPy, rssGuardPy, manifestPy, WRITE_M29_ROOT_GNOMIX_PROVENANCE.out.provenance, channel.value(provenanceB64))
    inferenceInputs = TRAIN_M29_ROOT_GNOMIX_B0.out.trained
        .join(VALIDATE_M29_ROOT_GNOMIX_B0.out.targets_ready, by: 0)
        .map { root_label, root_seed, training_dir, runtime_contract, validation_report, training_rss_gate, target_vcf ->
            tuple(root_label, root_seed, training_dir, target_vcf, runtime_contract, validation_report, training_rss_gate)
        }
    INFER_M29_ROOT_GNOMIX_B0(inferenceInputs, gnomixConfig, runnerPy, manifestPy, WRITE_M29_ROOT_GNOMIX_PROVENANCE.out.provenance, channel.value(provenanceB64))
    BIND_M29_ROOT_PREDICTIONS(INFER_M29_ROOT_GNOMIX_B0.out.predictions, binderPy, WRITE_M29_ROOT_GNOMIX_PROVENANCE.out.provenance)
}
