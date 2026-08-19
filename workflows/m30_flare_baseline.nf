nextflow.enable.dsl=2

include {
    WRITE_M30_FLARE_PROVENANCE
    M30_SCORER_KNOWN_ANSWERS
    M30_PREFLIGHT_ROOT17
    M30_RUN_FLARE_ROOT17
    M30_PREFLIGHT_ROOT18
    M30_RUN_FLARE_ROOT18
    M30_SCORE_FLARE_VS_GNOMIX
} from '../modules/30_FLARE_BASELINE'

workflow {
    def required = [
        params.m30_run_id, params.m30_preregistration, params.m30_genetic_map,
        params.m30_container_image, params.m30_container_digest, params.m30_flare_jar,
        params.m30_root17_truth, params.m30_root18_truth,
        params.m30_root17_reference_vcf, params.m30_root17_reference_tbi,
        params.m30_root17_target_vcf, params.m30_root17_target_tbi,
        params.m30_root17_sample_map, params.m30_root17_gnomix_binding,
        params.m30_root17_gnomix_fb, params.m30_root17_gnomix_msp,
        params.m30_root18_reference_vcf, params.m30_root18_reference_tbi,
        params.m30_root18_target_vcf, params.m30_root18_target_tbi,
        params.m30_root18_sample_map, params.m30_root18_gnomix_binding,
        params.m30_root18_gnomix_fb, params.m30_root18_gnomix_msp
    ]
    if (required.any { value -> !value }) {
        error 'M30 requires a run ID, immutable container, pinned FLARE JAR, map, and all root17/root18 inputs'
    }
    if (!(params.m30_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) {
        error '--m30_run_id contains unsupported characters'
    }
    if (!params.m30_container_image.contains('@sha256:')) {
        error '--m30_container_image must use an immutable @sha256 digest'
    }
    if (!params.m30_container_image.endsWith(params.m30_container_digest)) {
        error '--m30_container_digest does not match --m30_container_image'
    }
    if (params.m30_flare_jar_sha256 != '8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420') {
        error '--m30_flare_jar_sha256 differs from the preregistered FLARE v0.6.0 JAR'
    }
    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: workflow.commitId
    if (!gitCommit) error 'Set DNABR_GIT_COMMIT to the exact source commit'

    def repoDir = projectDir.resolve('..')
    def provenance = [
        experiment_id: 'M30_FLARE_BASELINE',
        git_commit: gitCommit,
        nextflow_version: workflow.nextflow.version.toString(),
        nextflow_command: workflow.commandLine,
        run_id: params.m30_run_id,
        container_image: params.m30_container_image,
        container_digest: params.m30_container_digest,
        flare_jar_sha256: params.m30_flare_jar_sha256,
        execution_order: ['root17_smoke', 'root18_after_root17_pass'],
        truth_accessed: false,
        scoring_implemented: false,
        flare2_status: 'DEFERRED_EXACT_THREE_PANEL_MATCH',
        provenance_scope: 'inference_only',
        scoring_permitted_in_inference: false,
        separate_scoring_stage_implemented: true
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()

    preregistration = channel.value(file(params.m30_preregistration, checkIfExists: true))
    geneticMap = channel.value(file(params.m30_genetic_map, checkIfExists: true))
    flareJar = channel.value(file(params.m30_flare_jar, checkIfExists: true))
    runnerPy = channel.value(file("${repoDir}/bin/m30_flare_baseline.py", checkIfExists: true))
    scorerPy = channel.value(file("${repoDir}/bin/m30_flare_scorer.py", checkIfExists: true))
    baseScorerPy = channel.value(file("${repoDir}/bin/m28d_b0_scorer.py", checkIfExists: true))
    root17Reference = channel.value(file(params.m30_root17_reference_vcf, checkIfExists: true))
    root17Target = channel.value(file(params.m30_root17_target_vcf, checkIfExists: true))
    root18Reference = channel.value(file(params.m30_root18_reference_vcf, checkIfExists: true))
    root18Target = channel.value(file(params.m30_root18_target_vcf, checkIfExists: true))
    root17Truth = channel.value(file(params.m30_root17_truth, checkIfExists: true))
    root18Truth = channel.value(file(params.m30_root18_truth, checkIfExists: true))
    root17GnomixBinding = channel.value(file(params.m30_root17_gnomix_binding, checkIfExists: true))
    root17GnomixFb = channel.value(file(params.m30_root17_gnomix_fb, checkIfExists: true))
    root17GnomixMsp = channel.value(file(params.m30_root17_gnomix_msp, checkIfExists: true))
    root18GnomixBinding = channel.value(file(params.m30_root18_gnomix_binding, checkIfExists: true))
    root18GnomixFb = channel.value(file(params.m30_root18_gnomix_fb, checkIfExists: true))
    root18GnomixMsp = channel.value(file(params.m30_root18_gnomix_msp, checkIfExists: true))

    root17Inputs = channel.value(tuple(
        'root17', 20260817,
        file(params.m30_root17_reference_vcf, checkIfExists: true),
        file(params.m30_root17_reference_tbi, checkIfExists: true),
        file(params.m30_root17_target_vcf, checkIfExists: true),
        file(params.m30_root17_target_tbi, checkIfExists: true),
        file(params.m30_root17_sample_map, checkIfExists: true),
        file(params.m30_root17_gnomix_binding, checkIfExists: true),
        file(params.m30_root17_gnomix_fb, checkIfExists: true),
        file(params.m30_root17_gnomix_msp, checkIfExists: true)
    ))
    root18Inputs = channel.value(tuple(
        'root18', 20260818,
        file(params.m30_root18_reference_vcf, checkIfExists: true),
        file(params.m30_root18_reference_tbi, checkIfExists: true),
        file(params.m30_root18_target_vcf, checkIfExists: true),
        file(params.m30_root18_target_tbi, checkIfExists: true),
        file(params.m30_root18_sample_map, checkIfExists: true),
        file(params.m30_root18_gnomix_binding, checkIfExists: true),
        file(params.m30_root18_gnomix_fb, checkIfExists: true),
        file(params.m30_root18_gnomix_msp, checkIfExists: true)
    ))

    WRITE_M30_FLARE_PROVENANCE(channel.value(provenanceB64))
    M30_SCORER_KNOWN_ANSWERS(
        preregistration,
        scorerPy,
        baseScorerPy,
        WRITE_M30_FLARE_PROVENANCE.out.provenance
    )
    M30_PREFLIGHT_ROOT17(root17Inputs, geneticMap, preregistration, runnerPy)
    M30_RUN_FLARE_ROOT17(
        M30_PREFLIGHT_ROOT17.out.prepared,
        root17Reference,
        root17Target,
        flareJar,
        runnerPy,
        WRITE_M30_FLARE_PROVENANCE.out.provenance,
        M30_SCORER_KNOWN_ANSWERS.out.receipt,
        preregistration,
        scorerPy
    )
    M30_PREFLIGHT_ROOT18(
        root18Inputs,
        geneticMap,
        preregistration,
        runnerPy,
        M30_RUN_FLARE_ROOT17.out.audit
    )
    M30_RUN_FLARE_ROOT18(
        M30_PREFLIGHT_ROOT18.out.prepared,
        root18Reference,
        root18Target,
        flareJar,
        runnerPy,
        WRITE_M30_FLARE_PROVENANCE.out.provenance,
        M30_SCORER_KNOWN_ANSWERS.out.receipt,
        preregistration,
        scorerPy
    )
    M30_SCORE_FLARE_VS_GNOMIX(
        M30_PREFLIGHT_ROOT17.out.prepared,
        M30_RUN_FLARE_ROOT17.out.predictions,
        M30_RUN_FLARE_ROOT17.out.audit,
        M30_PREFLIGHT_ROOT18.out.prepared,
        M30_RUN_FLARE_ROOT18.out.predictions,
        M30_RUN_FLARE_ROOT18.out.audit,
        root17Truth,
        root17Target,
        root17GnomixBinding,
        root17GnomixFb,
        root17GnomixMsp,
        root18Truth,
        root18Target,
        root18GnomixBinding,
        root18GnomixFb,
        root18GnomixMsp,
        geneticMap,
        preregistration,
        scorerPy,
        baseScorerPy,
        M30_SCORER_KNOWN_ANSWERS.out.receipt,
        WRITE_M30_FLARE_PROVENANCE.out.provenance
    )
}
