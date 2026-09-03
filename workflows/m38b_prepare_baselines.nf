nextflow.enable.dsl=2

include { M38B_BUILD_FLARE_CONTRACT } from '../modules/38B_FLARE_CONTRACT'
include { M38B_RUN_FLARE_F_MINUS_S660 } from '../modules/38B_FLARE_RUN'
include { M38B_PARSE_F_MINUS_S660_F0 } from '../modules/38B_PARSE_F0'
include { M38B_PROJECT_FULL_AND_TRUTH } from '../modules/38B_PROJECT_BASELINES'

workflow {
    def required = [
        'm38b_prepare_run_id',
        'm38b_prepare_results_dir',
        'm38b_prepare_reference_vcf',
        'm38b_prepare_reference_tbi',
        'm38b_prepare_target_vcf',
        'm38b_prepare_target_tbi',
        'm38b_prepare_sample_map',
        'm38b_prepare_genetic_map',
        'm38b_prepare_full_f0',
        'm38b_prepare_full_marker_cm',
        'm38b_prepare_full_truth',
        'm38b_prepare_selected_loci',
        'm38b_prepare_flare_jar',
    ]
    required.each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m38b_prepare_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error '--m38b_prepare_run_id must be an explicit safe identifier'
    if (params.m38b_prepare_results_dir !=
        'gs://teams-usp/frank/lai-exploracion-datos/runs')
        error 'M38B outputs must remain in the personal project bucket'
    if (params.m38b_prepare_root != 'R0' || params.m38b_prepare_partition != 'FIT')
        error 'M38B baseline preparation is restricted to R0 FIT'
    if (params.m38b_prepare_expected_full_loci != 42986 ||
        params.m38b_prepare_expected_selected_loci != 660 ||
        params.m38b_prepare_expected_fminus_loci != 42326 ||
        params.m38b_prepare_expected_reference_samples != 753 ||
        params.m38b_prepare_expected_fit_samples != 96)
        error 'M38B baseline counts differ from authenticated M34/M38 artifacts'
    if (params.m38b_prepare_expected_full_loci !=
        params.m38b_prepare_expected_fminus_loci +
        params.m38b_prepare_expected_selected_loci)
        error 'M38B locus-count partition does not close'

    def expectedPython = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99'
    def expectedFlare = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m30-flare-runtime@sha256:86bf36c5d23407ed187d546f2420a0d2c44fbb6eed12ba81ddfc0f75df6b3a84'
    if (params.m38b_prepare_python_image != expectedPython)
        error 'M38B Python image differs from the pinned runtime'
    if (params.m38b_prepare_flare_image != expectedFlare)
        error 'M38B FLARE image differs from the pinned M34 runtime'
    if (params.m38b_prepare_flare_jar != '/opt/flare/flare.jar')
        error 'M38B FLARE jar must be the pinned jar inside the FLARE image'

    def sourcePaths = [
        params.m38b_prepare_reference_vcf,
        params.m38b_prepare_reference_tbi,
        params.m38b_prepare_target_vcf,
        params.m38b_prepare_target_tbi,
        params.m38b_prepare_sample_map,
        params.m38b_prepare_full_f0,
        params.m38b_prepare_full_marker_cm,
        params.m38b_prepare_full_truth,
        params.m38b_prepare_selected_loci,
    ]*.toString()
    if (sourcePaths.any { path ->
        def lowered = path.toLowerCase()
        lowered.contains('/valid/') || lowered.contains('/test/')
    })
        error 'M38B baseline preparation cannot stage VALID or TEST artifacts'
    if (!sourcePaths.take(4).every { path ->
        path.startsWith('gs://teams-usp/frank/lai-exploracion-datos/runs/' +
                        'm38-r0-f-minus-s660-baseline-20260903a/fit/indexed/')
    })
        error 'M38B F-minus-S660 VCFs are not the canonical FIT derivatives'

    def hashKeys = [
        'm38b_prepare_reference_vcf_sha256',
        'm38b_prepare_reference_tbi_sha256',
        'm38b_prepare_target_vcf_sha256',
        'm38b_prepare_target_tbi_sha256',
        'm38b_prepare_sample_map_sha256',
        'm38b_prepare_genetic_map_sha256',
        'm38b_prepare_flare_jar_sha256',
        'm38b_prepare_full_f0_sha256',
        'm38b_prepare_full_marker_cm_sha256',
        'm38b_prepare_full_truth_sha256',
        'm38b_prepare_selected_loci_sha256',
    ]
    if (!hashKeys.every { key ->
        params[key] instanceof String && params[key] ==~ /[0-9a-f]{64}/
    })
        error 'one or more M38B canonical SHA-256 values are missing or malformed'

    def runResults = file(
        "${params.m38b_prepare_results_dir}/${params.m38b_prepare_run_id}",
        checkIfExists: false,
    )
    if (runResults.exists() && !workflow.resume)
        error 'the run-specific results directory already exists; outputs are append-safe'

    def repoDir = projectDir.resolve('..')
    def experimentFile = file(
        "${repoDir}/conf/m38b_r0_oof_contract.json", checkIfExists: true,
    )
    def experiment = new groovy.json.JsonSlurper().parse(experimentFile)
    if (experiment.experiment_id != 'M38B_S660_INCREMENTAL_LAI_CHR22_R0_FIT' ||
        experiment.status != 'PREREGISTERED_AMENDED_BEFORE_OUTCOME_ACCESS' ||
        experiment.claim_scope.target_partition != 'FIT_ONLY' ||
        experiment.claim_scope.valid_opened != false ||
        experiment.claim_scope.test_opened != false ||
        experiment.locus_universes.f_full_count != 42986 ||
        experiment.locus_universes.s660_count != 660 ||
        experiment.locus_universes.f_minus_s660_count != 42326)
        error 'the local M38B preregistration does not match the FIT baseline task'

    referenceVcf = Channel.value(file(params.m38b_prepare_reference_vcf, checkIfExists: true))
    referenceTbi = Channel.value(file(params.m38b_prepare_reference_tbi, checkIfExists: true))
    targetVcf = Channel.value(file(params.m38b_prepare_target_vcf, checkIfExists: true))
    targetTbi = Channel.value(file(params.m38b_prepare_target_tbi, checkIfExists: true))
    sampleMap = Channel.value(file(params.m38b_prepare_sample_map, checkIfExists: true))
    geneticMap = Channel.value(file(params.m38b_prepare_genetic_map, checkIfExists: true))
    fullF0 = Channel.value(file(params.m38b_prepare_full_f0, checkIfExists: true))
    fullMarkerCm = Channel.value(file(params.m38b_prepare_full_marker_cm, checkIfExists: true))
    fullTruth = Channel.value(file(params.m38b_prepare_full_truth, checkIfExists: true))
    selectedLoci = Channel.value(file(params.m38b_prepare_selected_loci, checkIfExists: true))
    experimentContract = Channel.value(experimentFile)

    buildContractPy = Channel.value(file(
        "${repoDir}/bin/m38b_build_flare_contract.py", checkIfExists: true,
    ))
    flareSources = Channel.value([
        file("${repoDir}/bin/m38b_run_flare.py", checkIfExists: true),
        file("${repoDir}/bin/m38b_build_flare_contract.py", checkIfExists: true),
        file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true),
    ])
    parseSources = Channel.value([
        file("${repoDir}/bin/m38b_parse_flare.py", checkIfExists: true),
        file("${repoDir}/bin/m34_parse_flare_truth.py", checkIfExists: true),
        file("${repoDir}/bin/m34_generate_mosaics.py", checkIfExists: true),
        file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true),
    ])
    projectSources = Channel.value([
        file("${repoDir}/bin/m38b_project_baselines.py", checkIfExists: true),
        file("${repoDir}/bin/m34_parse_flare_truth.py", checkIfExists: true),
        file("${repoDir}/bin/m34_generate_mosaics.py", checkIfExists: true),
        file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true),
        file("${repoDir}/bin/m38_build_f_minus_s660.py", checkIfExists: true),
        file("${repoDir}/bin/_experiment_invariants.py", checkIfExists: true),
    ])

    M38B_BUILD_FLARE_CONTRACT(
        referenceVcf,
        referenceTbi,
        targetVcf,
        targetTbi,
        sampleMap,
        geneticMap,
        experimentContract,
        params.m38b_prepare_flare_jar as String,
        buildContractPy,
    )
    M38B_RUN_FLARE_F_MINUS_S660(
        M38B_BUILD_FLARE_CONTRACT.out.contracted,
        geneticMap,
        params.m38b_prepare_flare_jar as String,
        flareSources,
    )
    M38B_PARSE_F_MINUS_S660_F0(
        M38B_RUN_FLARE_F_MINUS_S660.out.baseline,
        geneticMap,
        parseSources,
    )
    M38B_PROJECT_FULL_AND_TRUTH(
        M38B_PARSE_F_MINUS_S660_F0.out.parsed,
        fullF0,
        fullMarkerCm,
        fullTruth,
        selectedLoci,
        experimentContract,
        projectSources,
    )
}
