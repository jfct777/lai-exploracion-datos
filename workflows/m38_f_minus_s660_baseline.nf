nextflow.enable.dsl=2

include { M38_F_MINUS_S660_FILTER } from '../modules/38_F_MINUS_S660_FILTER'
include { M38_F_MINUS_S660_BGZIP_TABIX } from '../modules/38_F_MINUS_S660_BGZIP_TABIX'

workflow {
    def required = [
        'm38_fminus_run_id', 'm38_fminus_results_dir', 'm38_fminus_splits',
        'm38_fminus_fit_reference_vcf', 'm38_fminus_fit_target_vcf',
        'm38_fminus_fit_selected_loci', 'm38_fminus_fit_reference_sha256',
        'm38_fminus_fit_target_sha256', 'm38_fminus_fit_selected_sha256',
    ]
    def requestedSplits = params.m38_fminus_splits
    if (!(requestedSplits instanceof List) || requestedSplits.isEmpty())
        error '--m38_fminus_splits must be a non-empty list'
    requestedSplits = requestedSplits.collect { split -> split.toString().toUpperCase() }
    if (requestedSplits != requestedSplits.unique() ||
        !requestedSplits.every { split -> split in ['FIT', 'VALID'] })
        error '--m38_fminus_splits may contain FIT and explicitly requested VALID only'
    if ('VALID' in requestedSplits)
        required.addAll([
            'm38_fminus_valid_reference_vcf', 'm38_fminus_valid_target_vcf',
            'm38_fminus_valid_selected_loci', 'm38_fminus_valid_reference_sha256',
            'm38_fminus_valid_target_sha256', 'm38_fminus_valid_selected_sha256',
        ])
    required.each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m38_fminus_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error '--m38_fminus_run_id must be an explicit safe identifier'
    if (params.m38_fminus_results_dir !=
        'gs://teams-usp/frank/lai-exploracion-datos/runs')
        error 'M38 F-minus-S660 outputs must remain in the personal project bucket'
    if (params.m38_fminus_chromosome.toString().replaceFirst('^chr', '') != '22' ||
        params.m38_fminus_expected_full_loci != 42986 ||
        params.m38_fminus_expected_selected_loci != 660 ||
        params.m38_fminus_expected_reference_samples != 753 ||
        params.m38_fminus_expected_fit_samples != 96 ||
        params.m38_fminus_expected_valid_samples != 32)
        error 'M38 F-minus-S660 R0 axes or counts differ from authenticated M34 artifacts'

    def expectedPython = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99'
    def expectedTabix = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54'
    if (params.m38_fminus_python_image != expectedPython)
        error 'M38 F-minus-S660 Python image differs from the pinned runtime'
    if (params.m38_fminus_tabix_image != expectedTabix)
        error 'M38 F-minus-S660 Tabix image differs from the pinned runtime'

    def runResults = file(
        "${params.m38_fminus_results_dir}/${params.m38_fminus_run_id}",
        checkIfExists: false,
    )
    if (runResults.exists() && !workflow.resume)
        error 'the run-specific results directory already exists; outputs are append-safe'

    def cases = requestedSplits.collect { split ->
        if (split == 'FIT') {
            return tuple(
                'FIT',
                file(params.m38_fminus_fit_reference_vcf, checkIfExists: true),
                file(params.m38_fminus_fit_target_vcf, checkIfExists: true),
                file(params.m38_fminus_fit_selected_loci, checkIfExists: true),
                params.m38_fminus_fit_reference_sha256 as String,
                params.m38_fminus_fit_target_sha256 as String,
                params.m38_fminus_fit_selected_sha256 as String,
                params.m38_fminus_expected_fit_samples as Integer,
            )
        }
        return tuple(
            'VALID',
            file(params.m38_fminus_valid_reference_vcf, checkIfExists: true),
            file(params.m38_fminus_valid_target_vcf, checkIfExists: true),
            file(params.m38_fminus_valid_selected_loci, checkIfExists: true),
            params.m38_fminus_valid_reference_sha256 as String,
            params.m38_fminus_valid_target_sha256 as String,
            params.m38_fminus_valid_selected_sha256 as String,
            params.m38_fminus_expected_valid_samples as Integer,
        )
    }
    if (cases*.get(0) != requestedSplits)
        error 'M38 F-minus-S660 requested case construction drifted'

    def repoDir = projectDir.resolve('..')
    def sources = channel.value([
        file("${repoDir}/bin/m38_build_f_minus_s660.py", checkIfExists: true),
        file("${repoDir}/bin/_experiment_invariants.py", checkIfExists: true),
    ])
    M38_F_MINUS_S660_FILTER(channel.fromList(cases), sources)

    def indexInputs = M38_F_MINUS_S660_FILTER.out.filtered.flatMap {
            split, referenceVcf, targetVcf, filterReceipt ->
        [
            tuple(split, 'REFERENCE', referenceVcf, filterReceipt),
            tuple(split, 'TARGET', targetVcf, filterReceipt),
        ]
    }
    M38_F_MINUS_S660_BGZIP_TABIX(indexInputs)
}
