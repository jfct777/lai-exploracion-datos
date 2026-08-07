nextflow.enable.dsl=2

include { RARE_WINDOW_SCALE_SENSITIVITY } from '../subworkflows/25C_RARE_WINDOW_SCALE_SENSITIVITY'


workflow {
    def repoDir = projectDir.resolve('..')
    def chrom = params.rare_window_scale_chromosome.toString().replaceFirst('(?i)^chr', '')
    def inputRoot = params.rare_window_scale_input_dir
    def foldFeatures = channel
        .fromPath("${inputRoot}/fold_features/chr${chrom}.fold*.sample_window_features.tsv.gz", checkIfExists: true)
        .collect()
    def foldWindows = channel
        .fromPath("${inputRoot}/fold_features/chr${chrom}.fold*.windows.tsv", checkIfExists: true)
        .collect()
    def foldQc = file("${inputRoot}/fold_features/chr${chrom}.fold_features_qc.json", checkIfExists: true)
    def foldManifest = file(
        "${inputRoot}/fold_features/chr${chrom}.fold_features.manifest.json",
        checkIfExists: true,
    )
    def splitManifest = file(params.rare_window_scale_split_manifest, checkIfExists: true)
    def diagnosticCovariates = file(
        "${inputRoot}/diagnostic_covariates/train_diagnostic_covariates.tsv",
        checkIfExists: true,
    )
    def diagnosticCovariatesAudit = file(
        "${inputRoot}/diagnostic_covariates/train_diagnostic_covariates.audit.json",
        checkIfExists: true,
    )
    def diagnosticCovariatesManifest = file(
        "${inputRoot}/diagnostic_covariates/train_diagnostic_covariates.manifest.json",
        checkIfExists: true,
    )
    def preregistration = file(
        "${repoDir}/conf/m25c_scale_sensitivity_preregistration.json",
        checkIfExists: true,
    )
    def evaluateScalePy = file("${repoDir}/bin/evaluate_rare_window_scale_sensitivity.py")
    def evaluatePcaPy = file("${repoDir}/bin/evaluate_fold_rare_window_pca.py")
    def baseFeaturesPy = file("${repoDir}/bin/build_rare_window_features.py")
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py")

    RARE_WINDOW_SCALE_SENSITIVITY(
        foldFeatures,
        foldWindows,
        foldQc,
        foldManifest,
        splitManifest,
        diagnosticCovariates,
        diagnosticCovariatesAudit,
        diagnosticCovariatesManifest,
        preregistration,
        evaluateScalePy,
        evaluatePcaPy,
        baseFeaturesPy,
        manifestPy,
    )
}
