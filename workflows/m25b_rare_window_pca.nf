nextflow.enable.dsl=2

include { RARE_WINDOW_PCA } from '../subworkflows/25B_RARE_WINDOW_PCA'


def discoverRareVcfs(String inputDir, String globPattern) {
    def pattern = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.rare\.vcf\.gz$/
    return channel
        .fromPath("${inputDir}/${globPattern}")
        .filter { path -> path.getName() ==~ pattern }
        .map { vcf ->
            def match = (vcf.getName() =~ pattern)
            if( !match.matches() ) throw new IllegalArgumentException("Cannot extract chromosome from ${vcf}")
            def index = vcf.resolveSibling("${vcf.getName()}.tbi")
            if( !index.exists() ) throw new IllegalStateException("Missing .tbi for ${vcf}")
            tuple(match[0][1], vcf, index)
        }
}


workflow {
    def repoDir = projectDir.resolve('..')
    def foldFeaturesPy = file("${repoDir}/bin/build_fold_rare_window_features.py")
    def baseFeaturesPy = file("${repoDir}/bin/build_rare_window_features.py")
    def evaluatePy = file("${repoDir}/bin/evaluate_fold_rare_window_pca.py")
    def covariatesPy = file("${repoDir}/bin/build_train_diagnostic_covariates.py")
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py")
    def rareVcfs = discoverRareVcfs(
        params.rare_window_pca_input_dir,
        params.rare_window_pca_input_glob,
    )
    RARE_WINDOW_PCA(
        rareVcfs, foldFeaturesPy, baseFeaturesPy, evaluatePy, covariatesPy, manifestPy
    )
}
