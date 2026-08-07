nextflow.enable.dsl=2

include { RARE_WINDOW_FEATURES } from '../subworkflows/25_RARE_WINDOW_FEATURES'


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
    def features_py = file("${repoDir}/bin/build_rare_window_features.py")
    def manifest_py = file("${repoDir}/bin/write_stage_manifest.py")
    def rare_vcfs = discoverRareVcfs(
        params.rare_window_features_input_dir,
        params.rare_window_features_input_glob,
    )
    RARE_WINDOW_FEATURES(rare_vcfs, features_py, manifest_py)
}
