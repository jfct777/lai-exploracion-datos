nextflow.enable.dsl=2

import groovy.json.JsonOutput

include {
    BUILD_RARE_WINDOW_FEATURES;
    WRITE_RARE_WINDOW_FEATURES_RUN_PROVENANCE
} from '../modules/25_RARE_WINDOW_FEATURES'


def resolveRareWindowFeaturesGitCommit(dir) {
    try {
        def head = new File("${dir}/.git/HEAD").text.trim()
        if( !head.startsWith('ref:') ) return head
        def ref = head.substring(4).trim()
        def refFile = new File("${dir}/.git/${ref}")
        if( refFile.exists() ) return refFile.text.trim()
        def packed = new File("${dir}/.git/packed-refs")
        if( packed.exists() )
            for( line in packed.readLines() )
                if( line.endsWith(" ${ref}")) return line.split(' ')[0]
    } catch( ignored ) { }
    return 'unknown'
}


workflow RARE_WINDOW_FEATURES {
    take:
    rare_vcfs
    features_py
    manifest_py

    main:
    def feature_chr = params.rare_window_features_chromosome.toString().replaceFirst('(?i)^chr', '')
    if( !(feature_chr ==~ /\d+/) || feature_chr.toInteger() < 1 || feature_chr.toInteger() > 22 )
        throw new IllegalStateException("M25: rare_window_features_chromosome debe ser un autosoma 1..22")
    if( !params.rare_window_features_split_manifest ||
        !params.rare_window_features_upstream_qc ||
        !params.rare_window_features_upstream_manifest )
        throw new IllegalStateException("M25: faltan split manifest o artefactos de QC/provenance M24")

    def provenance = [
        git_commit       : System.getenv('DNABR_GIT_COMMIT') ?:
                           resolveRareWindowFeaturesGitCommit(projectDir.resolve('..').toString()),
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.m14_analysis_container_image ?: params.container_image,
        container_sha256 : params.m14_analysis_container_digest ?: params.container_digest ?: 'unavailable',
    ]
    def provenance_b64 = JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def run_provenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'descriptive TRAIN-transductive chr-window features; no training or evaluation',
        ascertainment    : 'cohort-ascertained lai_rare universe; MAC/MAF/orientation recomputed in TRAIN',
    ]
    def run_provenance_b64 = JsonOutput.prettyPrint(
        JsonOutput.toJson(run_provenance)
    ).bytes.encodeBase64().toString()
    WRITE_RARE_WINDOW_FEATURES_RUN_PROVENANCE(channel.value(run_provenance_b64))

    def reference_fai = file("${params.ref_fasta}.fai")
    def split_manifest = file(params.rare_window_features_split_manifest)
    def upstream_qc = file(params.rare_window_features_upstream_qc)
    def upstream_manifest = file(params.rare_window_features_upstream_manifest)
    def m25_inputs = rare_vcfs
        .filter { chr, _vcf, _tbi -> chr.toString().replaceFirst('(?i)^chr', '') == feature_chr }
        .map { _chr, vcf, vcf_tbi ->
            tuple(
                feature_chr, vcf, vcf_tbi, reference_fai, split_manifest,
                upstream_qc, upstream_manifest, features_py, manifest_py,
            )
        }
    BUILD_RARE_WINDOW_FEATURES(m25_inputs, channel.value(provenance_b64))

    emit:
    features = BUILD_RARE_WINDOW_FEATURES.out.features
    qc = BUILD_RARE_WINDOW_FEATURES.out.qc
    manifest = BUILD_RARE_WINDOW_FEATURES.out.manifest
}
