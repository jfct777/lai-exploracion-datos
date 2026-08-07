nextflow.enable.dsl=2

import groovy.json.JsonOutput

include {
    BUILD_TRAIN_DIAGNOSTIC_COVARIATES;
    BUILD_FOLD_RARE_WINDOW_FEATURES;
    EVALUATE_FOLD_RARE_WINDOW_PCA;
    WRITE_RARE_WINDOW_PCA_RUN_PROVENANCE
} from '../modules/25B_RARE_WINDOW_PCA'


def resolveM25BGitCommit(dir) {
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


workflow RARE_WINDOW_PCA {
    take:
    rare_vcfs
    fold_features_py
    base_features_py
    evaluate_py
    covariates_py
    manifest_py

    main:
    def chrom = params.rare_window_pca_chromosome.toString().replaceFirst('(?i)^chr', '')
    if( !(chrom ==~ /\d+/) || chrom.toInteger() < 1 || chrom.toInteger() > 22 )
        throw new IllegalStateException('M25B chromosome must be an autosome 1..22')
    if( !params.rare_window_pca_split_manifest || !params.rare_window_pca_feature_store ||
        !params.rare_window_pca_upstream_qc || !params.rare_window_pca_upstream_manifest )
        throw new IllegalStateException('M25B missing canonical split, covariates or M24 provenance')

    def provenance = [
        git_commit              : System.getenv('DNABR_GIT_COMMIT') ?:
                                  resolveM25BGitCommit(projectDir.resolve('..').toString()),
        nextflow_version        : workflow.nextflow.version.toString(),
        container_path          : params.m16_5_analysis_container_image,
        container_sha256        : params.m16_5_analysis_container_digest,
    ]
    def provenance_b64 = JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'M25B internal OOF PCA on canonical TRAIN; TEST genotypes are not emitted and TEST values are not scored',
        ascertainment    : 'cohort-ascertained lai_rare; selection/orientation recomputed in each outer FIT',
        rank_policy      : 'fixed capacity curve; no rank selected by reconstruction error',
    ]
    def runProvenanceB64 = JsonOutput.prettyPrint(
        JsonOutput.toJson(runProvenance)
    ).bytes.encodeBase64().toString()
    WRITE_RARE_WINDOW_PCA_RUN_PROVENANCE(channel.value(runProvenanceB64))

    def referenceFai = file("${params.ref_fasta}.fai")
    def splitManifest = file(params.rare_window_pca_split_manifest)
    def upstreamQc = file(params.rare_window_pca_upstream_qc)
    def upstreamManifest = file(params.rare_window_pca_upstream_manifest)
    def featureStore = file(params.rare_window_pca_feature_store)
    BUILD_TRAIN_DIAGNOSTIC_COVARIATES(
        splitManifest,
        featureStore,
        covariates_py,
        base_features_py,
        manifest_py,
        channel.value(provenance_b64),
    )
    def m25bInput = rare_vcfs
        .filter { chr, _vcf, _tbi -> chr.toString().replaceFirst('(?i)^chr', '') == chrom }
        .collect(flat: false)
        .map { rows ->
            if( rows.size() != 1 )
                throw new IllegalStateException("M25B expected exactly one VCF for chr${chrom}; found ${rows.size()}")
            def (_chr, vcf, vcfTbi) = rows[0]
            tuple(
                chrom, vcf, vcfTbi, referenceFai, splitManifest, upstreamQc,
                upstreamManifest, fold_features_py, base_features_py, manifest_py,
            )
        }
    BUILD_FOLD_RARE_WINDOW_FEATURES(m25bInput, channel.value(provenance_b64))
    EVALUATE_FOLD_RARE_WINDOW_PCA(
        chrom,
        BUILD_FOLD_RARE_WINDOW_FEATURES.out.features,
        BUILD_FOLD_RARE_WINDOW_FEATURES.out.windows,
        BUILD_FOLD_RARE_WINDOW_FEATURES.out.qc,
        BUILD_FOLD_RARE_WINDOW_FEATURES.out.manifest,
        splitManifest,
        BUILD_TRAIN_DIAGNOSTIC_COVARIATES.out.covariates,
        BUILD_TRAIN_DIAGNOSTIC_COVARIATES.out.audit,
        BUILD_TRAIN_DIAGNOSTIC_COVARIATES.out.manifest,
        evaluate_py,
        base_features_py,
        manifest_py,
        channel.value(provenance_b64),
    )

    emit:
    summary = EVALUATE_FOLD_RARE_WINDOW_PCA.out.summary
    manifest = EVALUATE_FOLD_RARE_WINDOW_PCA.out.manifest
}
