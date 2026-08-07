nextflow.enable.dsl=2

import groovy.json.JsonOutput

include {
    EVALUATE_RARE_WINDOW_SCALE_SENSITIVITY;
    WRITE_RARE_WINDOW_SCALE_RUN_PROVENANCE
} from '../modules/25C_RARE_WINDOW_SCALE_SENSITIVITY'


def resolveM25CGitCommit(dir) {
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


workflow RARE_WINDOW_SCALE_SENSITIVITY {
    take:
    fold_features
    fold_windows
    fold_qc
    fold_manifest
    split_manifest
    diagnostic_covariates
    diagnostic_covariates_audit
    diagnostic_covariates_manifest
    preregistration
    evaluate_scale_py
    evaluate_pca_py
    base_features_py
    manifest_py

    main:
    def chrom = params.rare_window_scale_chromosome.toString().replaceFirst('(?i)^chr', '')
    if( chrom != '22' )
        throw new IllegalStateException('M25C is preregistered for chr22 only')

    def provenance = [
        git_commit       : System.getenv('DNABR_GIT_COMMIT') ?:
                           resolveM25CGitCommit(projectDir.resolve('..').toString()),
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.m16_5_analysis_container_image,
        container_sha256 : params.m16_5_analysis_container_digest,
    ]
    if( provenance.git_commit == 'unknown' )
        throw new IllegalStateException('M25C requires a resolved git commit')
    def provenanceB64 = JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'M25C post-hoc scale/denominator sensitivity on chr22 canonical TRAIN; TEST remains blind',
        overlap_policy   : 'half-step grids evaluated as separate non-overlapping phases; phases are never concatenated',
        branching_policy : 'no automatic NMF/AE/LAI/TEST or autosomal scaling',
    ]
    def runProvenanceB64 = JsonOutput.prettyPrint(
        JsonOutput.toJson(runProvenance)
    ).bytes.encodeBase64().toString()
    WRITE_RARE_WINDOW_SCALE_RUN_PROVENANCE(channel.value(runProvenanceB64))

    EVALUATE_RARE_WINDOW_SCALE_SENSITIVITY(
        chrom,
        fold_features,
        fold_windows,
        fold_qc,
        fold_manifest,
        split_manifest,
        diagnostic_covariates,
        diagnostic_covariates_audit,
        diagnostic_covariates_manifest,
        preregistration,
        evaluate_scale_py,
        evaluate_pca_py,
        base_features_py,
        manifest_py,
        channel.value(provenanceB64),
    )

    emit:
    summary = EVALUATE_RARE_WINDOW_SCALE_SENSITIVITY.out.summary
    manifest = EVALUATE_RARE_WINDOW_SCALE_SENSITIVITY.out.manifest
}
