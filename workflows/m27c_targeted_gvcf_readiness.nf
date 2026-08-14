nextflow.enable.dsl=2

include {
    WRITE_TARGETED_GVCF_RUN_PROVENANCE;
    BENCHMARK_TARGETED_GVCF_ACCESS;
    AUDIT_TARGETED_GVCF_HEADERS;
    AUDIT_TARGETED_GVCF_READINESS
} from '../modules/27C_TARGETED_GVCF_READINESS'

workflow {
    def repoDir = projectDir.resolve('..')
    def gvcfs = channel
        .fromPath(params.targeted_gvcf_gvcf_glob, checkIfExists: true)
        .collect()
        .map { paths -> paths.sort { left, right -> left.getName() <=> right.getName() } }
    def indexes = channel
        .fromPath(params.targeted_gvcf_index_glob, checkIfExists: true)
        .collect()
        .map { paths -> paths.sort { left, right -> left.getName() <=> right.getName() } }
    def gnomixReference = file(params.targeted_gvcf_gnomix_reference_vcf, checkIfExists: true)
    def gnomixConfig = file(params.targeted_gvcf_gnomix_config, checkIfExists: true)
    def scaffold = file(params.targeted_gvcf_phased_scaffold_vcf, checkIfExists: true)
    def metadata = file(params.targeted_gvcf_metadata, checkIfExists: true)
    def referenceFasta = file(params.targeted_gvcf_reference_fasta, checkIfExists: true)
    def referenceFai = file(params.targeted_gvcf_reference_fai, checkIfExists: true)
    def preregistration = file("${repoDir}/conf/m27c_targeted_gvcf_preregistration.json", checkIfExists: true)
    def auditPy = file("${repoDir}/bin/audit_targeted_gvcf_readiness.py", checkIfExists: true)
    def corePy = file("${repoDir}/bin/m27c_gvcf_core.py", checkIfExists: true)
    def bridgePy = file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)
    def benchmarkPy = file("${repoDir}/bin/benchmark_targeted_gvcf_access.py", checkIfExists: true)
    def headerAuditPy = file("${repoDir}/bin/audit_gvcf_header_contract.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.m16_5_analysis_container_image,
        container_sha256 : params.m16_5_analysis_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'Read-only chr22 targeted gVCF readiness audit; no PC-Relate, Gnomix, simulation, training or TEST',
        compute_region   : params.cloud_region,
        gvcf_glob        : params.targeted_gvcf_gvcf_glob,
        input_manifest   : params.targeted_gvcf_input_manifest,
        phased_scaffold  : params.targeted_gvcf_phased_scaffold_vcf,
        gnomix_reference : params.targeted_gvcf_gnomix_reference_vcf,
        gnomix_config    : params.targeted_gvcf_gnomix_config,
        reference_fasta  : params.targeted_gvcf_reference_fasta,
        readers          : params.targeted_gvcf_readers,
        resource_screen  : params.targeted_gvcf_smoke_only,
        header_only      : params.targeted_gvcf_header_only,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    WRITE_TARGETED_GVCF_RUN_PROVENANCE(channel.value(runProvenanceB64))

    if( params.targeted_gvcf_smoke_only ) {
        def smokeGvcfs = gvcfs.map { paths -> paths.take(params.targeted_gvcf_smoke_samples as int) }
        def smokeIndexes = indexes.map { paths -> paths.take(params.targeted_gvcf_smoke_samples as int) }
        BENCHMARK_TARGETED_GVCF_ACCESS(
            smokeGvcfs,
            smokeIndexes,
            channel.value(gnomixReference),
            channel.value(benchmarkPy),
        )
    } else if( params.targeted_gvcf_header_only ) {
        def inputManifest = file(params.targeted_gvcf_input_manifest, checkIfExists: true)
        AUDIT_TARGETED_GVCF_HEADERS(
            gvcfs,
            channel.value(inputManifest),
            channel.value(headerAuditPy),
            channel.value(corePy),
        )
    } else {
        def inputManifest = file(params.targeted_gvcf_input_manifest, checkIfExists: true)
        AUDIT_TARGETED_GVCF_READINESS(
            gvcfs,
            indexes,
            channel.value(inputManifest),
            channel.value(scaffold),
            channel.value(gnomixReference),
            channel.value(gnomixConfig),
            channel.value(metadata),
            channel.value(referenceFasta),
            channel.value(referenceFai),
            channel.value(preregistration),
            channel.value(auditPy),
            channel.value(corePy),
            channel.value(bridgePy),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
    }
}
