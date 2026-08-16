nextflow.enable.dsl=2

include {
    WRITE_IBD_RARE_TRANSFER_RUN_PROVENANCE;
    AUDIT_IBD_RARE_TRANSFER_FEASIBILITY
} from '../modules/27E_IBD_RARE_TRANSFER_FEASIBILITY'

def sortedAutosomes(pattern, description) {
    channel
        .fromPath(pattern, checkIfExists: true)
        .collect()
        .map { paths ->
            def numbered = paths.collect { path ->
                def matcher = (path.getName() =~ /(?:^|[._])(?:chr_?)(\d{1,2})(?:[._]|$)/)
                if( !matcher.find() ) {
                    throw new IllegalStateException("M27E could not parse a chromosome from ${description}: ${path.getName()}")
                }
                [(matcher.group(1) as int), path]
            }
            def chromosomes = numbered.collect { it[0] }
            if( chromosomes.toSorted() != (1..22).toList() ) {
                throw new IllegalStateException("M27E expects exactly autosomes 1-22 for ${description}; found ${chromosomes.toSorted()}")
            }
            numbered.toSorted { left, right -> left[0] <=> right[0] }.collect { it[1] }
        }
}

workflow {
    def repoDir = projectDir.resolve('..')
    def ibdFiles = sortedAutosomes(params.ibd_rare_transfer_ibd_glob, 'Refined-IBD files')
    def ibdLogs = sortedAutosomes(params.ibd_rare_transfer_ibd_log_glob, 'Refined-IBD logs')
    def maps = sortedAutosomes(params.ibd_rare_transfer_map_glob, 'genetic maps')
    def rawWgsVcf = file(params.ibd_rare_transfer_raw_wgs_vcf, checkIfExists: true)
    def phasedPanelVcf = file(params.ibd_rare_transfer_phased_panel_vcf, checkIfExists: true)
    def gnomixReferenceVcf = file(params.ibd_rare_transfer_gnomix_reference_vcf, checkIfExists: true)
    def resolvedStrata = file(params.ibd_rare_transfer_resolved_strata, checkIfExists: true)
    def resolvedStrataManifest = file(params.ibd_rare_transfer_resolved_strata_manifest, checkIfExists: true)
    def preregistration = file(
        "${repoDir}/conf/m27e_ibd_rare_transfer_preregistration.json",
        checkIfExists: true,
    )
    def auditPy = file("${repoDir}/bin/audit_m27e_ibd_rare_transfer.py", checkIfExists: true)
    def bridgePy = file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.ibd_rare_transfer_container_image,
        container_sha256 : params.ibd_rare_transfer_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'Read-only autosomal IBD and chr22 rare-transfer feasibility; no new relatedness inference, simulation, LAI, training or TEST access',
        ibd_glob         : params.ibd_rare_transfer_ibd_glob,
        ibd_log_glob     : params.ibd_rare_transfer_ibd_log_glob,
        genetic_map_glob : params.ibd_rare_transfer_map_glob,
        raw_wgs_vcf      : params.ibd_rare_transfer_raw_wgs_vcf,
        phased_panel_vcf : params.ibd_rare_transfer_phased_panel_vcf,
        gnomix_reference : params.ibd_rare_transfer_gnomix_reference_vcf,
        resolved_strata  : params.ibd_rare_transfer_resolved_strata,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    WRITE_IBD_RARE_TRANSFER_RUN_PROVENANCE(channel.value(runProvenanceB64))
    AUDIT_IBD_RARE_TRANSFER_FEASIBILITY(
        ibdFiles,
        ibdLogs,
        maps,
        channel.value(rawWgsVcf),
        channel.value(phasedPanelVcf),
        channel.value(gnomixReferenceVcf),
        channel.value(resolvedStrata),
        channel.value(resolvedStrataManifest),
        channel.value(preregistration),
        channel.value(auditPy),
        channel.value(bridgePy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
}
