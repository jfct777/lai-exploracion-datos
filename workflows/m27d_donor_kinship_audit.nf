nextflow.enable.dsl=2

include {
    WRITE_DONOR_KINSHIP_RUN_PROVENANCE;
    PREPARE_DONOR_KINSHIP_RESOURCES;
    BENCHMARK_DONOR_KINSHIP_RESOURCES
} from '../modules/27D_DONOR_KINSHIP_AUDIT'

workflow {
    if( !params.donor_kinship_smoke_only ) {
        throw new IllegalStateException(
            'M27D full donor audit is not implemented or authorized. Use --donor_kinship_smoke_only true.'
        )
    }

    def phase = params.donor_kinship_phase?.toString()
    if( !(phase in ['prepare', 'benchmark']) ) {
        throw new IllegalStateException('M27D phase must be prepare or benchmark.')
    }

    def repoDir = projectDir.resolve('..')
    def preregistration = file(
        "${repoDir}/conf/m27d_donor_kinship_preregistration.json",
        checkIfExists: true,
    )
    def pcrelateSmokeR = file("${repoDir}/bin/m27d_resource_smoke.R", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.donor_kinship_container_image,
        container_sha256 : params.donor_kinship_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : "M27D ${phase} technical phase only; PC-Relate without KING; no donor certification, Gnomix, simulation, training or TEST",
        compute_region   : params.cloud_region,
        panel_vcf_glob   : params.donor_kinship_panel_vcf_glob,
        sample_metadata  : params.donor_kinship_metadata,
        exclude_regions : params.donor_kinship_exclude_regions_bed,
        thread_grid      : params.donor_kinship_thread_grid,
        phase            : phase,
        architecture     : 'persistent marker preparation followed by reusable PC-Relate benchmark',
        full_run_authorized: false,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    WRITE_DONOR_KINSHIP_RUN_PROVENANCE(channel.value(runProvenanceB64))
    if( phase == 'prepare' ) {
        def panelVcfs = channel
            .fromPath(params.donor_kinship_panel_vcf_glob, checkIfExists: true)
            .collect()
            .map { paths ->
                paths.sort { left, right ->
                    def leftMatch = (left.getName() =~ /hg38\.(\d+)\.norm/)
                    def rightMatch = (right.getName() =~ /hg38\.(\d+)\.norm/)
                    if( !leftMatch.find() || !rightMatch.find() ) {
                        throw new IllegalStateException('M27D could not parse chromosome from panel VCF name.')
                    }
                    (leftMatch.group(1) as int) <=> (rightMatch.group(1) as int)
                }
            }
        def metadata = file(params.donor_kinship_metadata, checkIfExists: true)
        def excludeBed = file(params.donor_kinship_exclude_regions_bed, checkIfExists: true)
        def sampleStrataPy = file("${repoDir}/bin/m27d_prepare_sample_strata.py", checkIfExists: true)
        def prepareResourcesR = file("${repoDir}/bin/m27d_prepare_genotype_resources.R", checkIfExists: true)
        def bridgePy = file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)
        PREPARE_DONOR_KINSHIP_RESOURCES(
            panelVcfs,
            channel.value(metadata),
            channel.value(excludeBed),
            channel.value(preregistration),
            channel.value(sampleStrataPy),
            channel.value(prepareResourcesR),
            channel.value(bridgePy),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
    } else {
        def requiredPreparedParams = [
            donor_kinship_prepared_gds: params.donor_kinship_prepared_gds,
            donor_kinship_prepared_anchor_rds: params.donor_kinship_prepared_anchor_rds,
            donor_kinship_prepared_strict_rds: params.donor_kinship_prepared_strict_rds,
            donor_kinship_prepared_strata: params.donor_kinship_prepared_strata,
            donor_kinship_preparation_manifest: params.donor_kinship_preparation_manifest,
        ]
        def missingPrepared = requiredPreparedParams.findAll { key, value -> !value }.keySet()
        if( missingPrepared ) {
            throw new IllegalStateException("M27D benchmark is missing prepared inputs: ${missingPrepared.join(', ')}")
        }
        BENCHMARK_DONOR_KINSHIP_RESOURCES(
            channel.value(file(params.donor_kinship_prepared_gds, checkIfExists: true)),
            channel.value(file(params.donor_kinship_prepared_anchor_rds, checkIfExists: true)),
            channel.value(file(params.donor_kinship_prepared_strict_rds, checkIfExists: true)),
            channel.value(file(params.donor_kinship_prepared_strata, checkIfExists: true)),
            channel.value(file(params.donor_kinship_preparation_manifest, checkIfExists: true)),
            channel.value(preregistration),
            channel.value(pcrelateSmokeR),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
    }
}
