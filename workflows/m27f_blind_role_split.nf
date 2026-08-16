nextflow.enable.dsl=2

include {
    WRITE_M27F_SPLIT_RUN_PROVENANCE;
    AUDIT_M27F_BLIND_ROLE_SPLIT
} from '../modules/27F_BLIND_ROLE_SPLIT'

def sortedAutosomes(pattern, description) {
    channel.fromPath(pattern, checkIfExists: true).collect().map { paths ->
        def numbered = paths.collect { path ->
            def matcher = (path.getName() =~ /(?:^|[._])(?:chr_?)(\d{1,2})(?:[._]|$)/)
            if( !matcher.find() ) throw new IllegalStateException("Cannot parse chromosome for ${description}: ${path}")
            [(matcher.group(1) as int), path]
        }
        if( numbered.collect { it[0] }.toSorted() != (1..22).toList() ) {
            throw new IllegalStateException("M27F expects autosomes 1-22 for ${description}")
        }
        numbered.toSorted { a, b -> a[0] <=> b[0] }.collect { it[1] }
    }
}

workflow {
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        syntax_parser: System.getenv('NXF_SYNTAX_PARSER') ?: 'default',
        run_id: params.cloud_run_id,
        scientific_scope: 'Metadata and frozen IBD split only; VCF headers but no genotypes, rare support, LAI or TEST inspection',
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command: workflow.commandLine,
        launch_dir: workflow.launchDir.toString(),
        project_dir: projectDir.toString(),
        ibd_glob: params.m27f_split_ibd_glob,
        genetic_map_glob: params.m27f_split_map_glob,
        panel_vcf: params.m27f_split_panel_vcf,
        discovery_vcf: params.m27f_split_discovery_vcf,
        resolved_strata: params.m27f_split_resolved_strata,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()
    WRITE_M27F_SPLIT_RUN_PROVENANCE(channel.value(runProvenanceB64))
    AUDIT_M27F_BLIND_ROLE_SPLIT(
        sortedAutosomes(params.m27f_split_ibd_glob, 'IBD'),
        sortedAutosomes(params.m27f_split_map_glob, 'maps'),
        file(params.m27f_split_panel_vcf, checkIfExists: true),
        file(params.m27f_split_discovery_vcf, checkIfExists: true),
        file(params.m27f_split_resolved_strata, checkIfExists: true),
        file(params.m27f_split_resolved_strata_manifest, checkIfExists: true),
        file("${repoDir}/conf/m27f_split_preregistration.json", checkIfExists: true),
        file("${repoDir}/bin/audit_m27f_blind_split.py", checkIfExists: true),
        file("${repoDir}/bin/audit_m27e_ibd_rare_transfer.py", checkIfExists: true),
        file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true),
        channel.value(provenanceB64),
    )
}
