nextflow.enable.dsl=2

// L1 catalogue correspondence (default) or authenticated genotype audit.
// Neither mode performs simulation, phasing, training or biological validation.
process R02_LAI_CATALOGUE_CORRESPONDENCE {
    tag "chr${params.chromosome}_${params.panel_role}_catalogue_only"
    cache 'deep'
    cpus 1
    input:
    path source_files, stageAs: 'l1_code/*'
    path rare_vcf
    path rare_contract
    path panel_vcf
    path roles
    path build_evidence
    path contig_map
    path input_hashes
    output:
    path 'catalogue_correspondence', emit: correspondence
    script:
    def source_bin = 'l1_code'
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_lai_allele_support.py' \
      --rare-vcf '${rare_vcf}' --rare-contract '${rare_contract}' --panel-vcf '${panel_vcf}' \
      --roles '${roles}' --build-evidence '${build_evidence}' --contig-map '${contig_map}' \
      --input-hashes '${input_hashes}' --panel-role '${params.panel_role}' \
      --chromosome '${params.chromosome}' --expected-source-samples ${params.expected_source_samples} \
      --expected-rare-sites ${params.expected_rare_sites} \
      --max-panel-records ${params.max_panel_records} --max-panel-bytes ${params.max_panel_bytes} \
      --max-panel-index-bytes ${params.max_panel_index_bytes} --max-line-bytes ${params.max_line_bytes} \
      --min-free-gib ${params.min_free_gib} --resource-check-rows ${params.resource_check_rows} \
      --outdir catalogue_correspondence
    """
}

process R02_LAI_GENOTYPE_SUPPORT {
    tag "chr${params.chromosome}_${params.panel_role}_genotype_support"
    cache 'deep'
    cpus 1
    input:
    path source_files, stageAs: 'l1_code/*'
    path rare_vcf
    path rare_contract
    path panel_vcf
    path roles
    path build_evidence
    path contig_map
    path input_hashes
    path genotype_contract
    // Both audit tasks emit reference_verification.json; retain separate roles.
    path source_build_verification, stageAs: 'source_audit/*'
    path panel_build_verification, stageAs: 'panel_audit/*'
    output:
    path 'genotype_support', emit: support
    script:
    def source_bin = 'l1_code'
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_lai_allele_support.py' --mode genotype_support \\
      --rare-vcf '${rare_vcf}' --rare-contract '${rare_contract}' --panel-vcf '${panel_vcf}' \\
      --roles '${roles}' --build-evidence '${build_evidence}' --contig-map '${contig_map}' \\
      --input-hashes '${input_hashes}' --panel-role '${params.panel_role}' \\
      --genotype-contract '${genotype_contract}' \\
      --source-build-verification '${source_build_verification}' --panel-build-verification '${panel_build_verification}' \\
      --chromosome '${params.chromosome}' --expected-source-samples ${params.expected_source_samples} \\
      --expected-rare-sites ${params.expected_rare_sites} \\
      --max-panel-records ${params.max_panel_records} --max-panel-bytes ${params.max_panel_bytes} \\
      --max-panel-index-bytes ${params.max_panel_index_bytes} --max-line-bytes ${params.max_line_bytes} \\
      --min-free-gib ${params.min_free_gib} --resource-check-rows ${params.resource_check_rows} \\
      --outdir genotype_support
    """
}

workflow {
    def mode = params.mode ?: 'catalogue_only'
    if (!(mode in ['catalogue_only','genotype_support'])) error 'Invalid L1 mode'
    def paths = ['source_bin','rare_vcf','rare_contract','panel_vcf','roles','build_evidence','contig_map','input_hashes']
    if (mode == 'genotype_support') paths += ['genotype_contract','source_build_verification','panel_build_verification']
    def integers = ['expected_source_samples','expected_rare_sites','max_panel_records','max_panel_bytes',
                    'max_panel_index_bytes','max_line_bytes','resource_check_rows']
    (paths + integers + ['chromosome','panel_role','min_free_gib']).each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    integers.each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Positive integer required: ${key}"
    }
    if (!(params.chromosome.toString() ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Autosome 1-22 required'
    if (!(params.panel_role.toString() ==~ /[A-Z][A-Z0-9_]*/)) error 'Invalid role name'
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'Nonnegative free disk reserve required'
    // Directory inputs are not recursively content-hashed by all Nextflow
    // versions. Stage the exact import closure as explicit deep-cache inputs.
    def sourceFiles = ['r02_lai_allele_support.py','r02_lai_genotype_support.py',
                       'r02_lai_build_verification.py','r02_genomic_pair_evidence.py','preprocess_storage_guard.py'].collect { name ->
        file("${params.source_bin}/${name}", checkIfExists:true)
    }
    if (mode == 'catalogue_only') {
      R02_LAI_CATALOGUE_CORRESPONDENCE(
        sourceFiles,file(params.rare_vcf,checkIfExists:true),
        file(params.rare_contract,checkIfExists:true),file(params.panel_vcf,checkIfExists:true),
        file(params.roles,checkIfExists:true),file(params.build_evidence,checkIfExists:true),
        file(params.contig_map,checkIfExists:true),file(params.input_hashes,checkIfExists:true))
    } else {
      R02_LAI_GENOTYPE_SUPPORT(
        sourceFiles,file(params.rare_vcf,checkIfExists:true),
        file(params.rare_contract,checkIfExists:true),file(params.panel_vcf,checkIfExists:true),
        file(params.roles,checkIfExists:true),file(params.build_evidence,checkIfExists:true),
        file(params.contig_map,checkIfExists:true),file(params.input_hashes,checkIfExists:true),
        file(params.genotype_contract,checkIfExists:true),file(params.source_build_verification,checkIfExists:true),
        file(params.panel_build_verification,checkIfExists:true))
    }
}
