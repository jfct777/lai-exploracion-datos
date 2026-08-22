nextflow.enable.dsl=2

include { M33_A0_AUTHENTICATE_SOURCES; M33_A0_VALIDATE_INDEXES; M33_A0_AUDIT_LEGACY_ROOT } from '../modules/33_A0_REAL_ADAPTER'

workflow {
    def required = [
        'run_id', 'git_commit', 'root_label', 'root_seed', 'tree_sequence', 'pools',
        'rare_catalog', 'rare_haplotypes', 'm31_sites', 'm31_target',
        'ref_vcf', 'ref_tbi', 'target_vcf', 'target_tbi', 'ref_pairs', 'panel_map', 'flare_anc', 'genetic_map'
    ]
    def missing = required.findAll { !params["m33_a0_${it}"] }
    if (missing) error "M33 A0 requires explicit parameters: ${missing.join(', ')}"
    if (!(params.m33_a0_git_commit ==~ /[0-9a-f]{40}/)) error 'M33 A0 requires an exact Git commit'
    def allowedTechnicalRoots = [root17: 20260817, root18: 20260818]
    if (allowedTechnicalRoots[params.m33_a0_root_label] != (params.m33_a0_root_seed as int)) {
        error 'A0 requires an exact registered technical root label/seed pair'
    }

    def repo = projectDir.resolve('..')
    def sourceAuthPy = file("${repo}/bin/m33_a0_source_auth.py", checkIfExists:true)
    def adapterPy = file("${repo}/bin/m33_a0_real_adapter.py", checkIfExists:true)
    def tabixAuditPy = file("${repo}/bin/m33_a0_tabix_audit.py", checkIfExists:true)
    def orderedLinearPy = file("${repo}/bin/m31_ordered_linear.py", checkIfExists:true)
    def rarePreflightPy = file("${repo}/bin/m31_ordered_rare_preflight.py", checkIfExists:true)
    def registry = file(params.m33_a0_asset_registry, checkIfExists:true)
    def prereg = file(params.m33_a0_preregistration, checkIfExists:true)
    def configNf = file("${repo}/conf/m33_a0_real_adapter.config", checkIfExists:true)
    def moduleNf = file("${repo}/modules/33_A0_REAL_ADAPTER.nf", checkIfExists:true)
    def workflowNf = file("${repo}/workflows/m33_a0_real_adapter.nf", checkIfExists:true)
    def adapterTest = file("${repo}/tests/test_m33_a0_real_adapter.py", checkIfExists:true)
    def nextflowTest = file("${repo}/tests/test_m33_a0_real_adapter_nextflow.py", checkIfExists:true)

    M33_A0_AUTHENTICATE_SOURCES(
        sourceAuthPy, adapterPy, tabixAuditPy, orderedLinearPy, rarePreflightPy, registry, configNf,
        prereg, moduleNf, workflowNf, adapterTest, nextflowTest,
        params.m33_a0_git_commit, repo.toString()
    )

    def root = Channel.of(tuple(
        params.m33_a0_root_label, params.m33_a0_root_seed as int,
        file(params.m33_a0_tree_sequence, checkIfExists:true),
        file(params.m33_a0_pools, checkIfExists:true),
        file(params.m33_a0_rare_catalog, checkIfExists:true),
        file(params.m33_a0_rare_haplotypes, checkIfExists:true),
        file(params.m33_a0_m31_sites, checkIfExists:true),
        file(params.m33_a0_m31_target, checkIfExists:true),
        file(params.m33_a0_ref_vcf, checkIfExists:true),
        file(params.m33_a0_ref_tbi, checkIfExists:true),
        file(params.m33_a0_target_vcf, checkIfExists:true),
        file(params.m33_a0_target_tbi, checkIfExists:true),
        file(params.m33_a0_ref_pairs, checkIfExists:true),
        file(params.m33_a0_panel_map, checkIfExists:true),
        file(params.m33_a0_flare_anc, checkIfExists:true),
        file(params.m33_a0_genetic_map, checkIfExists:true)
    ))
    def indexRoot = Channel.of(tuple(
        params.m33_a0_root_label,
        file(params.m33_a0_ref_vcf, checkIfExists:true),
        file(params.m33_a0_ref_tbi, checkIfExists:true),
        file(params.m33_a0_target_vcf, checkIfExists:true),
        file(params.m33_a0_target_tbi, checkIfExists:true)
    ))
    M33_A0_VALIDATE_INDEXES(
        indexRoot, tabixAuditPy, M33_A0_AUTHENTICATE_SOURCES.out.auth,
        params.m33_a0_git_commit, params.m33_a0_tabix_container_image_id
    )
    M33_A0_AUDIT_LEGACY_ROOT(
        root, prereg, registry, M33_A0_AUTHENTICATE_SOURCES.out.auth,
        M33_A0_VALIDATE_INDEXES.out.audit,
        adapterPy, sourceAuthPy, tabixAuditPy, orderedLinearPy, rarePreflightPy,
        configNf, moduleNf, workflowNf, adapterTest, nextflowTest,
        params.m33_a0_git_commit,
        workflow.nextflow.version.toString(), params.m33_a0_container_image_id
    )
}
