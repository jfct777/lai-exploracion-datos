nextflow.enable.dsl=2

include {
    M33_I0_STAGE_REAL_SOURCE
    M33_I0_DERIVE_REAL_INDEX
    M33_I0_AGGREGATE_REAL
} from '../modules/33_I0_REAL'

workflow {
    if( params.m33_i0_real_enabled?.toString() != 'true' ) {
        throw new IllegalStateException('M33 I0 real permanece bloqueado sin --m33_i0_real_enabled true.')
    }
    if( !params.m33_i0_real_run_id ||
        !(params.m33_i0_real_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/) ) {
        throw new IllegalStateException('m33_i0_real_run_id no es válido.')
    }
    if( workflow.resume ) {
        throw new IllegalStateException('M33 I0 real no admite -resume; cada build A/B debe ser fresco.')
    }
    local_results = params.m33_i0_real_local_results ? new File(params.m33_i0_real_local_results.toString()) : null
    if( local_results == null ||
        !local_results.canonicalPath.startsWith('/tmp/') ||
        local_results.path.contains('..') ||
        local_results.exists() ) {
        throw new IllegalStateException('m33_i0_real_local_results debe ser un directorio absoluto nuevo bajo /tmp/.')
    }
    if( params.m33_i0_real_safe_bridge?.toString() != 'false' ||
        params.m33_i0_real_materialize?.toString() != 'false' ||
        params.m33_i0_real_training?.toString() != 'false' ||
        params.m33_i0_real_truth?.toString() != 'false' ||
        params.m33_i0_real_test?.toString() != 'false' ||
        params.m33_i0_real_ready?.toString() != 'false' ) {
        throw new IllegalStateException('I0 real no puede abrir etapas posteriores, truth, TEST ni READY.')
    }
    expected_image = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54'
    if( params.m33_i0_real_tabix_image != expected_image ) {
        throw new IllegalStateException('La imagen Tabix efectiva difiere del digest autorizado.')
    }

    real_script = file("${baseDir}/../bin/m33_i0_real.py", checkIfExists: true)
    helper_script = file("${baseDir}/../bin/m33_i0_index.py", checkIfExists: true)
    authorization = file(params.m33_i0_real_authorization, checkIfExists: true)
    contract = file(params.m33_i0_real_contract, checkIfExists: true)
    source_auth = file(params.m33_i0_real_source_auth, checkIfExists: true)
    repo_root = file("${baseDir}/..", checkIfExists: true).toString()
    roots = Channel.of('root17', 'root18')

    M33_I0_STAGE_REAL_SOURCE(
        roots,
        real_script,
        helper_script,
        authorization,
        contract,
        source_auth,
        repo_root,
    )
    M33_I0_DERIVE_REAL_INDEX(
        M33_I0_STAGE_REAL_SOURCE.out,
        real_script,
        helper_script,
        authorization,
        contract,
        source_auth,
    )
    root_sources = M33_I0_DERIVE_REAL_INDEX.out.map { root, source, tbi, receipt, marker -> source }.collect()
    root_receipts = M33_I0_DERIVE_REAL_INDEX.out.map { root, source, tbi, receipt, marker -> receipt }.collect()
    root_markers = M33_I0_DERIVE_REAL_INDEX.out.map { root, source, tbi, receipt, marker -> marker }.collect()
    root_indexes = M33_I0_DERIVE_REAL_INDEX.out.map { root, source, tbi, receipt, marker -> tbi }.collect()
    M33_I0_AGGREGATE_REAL(
        root_receipts,
        root_markers,
        root_sources,
        root_indexes,
        real_script,
        helper_script,
        authorization,
        contract,
        source_auth,
    )
}
