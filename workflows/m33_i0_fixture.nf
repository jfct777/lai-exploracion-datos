nextflow.enable.dsl=2

include { M33_I0_MAKE_FIXTURE; M33_I0_DERIVE_FIXTURE_INDEX } from '../modules/33_I0_FIXTURE'

workflow {
    if( params.m33_i0_fixture_enabled?.toString() != 'true' ) {
        throw new IllegalStateException('M33 I0 fixture permanece bloqueado sin --m33_i0_fixture_enabled true.')
    }
    if( !params.m33_i0_fixture_run_id ||
        !(params.m33_i0_fixture_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/) ) {
        throw new IllegalStateException('m33_i0_fixture_run_id no es válido.')
    }
    if( params.m33_i0_fixture_real_asset_read?.toString() != 'false' ||
        params.m33_i0_fixture_safe_bridge?.toString() != 'false' ||
        params.m33_i0_fixture_materialize?.toString() != 'false' ||
        params.m33_i0_fixture_training?.toString() != 'false' ) {
        throw new IllegalStateException('I0 fixture no puede abrir datos reales ni etapas posteriores.')
    }
    i0_script = file("${baseDir}/../bin/m33_i0_index.py", checkIfExists: true)
    authorization = file(params.m33_i0_fixture_authorization, checkIfExists: true)
    contract = file(params.m33_i0_fixture_contract, checkIfExists: true)
    source_auth = file(params.m33_i0_fixture_source_auth, checkIfExists: true)
    repo_root = file("${baseDir}/..", checkIfExists: true).toString()
    M33_I0_MAKE_FIXTURE(i0_script, authorization, contract, source_auth, repo_root)
    M33_I0_DERIVE_FIXTURE_INDEX(
        M33_I0_MAKE_FIXTURE.out,
        i0_script,
        authorization,
        contract,
        source_auth,
        repo_root,
    )
}
