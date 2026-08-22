nextflow.enable.dsl=2

include { M33_TABIX_BUILD_ONE as M33_TABIX_BUILD_A } from '../modules/33_TABIX_KAT'
include { M33_TABIX_BUILD_ONE as M33_TABIX_BUILD_B } from '../modules/33_TABIX_KAT'
include { M33_TABIX_COMPARE } from '../modules/33_TABIX_KAT'
include { M33_TABIX_PUBLISH } from '../modules/33_TABIX_KAT'

workflow {
    kat_script = file("${baseDir}/../bin/m33_tabix_kat.py", checkIfExists: true)
    publisher = file("${baseDir}/../bin/m33_gcs_append_only.py", checkIfExists: true)
    storage_validator = file("${baseDir}/../bin/m33_storage_policy.py", checkIfExists: true)
    storage_policy = file("${baseDir}/../conf/m33_storage_namespace_policy.json", checkIfExists: true)
    controller_receipt = file(System.getenv('DNABR_M33_CONTROLLER_RECEIPT'), checkIfExists: true)
    M33_TABIX_BUILD_A(kat_script, 'A')
    M33_TABIX_BUILD_B(kat_script, 'B')
    M33_TABIX_COMPARE(M33_TABIX_BUILD_A.out, M33_TABIX_BUILD_B.out, kat_script)
    M33_TABIX_PUBLISH(
        M33_TABIX_COMPARE.out[0], controller_receipt, storage_policy,
        storage_validator, publisher
    )
}
