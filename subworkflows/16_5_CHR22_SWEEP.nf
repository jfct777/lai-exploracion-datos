nextflow.enable.dsl=2

include { PREPARE_M165_CHR22_SWEEP; RUN_M165_CHR22_SWEEP } from '../modules/16_5_CHR22_SWEEP'

workflow M165_CHR22_SWEEP {
    take:
    pair_summary
    configuration_summary
    sample_ids
    settings

    main:
    PREPARE_M165_CHR22_SWEEP(pair_summary, configuration_summary, sample_ids, settings,
        file("${projectDir}/bin/m165_chr22_sweep.py"))
    configurations = PREPARE_M165_CHR22_SWEEP.out.configurations.flatten()
        .map { directory -> tuple(directory.name, directory) }
    RUN_M165_CHR22_SWEEP(configurations,
        file("${projectDir}/bin/m165_chr22_sweep.py"),
        file("${projectDir}/bin/ibd_community_enhanced.py"))

    emit:
    preparation = PREPARE_M165_CHR22_SWEEP.out.receipt
    results = RUN_M165_CHR22_SWEEP.out.results
}
