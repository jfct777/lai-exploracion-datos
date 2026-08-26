nextflow.enable.dsl=2

process M34_NAM_TABIX_INDEX {
    tag { "m34_tabix_${split}_${vcfRole}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/indexed/${vcfRole.toLowerCase()}"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_tabix_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_tabix_cpus }
    memory { params.m34_inputs_tabix_memory }
    time { params.m34_inputs_tabix_time }
    maxForks params.m34_inputs_tabix_max_forks

    input:
    tuple val(split), val(vcfRole), path(vcf)

    output:
    tuple val(split), val(vcfRole), path(vcf), path("${vcf}.tbi"), emit: indexed

    script:
    """
    set -euo pipefail
    test ! -e ${vcf}.tbi
    tabix -p vcf ${vcf}
    test -s ${vcf}.tbi
    tabix -l ${vcf} | grep -q .
    """

    stub:
    """
    set -euo pipefail
    touch ${vcf}.tbi
    """
}
