nextflow.enable.dsl=2

process M38_F_MINUS_S660_BGZIP_TABIX {
    tag { "m38_f_minus_s660_index_${split}_${vcfRole}" }
    publishDir {
        "${params.m38_fminus_results_dir}/${params.m38_fminus_run_id}/${split.toLowerCase()}/indexed/${vcfRole.toLowerCase()}"
    }, mode: 'copy', overwrite: false
    container params.m38_fminus_tabix_image
    containerOptions { "--network none --user ${params.m38_fminus_container_user}" }
    cpus { params.m38_fminus_index_cpus }
    memory { params.m38_fminus_index_memory }
    time { params.m38_fminus_index_time }
    maxForks params.m38_fminus_index_max_forks

    input:
    tuple val(split), val(vcfRole), path(vcf), path(filterReceipt)

    output:
    tuple val(split), val(vcfRole),
          path("${vcf}.gz"), path("${vcf}.gz.tbi"),
          emit: indexed

    script:
    """
    set -euo pipefail
    test ! -e '${vcf}.gz'
    bgzip --threads 1 --stdout '${vcf}' > '${vcf}.gz'
    test -s '${vcf}.gz'
    test ! -e '${vcf}.gz.tbi'
    tabix -p vcf '${vcf}.gz'
    test -s '${vcf}.gz.tbi'
    python3 -c 'import json,subprocess,sys; receipt=json.load(open(sys.argv[1], encoding="utf-8")); expected=receipt["identity"]["source_contig_label"]; observed=subprocess.check_output(["tabix", "-l", sys.argv[2]], text=True).splitlines(); assert observed == [expected], (expected, observed)' '${filterReceipt}' '${vcf}.gz'
    """

    stub:
    """
    set -euo pipefail
    touch '${vcf}.gz' '${vcf}.gz.tbi'
    """
}
