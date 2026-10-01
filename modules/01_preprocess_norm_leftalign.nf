nextflow.enable.dsl=2

process PREPROCESS_NORM_LEFTALIGN {
    tag "chr${chr}"

    publishDir "${params.outdir}/01_norm", mode: 'copy', overwrite: false

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(ref_fasta)
    path mark_original_alleles_py

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.norm.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.norm.vcf.gz.tbi")
    path "dnabr.hg38.2723.chr${chr}.norm.log"

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.preprocess_norm_leftalign?.threads ?: 2
    def largeRoot = params.preprocess_large_temp_dir ?: ''
    if (largeRoot && (!largeRoot.startsWith('/') || largeRoot.contains('\n') || largeRoot.contains('\r'))) {
        throw new IllegalArgumentException('preprocess_large_temp_dir must be an absolute filesystem directory')
    }
    def quotedRoot = "'" + largeRoot.replace("'", "'\"'\"'") + "'"
    // Only newly created sequential bulk outputs use this directory. Nextflow
    // work/cache/logs stay local; callers must disable copy publication here.
    def bulkStart = largeRoot ? """
    local_task_dir=\$PWD
    input_vcf=\$(readlink -f '${vcf_gz}')
    input_ref=\$(readlink -f '${ref_fasta}')
    marker_script=\$(readlink -f '${mark_original_alleles_py}')
    test -f "\${input_ref}.fai" || { echo 'Indexed FASTA is required in bounded-disk mode' >&2; exit 1; }
    test -d ${quotedRoot}
    bulk_dir=\$(mktemp -d ${quotedRoot}/m01_chr${chr}.XXXXXXXX)
    printf '%s\\n' "\$bulk_dir" > '${sample_id}.m01.large_temp_dir.txt'
    cd "\$bulk_dir"
    """ : """
    input_vcf='${vcf_gz}'
    input_ref='${ref_fasta}'
    marker_script='${mark_original_alleles_py}'
    test -f "\${input_ref}.fai" || samtools faidx "\$input_ref"
    """
    def bulkFinish = largeRoot ? """
    cd "\$local_task_dir"
    ln -s "\$bulk_dir/${sample_id}.norm.vcf.gz" '${sample_id}.norm.vcf.gz'
    ln -s "\$bulk_dir/${sample_id}.norm.vcf.gz.tbi" '${sample_id}.norm.vcf.gz.tbi'
    cp "\$bulk_dir/${sample_id}.norm.log" '${sample_id}.norm.log'
    """ : ''
    """
    set -euo pipefail

    ${bulkStart}

    python3 "\$marker_script" --input "\$input_vcf" --output ${sample_id}.original.bcf
    bcftools norm -m -any -f "\$input_ref" --threads ${threads} -Oz -o ${sample_id}.norm.vcf.gz ${sample_id}.original.bcf 2> ${sample_id}.norm.log

    bcftools index --threads ${threads} -t ${sample_id}.norm.vcf.gz
    ${bulkFinish}
    """
}
