nextflow.enable.dsl=2

// Opt-in M01 v2: the original-allele BCF is a declared Nextflow checkpoint.
// The legacy M01 process remains unchanged for existing caches and frozen runs.
def checkpointShellQuote(value) {
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

def checkpointBulkRoot(value) {
    def root = value ? value.toString() : ''
    if (root && (!root.startsWith('/') || root.contains('\n') || root.contains('\r'))) {
        throw new IllegalArgumentException('preprocess_large_temp_dir must be an absolute filesystem directory')
    }
    return root
}

process ANNOTATE_ORIGINAL_ALLELES {
    tag "chr${chr}"

    // One CPU preserves the old serial algorithm. A caller can reserve more
    // CPUs for the annotator's bounded BGZF input/output worker allocation.
    cpus { params.resources?.preprocess_original_alleles?.cpus ?: 1 }
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi)
    path mark_original_alleles_py
    path preprocess_storage_guard_py

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.original.bcf"), emit: annotated
    path "dnabr.hg38.2723.chr${chr}.annotation.log", emit: annotation_log
    path "dnabr.hg38.2723.chr${chr}.annotation.storage.json", emit: storage_receipt

    script:
    def base = "dnabr.hg38.2723.chr${chr}"
    def bulkRoot = checkpointBulkRoot(params.preprocess_large_temp_dir)
    def minimumFreeGiB = params.preprocess_minimum_free_gib == null ? 12 : params.preprocess_minimum_free_gib
    // The runner supplies an initial capacity estimate from the authenticated
    // raw size. It is an admission estimate, not a bound on all later outputs.
    def requiredFreeBytes = params.preprocess_required_free_bytes ?: 0
    def bulkSetup = bulkRoot ? """
    test -d ${checkpointShellQuote(bulkRoot)}
    python3 "\$storage_guard" --directory ${checkpointShellQuote(bulkRoot)} --minimum-free-gib ${checkpointShellQuote(minimumFreeGiB)} --required-free-bytes ${checkpointShellQuote(requiredFreeBytes)} > '${base}.annotation.storage.json'
    bulk_dir=\$(mktemp -d ${checkpointShellQuote(bulkRoot)}/m01_chr${chr}.XXXXXXXX)
    printf '%s\\n' "\$bulk_dir" > '${base}.m01.large_temp_dir.txt'
    cd "\$bulk_dir"
    """ : """
    python3 "\$storage_guard" --directory "\$PWD" --minimum-free-gib ${checkpointShellQuote(minimumFreeGiB)} --required-free-bytes ${checkpointShellQuote(requiredFreeBytes)} > '${base}.annotation.storage.json'
    """
    def bulkFinish = bulkRoot ? """
    cd "\$local_task_dir"
    ln -s "\$bulk_dir/${base}.original.bcf" '${base}.original.bcf'
    """ : ''
    """
    set -euo pipefail
    local_task_dir=\$PWD
    input_vcf=\$(readlink -f ${checkpointShellQuote(vcf_gz)})
    marker_script=\$(readlink -f ${checkpointShellQuote(mark_original_alleles_py)})
    storage_guard=\$(readlink -f ${checkpointShellQuote(preprocess_storage_guard_py)})
    ${bulkSetup}
    python3 "\$marker_script" --input "\$input_vcf" --output '${base}.original.bcf' --threads ${task.cpus} | tee "\$local_task_dir/${base}.annotation.log"
    ${bulkFinish}
    """
}

process NORMALIZE_ANNOTATED_ALLELES {
    tag "chr${chr}"

    // In bounded-disk production, outputs stay in the declared local bulk
    // directory; the outer runner publishes verified final products once.
    publishDir "${params.outdir}/01_norm", mode: 'copy', overwrite: false,
               enabled: !params.preprocess_large_temp_dir

    cpus { params.resources?.preprocess_norm_leftalign?.cpus ?: params.cpus }
    memory { params.resources?.preprocess_norm_leftalign?.memory ?: params.memory }
    time params.time

    input:
    tuple val(chr), path(original_bcf), path(ref_fasta)
    path preprocess_storage_guard_py

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.norm.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.norm.vcf.gz.tbi"), emit: norm
    path "dnabr.hg38.2723.chr${chr}.norm.log", emit: norm_log
    path "dnabr.hg38.2723.chr${chr}.normalization.storage.json", emit: storage_receipt

    script:
    def base = "dnabr.hg38.2723.chr${chr}"
    // bcftools --threads are ADDITIONAL compression threads; retain one CPU
    // for the main thread. Zero is valid for a one-CPU process.
    def requested = params.resources?.preprocess_norm_leftalign?.threads
    def maximum = Math.max(0, (task.cpus as int) - 1)
    def threads = requested == null ? maximum : Math.min(maximum, requested as int)
    if (threads < 0) throw new IllegalArgumentException('Normalization threads must be nonnegative')
    def bulkRoot = checkpointBulkRoot(params.preprocess_large_temp_dir)
    // Annotation has already occupied part of the initial capacity estimate.
    // Do not demand that full estimate twice; enforce the remaining free-space
    // floor here and retain the runner's ongoing low-space checks.
    def minimumFreeGiB = params.preprocess_minimum_free_gib == null ? 12 : params.preprocess_minimum_free_gib
    def bulkSetup = bulkRoot ? """
    test -d ${checkpointShellQuote(bulkRoot)}
    python3 "\$storage_guard" --directory ${checkpointShellQuote(bulkRoot)} --minimum-free-gib ${checkpointShellQuote(minimumFreeGiB)} > '${base}.normalization.storage.json'
    bulk_dir=\$(mktemp -d ${checkpointShellQuote(bulkRoot)}/m01_chr${chr}.XXXXXXXX)
    printf '%s\\n' "\$bulk_dir" > '${base}.m01.large_temp_dir.txt'
    cd "\$bulk_dir"
    """ : """
    python3 "\$storage_guard" --directory "\$PWD" --minimum-free-gib ${checkpointShellQuote(minimumFreeGiB)} > '${base}.normalization.storage.json'
    """
    def bulkFinish = bulkRoot ? """
    cd "\$local_task_dir"
    ln -s "\$bulk_dir/${base}.norm.vcf.gz" '${base}.norm.vcf.gz'
    ln -s "\$bulk_dir/${base}.norm.vcf.gz.tbi" '${base}.norm.vcf.gz.tbi'
    cp "\$bulk_dir/${base}.norm.log" '${base}.norm.log'
    """ : ''
    """
    set -euo pipefail
    local_task_dir=\$PWD
    input_bcf=\$(readlink -f ${checkpointShellQuote(original_bcf)})
    input_ref=\$(readlink -f ${checkpointShellQuote(ref_fasta)})
    storage_guard=\$(readlink -f ${checkpointShellQuote(preprocess_storage_guard_py)})
    test -f "\${input_ref}.fai" || { echo 'Indexed FASTA is required for checkpointed M01' >&2; exit 1; }
    ${bulkSetup}
    bcftools norm -m -any -f "\$input_ref" --threads ${threads} -Oz -o '${base}.norm.vcf.gz' "\$input_bcf" 2> '${base}.norm.log'
    bcftools index --threads ${threads} -t '${base}.norm.vcf.gz'
    ${bulkFinish}
    """
}

workflow PREPROCESS_NORM_LEFTALIGN_CHECKPOINTED {
    take:
    raw_with_reference
    mark_original_alleles_py
    preprocess_storage_guard_py

    main:
    raw = raw_with_reference.map { chr, vcf, tbi, reference -> tuple(chr, vcf, tbi) }
    references = raw_with_reference.map { chr, vcf, tbi, reference -> tuple(chr, reference) }
    ANNOTATE_ORIGINAL_ALLELES(raw, mark_original_alleles_py, preprocess_storage_guard_py)
    NORMALIZE_ANNOTATED_ALLELES(ANNOTATE_ORIGINAL_ALLELES.out.annotated.join(references), preprocess_storage_guard_py)

    emit:
    norm = NORMALIZE_ANNOTATED_ALLELES.out.norm
    norm_log = NORMALIZE_ANNOTATED_ALLELES.out.norm_log
}
