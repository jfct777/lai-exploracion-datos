nextflow.enable.dsl=2

params.request = null
params.request_sha256 = null
params.implementation = null
params.output_dir = null
params.container = null
params.compare_max_forks = 2
params.stage_time = '2h'
params.compare_time = '30m'

def required(name, value) {
    if (value == null || value.toString().trim() == '') {
        error "Missing required parameter --${name}"
    }
    value
}

process STAGE_COMMON_VIEW_ARCHIVE {
    tag 'generation-pinned-common-archive'
    cpus 1
    memory '1 GB'
    time params.stage_time
    cache 'deep'

    input:
    tuple path(request), val(request_sha256), path(implementation)

    output:
    tuple path('staged/common-views.tar'), path('staged/archive-index.json')

    script:
    """
    python3 ${implementation} \
      --stage stage \
      --request ${request} \
      --request-sha256 ${request_sha256} \
      --output staged
    """
}

process DEDUPLICATE_COMMON_VIEW {
    tag "chr${chrom}"
    cpus 1
    memory '4 GB'
    time params.compare_time
    maxForks params.compare_max_forks as int
    container params.container
    cache 'deep'
    publishDir "${params.output_dir}/views", mode: 'copy', overwrite: false

    input:
    tuple val(chrom), path(archive), path(index), path(request), val(request_sha256), path(implementation)

    output:
    tuple val(chrom), path("common_chr${chrom}")

    script:
    """
    python3 ${implementation} \
      --stage dedupe \
      --request ${request} \
      --request-sha256 ${request_sha256} \
      --archive ${archive} \
      --index ${index} \
      --chromosome ${chrom} \
      --output common_chr${chrom}
    """
}

process CATALOGUE_COMMON_VIEWS {
    tag 'all-22-equivalent-common-views'
    cpus 1
    memory '1 GB'
    time '10m'
    container params.container
    cache 'deep'
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple path(directories), path(request), val(request_sha256), path(implementation)

    output:
    path 'catalogue'

    script:
    """
    python3 ${implementation} \
      --stage aggregate \
      --request ${request} \
      --request-sha256 ${request_sha256} \
      --directories ${directories.join(' ')} \
      --output catalogue
    """
}

workflow {
    request_file = file(required('request', params.request), checkIfExists: true)
    implementation_file = file(required('implementation', params.implementation), checkIfExists: true)
    request_sha = required('request_sha256', params.request_sha256)
    required('output_dir', params.output_dir)
    container_ref = required('container', params.container)
    if (!container_ref.contains('@sha256:')) {
        error '--container must be pinned by sha256 digest'
    }
    if ((params.compare_max_forks as int) < 1 || (params.compare_max_forks as int) > 2) {
        error '--compare_max_forks must be 1 or 2'
    }

    staged = STAGE_COMMON_VIEW_ARCHIVE(channel.of(tuple(request_file, request_sha, implementation_file)))
    compare_inputs = staged.flatMap { archive, index ->
        (1..22).collect { c ->
            def chrom = String.format('%02d', c)
            tuple(chrom, archive, index, request_file, request_sha, implementation_file)
        }
    }
    compared = DEDUPLICATE_COMMON_VIEW(compare_inputs)
    catalogue_input = compared.map { _chrom, directory -> directory }.collect()
        .map { directories -> tuple(directories, request_file, request_sha, implementation_file) }
    CATALOGUE_COMMON_VIEWS(catalogue_input)
}
