nextflow.enable.dsl=2

params.request = null
params.request_sha256 = null
params.implementation = null
params.output_dir = null
params.container = null
params.source_cache_manifest = null
params.stage_max_forks = 2
params.repair_max_forks = 2
params.stage_time = '2h'
params.repair_time = '8h'

def required(name, value) {
    if (value == null || value.toString().trim() == '') error "Missing required parameter --${name}"
    value
}

def safeGs(uri) {
    if (!(uri ==~ /^gs:\/\/projects-usp\/[A-Za-z0-9._\/-]+$/)) error "Unsafe/non-project GCS URI: ${uri}"
    uri
}

def safeGeneration(value) {
    def text = value.toString()
    if (!(text ==~ /^[1-9][0-9]*$/)) error "Invalid GCS generation: ${text}"
    text
}

def safeCached(path) {
    def text = path.toString()
    if (!(text ==~ /^\/opt\/r02-integrated-20261008ah\/k04-pairs\/[A-Za-z0-9._\/-]+$/)) {
        error "Cached source is outside the K04 root: ${text}"
    }
    file(text, checkIfExists: true)
}

process STAGE_K04_CHROMOSOME_SOURCES {
    tag "chr${chrom}"
    cpus 1
    memory '1 GB'
    time params.stage_time
    maxForks params.stage_max_forks as int
    cache 'deep'

    input:
    tuple val(chrom), val(event_uri), val(event_generation), val(event_sha),
          val(partial_uri), val(partial_generation), val(partial_sha),
          val(manifest_uri), val(manifest_generation), val(manifest_sha),
          val(markers_uri), val(markers_generation), val(markers_sha)

    output:
    tuple val(chrom), path("chr${chrom}/event.tar.gz"), path("chr${chrom}/partial.npz"),
          path("chr${chrom}/partial-manifest.json"), path("chr${chrom}/common_markers.tsv")

    script:
    """
    set -euo pipefail
    mkdir -p chr${chrom}
    gcloud storage cp '${event_uri}#${event_generation}' chr${chrom}/event.tar.gz
    gcloud storage cp '${partial_uri}#${partial_generation}' chr${chrom}/partial.npz
    gcloud storage cp '${manifest_uri}#${manifest_generation}' chr${chrom}/partial-manifest.json
    gcloud storage cp '${markers_uri}#${markers_generation}' chr${chrom}/common_markers.tsv
    printf '%s  %s\n' '${event_sha}' chr${chrom}/event.tar.gz > chr${chrom}/SHA256SUMS
    printf '%s  %s\n' '${partial_sha}' chr${chrom}/partial.npz >> chr${chrom}/SHA256SUMS
    printf '%s  %s\n' '${manifest_sha}' chr${chrom}/partial-manifest.json >> chr${chrom}/SHA256SUMS
    printf '%s  %s\n' '${markers_sha}' chr${chrom}/common_markers.tsv >> chr${chrom}/SHA256SUMS
    sha256sum -c chr${chrom}/SHA256SUMS
    """

    stub:
    """
    mkdir -p chr${chrom}
    touch chr${chrom}/event.tar.gz chr${chrom}/partial.npz \
      chr${chrom}/partial-manifest.json chr${chrom}/common_markers.tsv
    """
}

process STAGE_K04_SUPPLEMENT_ARCHIVE {
    tag 'exact-common-supplements'
    cpus 1
    memory '512 MB'
    time '20m'
    cache 'deep'

    input:
    tuple val(uri), val(generation), val(expected_sha)

    output:
    path 'common-supplements.tar'

    script:
    """
    set -euo pipefail
    gcloud storage cp '${uri}#${generation}' common-supplements.tar
    printf '%s  %s\n' '${expected_sha}' common-supplements.tar > SHA256SUMS
    sha256sum -c SHA256SUMS
    """

    stub:
    """
    touch common-supplements.tar
    """
}

process ADMIT_K04_CHROMOSOME_CACHE {
    tag "chr${chrom}"
    cpus 1
    memory '512 MB'
    time params.stage_time
    maxForks params.stage_max_forks as int
    cache 'deep'

    input:
    tuple val(chrom), path(event), val(event_sha), path(partial), val(partial_sha),
          path(manifest), val(manifest_sha), path(markers), val(markers_sha)

    output:
    tuple val(chrom), path(event), path(partial), path(manifest), path(markers)

    script:
    """
    set -euo pipefail
    printf '%s  %s\n' '${event_sha}' '${event}' > SHA256SUMS
    printf '%s  %s\n' '${partial_sha}' '${partial}' >> SHA256SUMS
    printf '%s  %s\n' '${manifest_sha}' '${manifest}' >> SHA256SUMS
    printf '%s  %s\n' '${markers_sha}' '${markers}' >> SHA256SUMS
    sha256sum -c SHA256SUMS
    """
}

process ADMIT_K04_SUPPLEMENT_CACHE {
    tag 'exact-common-supplements'
    cpus 1
    memory '512 MB'
    time '20m'
    cache 'deep'

    input:
    tuple path(archive), val(expected_sha)

    output:
    path archive

    script:
    """
    set -euo pipefail
    printf '%s  %s\n' '${expected_sha}' '${archive}' > SHA256SUMS
    sha256sum -c SHA256SUMS
    """
}

process REPAIR_K04_CHROMOSOME {
    tag "chr${chrom}"
    cpus 1
    memory '4 GB'
    time params.repair_time
    maxForks params.repair_max_forks as int
    container params.container
    cache 'deep'
    publishDir "${params.output_dir}/chromosomes", mode: 'copy', overwrite: false

    input:
    tuple val(chrom), path(event), path(partial), path(partial_manifest), path(markers),
          path(supplements),
          path(request), val(request_sha), path(implementation)

    output:
    tuple val(chrom), path("repair_chr${chrom}")

    script:
    """
    python3 ${implementation} \
      --stage chromosome \
      --request ${request} \
      --request-sha256 ${request_sha} \
      --chromosome ${chrom} \
      --event-archive ${event} \
      --partial ${partial} \
      --partial-manifest ${partial_manifest} \
      --common-markers ${markers} \
      --supplement-archive ${supplements} \
      --output repair_chr${chrom}
    """

    stub:
    """
    mkdir repair_chr${chrom}
    touch repair_chr${chrom}/repair.npz repair_chr${chrom}/manifest.json
    """
}

process AGGREGATE_K04_REPAIR {
    tag '22-whole-chromosome-exclusions'
    cpus 1
    memory '2 GB'
    time '20m'
    container params.container
    cache 'deep'
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple path(directories), path(request), val(request_sha), path(implementation)

    output:
    path 'catalogue'

    script:
    """
    python3 ${implementation} \
      --stage aggregate \
      --request ${request} \
      --request-sha256 ${request_sha} \
      --chromosome-directories ${directories.join(' ')} \
      --output catalogue
    """

    stub:
    """
    mkdir catalogue
    touch catalogue/repair.npz catalogue/manifest.json
    """
}

workflow {
    request_file = file(required('request', params.request), checkIfExists: true)
    implementation_file = file(required('implementation', params.implementation), checkIfExists: true)
    request_sha = required('request_sha256', params.request_sha256)
    required('output_dir', params.output_dir)
    container_ref = required('container', params.container)
    if (!container_ref.contains('@sha256:')) error '--container must be pinned by sha256 digest'
    if ((params.stage_max_forks as int) < 1 || (params.stage_max_forks as int) > 2 ||
        (params.repair_max_forks as int) < 1 || (params.repair_max_forks as int) > 2) {
        error 'stage_max_forks and repair_max_forks must be 1 or 2'
    }

    def requestObject = new groovy.json.JsonSlurper().parse(request_file)
    if (requestObject.schema != 'r02_k04_common_pair_repair_request_v2') error 'Wrong request schema'
    def supplement = requestObject.supplement_archive
    if (params.source_cache_manifest != null) {
        cache_file = file(params.source_cache_manifest, checkIfExists: true)
        def cacheObject = new groovy.json.JsonSlurper().parse(cache_file)
        if (cacheObject.schema != 'r02_k04_staged_source_cache_v1' ||
            cacheObject.request_sha256 != request_sha ||
            cacheObject.chromosomes.keySet() as Set != requestObject.chromosomes.keySet() as Set) {
            error 'Cached-source manifest/request/22-chromosome inventory differs'
        }
        cached_rows = channel.fromList((1..22).collect { value ->
            def chrom = value.toString(); def row = requestObject.chromosomes[chrom]
            def cached = cacheObject.chromosomes[chrom]
            tuple(String.format('%02d', value),
                  safeCached(cached.event_archive), row.event_archive.sha256,
                  safeCached(cached.partial), row.partial.sha256,
                  safeCached(cached.partial_manifest), row.partial_manifest.sha256,
                  safeCached(cached.common_markers), row.common_markers.sha256)
        })
        staged = ADMIT_K04_CHROMOSOME_CACHE(cached_rows)
        supplement_archive = ADMIT_K04_SUPPLEMENT_CACHE(
            channel.of(tuple(safeCached(cacheObject.supplement_archive), supplement.sha256)))
    } else {
        source_rows = channel.fromList((1..22).collect { value ->
            def chrom = value.toString(); def row = requestObject.chromosomes[chrom]
            def event = row.event_archive; def partial = row.partial
            def manifest = row.partial_manifest; def markers = row.common_markers
            tuple(String.format('%02d', value),
                  safeGs(event.uri), safeGeneration(event.generation), event.sha256,
                  safeGs(partial.uri), safeGeneration(partial.generation), partial.sha256,
                  safeGs(manifest.uri), safeGeneration(manifest.generation), manifest.sha256,
                  safeGs(markers.uri), safeGeneration(markers.generation), markers.sha256)
        })
        staged = STAGE_K04_CHROMOSOME_SOURCES(source_rows)
        supplement_archive = STAGE_K04_SUPPLEMENT_ARCHIVE(
            channel.of(tuple(safeGs(supplement.uri), safeGeneration(supplement.generation), supplement.sha256)))
    }
    repair_inputs = staged.combine(supplement_archive).map { chrom, event, partial, manifest, markers, supplements ->
        tuple(chrom, event, partial, manifest, markers, supplements,
              request_file, request_sha, implementation_file)
    }
    repaired = REPAIR_K04_CHROMOSOME(repair_inputs)
    aggregate_input = repaired.map { _chrom, directory -> directory }.collect()
        .map { directories -> tuple(directories, request_file, request_sha, implementation_file) }
    AGGREGATE_K04_REPAIR(aggregate_input)
}
