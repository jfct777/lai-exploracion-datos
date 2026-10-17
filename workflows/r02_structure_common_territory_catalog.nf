nextflow.enable.dsl=2

def safeAbsolute(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def fileHash(path) {
    def digest=java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream,digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}

def pinned(name,pin) {
    if(!safeAbsolute(name)||!(pin ==~ /[0-9a-f]{64}/)) error 'Safe pinned catalogue input required'
    def path=file(name,checkIfExists:true)
    if(fileHash(path)!=pin) error 'Catalogue input changed before cache lookup'
    path
}

process COMMON_TERRITORY_CATALOGUE {
    tag '88_common_territories'
    cpus 1
    memory '1 GB'
    time '5m'
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path code
    tuple path(territories), val(territoriesPin), path(inventory), val(inventoryPin),
          path(policyRequest), val(policyRequestPin), path(bridgeRequest), val(bridgeRequestPin),
          path(bridgeAdmission), val(bridgeAdmissionPin), path(searchAdmission), val(searchAdmissionPin),
          path(assemblySources), val(assemblySourcesPin), path(commonArchiveSource), val(commonArchiveSourcePin),
          path(wholeEvaluation), val(wholeEvaluationPin), path(ldEvaluation), val(ldEvaluationPin),
          path(neuralTargetManifest), val(neuralTargetManifestPin)
    output:
    path 'common-territory-catalogue'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1
    python3 r02_structure_common_territory_catalog.py \
      --territories '${territories}' --territories-sha256 '${territoriesPin}' \
      --inventory '${inventory}' --inventory-sha256 '${inventoryPin}' \
      --policy-request '${policyRequest}' --policy-request-sha256 '${policyRequestPin}' \
      --bridge-request '${bridgeRequest}' --bridge-request-sha256 '${bridgeRequestPin}' \
      --bridge-admission '${bridgeAdmission}' --bridge-admission-sha256 '${bridgeAdmissionPin}' \
      --search-admission '${searchAdmission}' --search-admission-sha256 '${searchAdmissionPin}' \
      --assembly-sources '${assemblySources}' --assembly-sources-sha256 '${assemblySourcesPin}' \
      --common-archive-source '${commonArchiveSource}' \
      --common-archive-source-sha256 '${commonArchiveSourcePin}' \
      --whole-evaluation '${wholeEvaluation}' --whole-evaluation-sha256 '${wholeEvaluationPin}' \
      --ld-evaluation '${ldEvaluation}' --ld-evaluation-sha256 '${ldEvaluationPin}' \
      --neural-target-manifest '${neuralTargetManifest}' \
      --neural-target-manifest-sha256 '${neuralTargetManifestPin}' \
      --output common-territory-catalogue
    """
}

workflow {
    if(![params.code,params.territories,params.inventory,params.policy_request,
         params.bridge_request,params.bridge_admission,params.search_admission,
         params.assembly_sources,params.common_archive_source,params.whole_evaluation,
         params.ld_evaluation,params.neural_target_manifest,
         params.output_dir].every{safeAbsolute(it)}) error 'Explicit absolute catalogue paths required'
    def sources=tuple(
        pinned(params.territories,params.territories_sha256),params.territories_sha256,
        pinned(params.inventory,params.inventory_sha256),params.inventory_sha256,
        pinned(params.policy_request,params.policy_request_sha256),params.policy_request_sha256,
        pinned(params.bridge_request,params.bridge_request_sha256),params.bridge_request_sha256,
        pinned(params.bridge_admission,params.bridge_admission_sha256),params.bridge_admission_sha256,
        pinned(params.search_admission,params.search_admission_sha256),params.search_admission_sha256,
        pinned(params.assembly_sources,params.assembly_sources_sha256),params.assembly_sources_sha256,
        pinned(params.common_archive_source,params.common_archive_source_sha256),params.common_archive_source_sha256,
        pinned(params.whole_evaluation,params.whole_evaluation_sha256),params.whole_evaluation_sha256,
        pinned(params.ld_evaluation,params.ld_evaluation_sha256),params.ld_evaluation_sha256,
        pinned(params.neural_target_manifest,params.neural_target_manifest_sha256),params.neural_target_manifest_sha256)
    COMMON_TERRITORY_CATALOGUE(pinned(params.code,params.code_sha256),Channel.value(sources))
}
