nextflow.enable.dsl=2
params.request = null
params.request_sha256 = null
params.source_bin = "${projectDir}/../bin"
params.output_dir = null

process AUDIT_EXISTING_LAI_SEPARATION {
    cpus 1
    memory '512 MB'
    time '10 min'
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path request
    val digest
    path code, stageAs:'code/*'
    path evidence, stageAs:'evidence/input????????'
    output:
    path 'separation-audit.json'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 code/r02_lai_separation.py --request '${request}' --request-sha256 '$digest' --output separation-audit.json
    """
}

workflow {
    if(!params.request || !params.output_dir || !(params.request_sha256 ==~ /[0-9a-f]{64}/))
        error 'Pinned request and output_dir required'
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.request,params.output_dir,params.source_bin].every{safe(it)}) error 'safe absolute paths required'
    def code=['r02_lai_separation.py','r02_lai_target_dependencies.py','r02_lai_kinship_dependencies.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    def request=file(params.request,checkIfExists:true)
    def specification=new groovy.json.JsonSlurper().parse(request)
    def units=new groovy.json.JsonSlurper().parse(file(specification.sources.units.path,checkIfExists:true))
    def paths=specification.sources.values().collect{it.path}
    units.units.each{unit ->
        paths.addAll([unit.sample_ids.path,unit.donor_ids.path,unit.mosaic_hashes.path])
        paths.addAll(unit.evidence.findAll{k,v -> k!='overlay'}.values().collect{it.path})
    }
    // Deep-cache input identities include all small evidence, not only the request.
    def evidence=paths.unique().collect{p -> if(!safe(p))error 'unsafe evidence path';file(p,checkIfExists:true)}
    AUDIT_EXISTING_LAI_SEPARATION(request,params.request_sha256,code,evidence)
}
