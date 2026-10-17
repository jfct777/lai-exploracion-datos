nextflow.enable.dsl=2

def safePath(p) {
    return p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

process R02_STRUCTURE_REGION_SUPPORT {
    cpus params.task_cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'support.json'
    path sources, stageAs:'sources/*'
    val sourceSignatures
    path code, stageAs:'code'
    val contractHash
    output:
    path 'public', emit: evidence
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    echo '${contractHash}  support.json' | sha256sum -c -
    python3 code/r02_structure_region_support.py --contract support.json --output support --public-output public
    """
}

workflow {
    if (![params.contract,params.code,params.output_dir].every {safePath(it)}) error 'Absolute safe paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Support contract SHA256 required'
    if (!(params.task_cpus?.toString() ==~ /[1-9][0-9]*/) || !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)) error 'Explicit positive resources required'
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[hmd]/)) error 'Explicit task deadline required'
    def c=new groovy.json.JsonSlurper().parse(new File(params.contract))
    if (c.resources.threads != params.task_cpus.toInteger()) error 'CPU reservation and tool threads must agree'
    if (params.memory_gb.toInteger()*1024 < c.resources.max_array_mb+c.resources.plink_memory_mb+512) error 'Insufficient memory reservation'
    def desc=[c.source.common_vcf,c.source.common_index,c.persons,c.map,c.policy]
    def sources=desc.collect { d ->
        if (!safePath(d.path)) error 'Safe absolute authenticated source paths required'
        file(d.path,checkIfExists:true)
    }
    def signatures=sources.collect {p ->
        java.nio.file.Files.readAttributes(p.toRealPath(),'unix:dev,ino,size,lastModifiedTime,ctime').collectEntries {k,v -> [(k):v.toString()]}
    }
    R02_STRUCTURE_REGION_SUPPORT(file(params.contract,checkIfExists:true),sources,signatures,
        file(params.code,checkIfExists:true),params.contract_sha256)
}
