nextflow.enable.dsl=2

def safePath(p) {
    return p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

process R02_STRUCTURE_REGIONS {
    cpus 1
    memory '1 GB'
    time '10m'
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'regions.json'
    path maps, stageAs:'maps/*'
    path code, stageAs:'code'
    val contractHash
    output:
    path 'geometry', emit: geometry
    script:
    """
    umask 077
    echo '${contractHash}  regions.json' | sha256sum -c -
    python3 code/r02_structure_regions.py --contract regions.json --maps-dir maps --output geometry
    """
}

workflow {
    if (![params.contract,params.code,params.output_dir].every {safePath(it)}) error 'Absolute safe paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    def c=new groovy.json.JsonSlurper().parse(new File(params.contract))
    if (!c.maps) error 'Nonempty maps required'
    def maps=c.maps.collect {chrom,source ->
        if (!safePath(source.path)) error 'Absolute safe map paths required'
        file(source.path,checkIfExists:true)
    }
    R02_STRUCTURE_REGIONS(file(params.contract,checkIfExists:true),maps,
        file(params.code,checkIfExists:true),params.contract_sha256)
}
