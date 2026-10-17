nextflow.enable.dsl=2

process K04_SUPPORT_REDUCE {
    cpus 1
    memory '1 GB'
    time '5m'
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path specification, stageAs: 'inputs.json'
    val digest
    path code, stageAs: 'r02_k04_identified_comparison.py'
    path source_products, stageAs: 'sources??/*'
    output:
    path 'support.json'
    script:
    """
    umask 077
    python3 r02_k04_identified_comparison.py --inputs inputs.json \\
      --inputs-sha256 '${digest}' --output support.json
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') &&
        !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.inputs,params.source_code,params.output_dir].every{safe.call(it)} ||
       !(params.inputs_sha256 ==~ /[0-9a-f]{64}/)) error 'Safe absolute paths and SHA256 required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream{stream -> new java.security.DigestInputStream(stream,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    def inputs=file(params.inputs,checkIfExists:true)
    if(hash.call(inputs)!=params.inputs_sha256) error 'Specification SHA256 differs before cache lookup'
    def spec=new groovy.json.JsonSlurper().parse(inputs.toFile())
    def products=[]
    spec.chromosomes.each{entry ->
        if(!safe.call(entry.directory)) error 'Safe product directory required'
        def m=file(entry.directory+'/manifest.json',checkIfExists:true)
        if(hash.call(m)!=entry.manifest_sha256) error 'Manifest SHA256 differs before cache lookup'
        products.add(m)
        def manifest=new groovy.json.JsonSlurper().parse(m.toFile())
        ['summary.json','persons.tsv','person_metrics.tsv'].each{name ->
            def p=file(entry.directory+'/'+name,checkIfExists:true)
            if(hash.call(p)!=manifest.outputs_sha256[name]) error 'Source SHA256 differs before cache lookup'
            products.add(p)
        }
    }
    // Explicit hashed source files participate in cache; unique staging names
    // avoid basename collisions. Original absolute paths remain read-only inputs.
    K04_SUPPORT_REDUCE(inputs,params.inputs_sha256,file(params.source_code,checkIfExists:true),products)
}
