nextflow.enable.dsl=2

process R02_LAI_DEVELOPMENT_DESIGN {
    cpus 1
    memory { "${limits.memory_gib} GB" }
    time { "${limits.timeout_seconds} sec" }
    maxForks 1
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'contract.json'
    val digest
    val limits
    path dependencies, stageAs:'inputs/input????????'
    path code_files
    output:
    path 'design', emit: design
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_development_design.py --contract contract.json --contract-sha256 '$digest' --output-dir design
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if(!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream ->
            byte[] buffer=new byte[1048576]; int count
            while((count=stream.read(buffer))>0) md.update(buffer,0,count)
        }
        md.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid source descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash(p)!=d.sha256) error 'Source SHA256 mismatch before cache lookup'
        p
    }
    def parse={p -> new groovy.json.JsonSlurper().parse(p.toFile())}
    def cpath=descriptor([path:params.contract,sha256:params.contract_sha256]); def c=parse(cpath)
    if(c.schema!='r02_lai_development_design_input_v1') error 'Invalid design schema'
    def material=descriptor(c.material_report); def review=descriptor(c.pair_review)
    def dependencies=[material,review,descriptor(c.original_roles)]
    [material,review].each {source ->
        parse(source).outputs_sha256.each {name,digest ->
            if(!(name ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/) || name=='manifest.json') error 'Unsafe evidence output'
            dependencies+=descriptor([path:source.parent.resolve(name).toString(),sha256:digest])
        }
    }
    parse(material).inputs_sha256.each {name,digest ->
        def p=file(name)
        if(p.name=='manifest.json' && p.parent.name=='private_preparation') {
            def source=descriptor([path:name,sha256:digest]); dependencies+=source
            parse(source).outputs_sha256.each {output,outputHash ->
                if(!(output ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/) || output=='manifest.json') error 'Unsafe donor output'
                dependencies+=descriptor([path:source.parent.resolve(output).toString(),sha256:outputHash])
            }
        }
    }
    def code=['r02_lai_development_design.py','r02_lai_simulation_plan.py','r02_lai_simulate.py','r02_lai_support_review.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_LAI_DEVELOPMENT_DESIGN(cpath,params.contract_sha256,c.resources,dependencies,code)
}
