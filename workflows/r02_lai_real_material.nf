nextflow.enable.dsl=2

process R02_LAI_REAL_MATERIAL {
    tag "chr${chromosome}"
    cpus 3
    memory '2 GB'
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/chr${chromosome}", mode:'copy', overwrite:false
    input:
    tuple val(chromosome),path(contract),val(digest),path(dependencies, stageAs:'deps/input????/*')
    path code
    output:
    path 'material',emit: material
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_real_material.py --contract '$contract' --contract-sha256 '$digest' --output-dir material
    """
}

workflow {
    if(!params.campaign || !params.campaign_sha256 || !params.source_bin || !params.output_dir) error 'Explicit campaign, code and output required'
    if(params.max_forks != 1 || !params.task_time) error 'One task at a time and explicit duration required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream { stream ->
            byte[] buffer=new byte[1048576];int count
            while((count=stream.read(buffer))>0) md.update(buffer,0,count)
        }
        md.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !(d.path instanceof String) || !d.path.startsWith('/') || d.path.find(/[\n\r'"`$\\;|&<>]/)) error 'Invalid source descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash(p)!=d.sha256) error 'Source SHA256 mismatch before cache'
        p
    }
    def cp=file(params.campaign,checkIfExists:true)
    if(hash(cp)!=params.campaign_sha256) error 'Campaign SHA256 mismatch'
    def campaign=new groovy.json.JsonSlurper().parse(cp.toFile())
    if(campaign.schema!='r02_lai_real_material_campaign_v1' || !campaign.jobs || campaign.jobs*.chromosome.toSet().size()!=campaign.jobs.size()) error 'Invalid/duplicate campaign jobs'
    def code=['r02_lai_real_material.py','r02_lai_role_subset.py','r02_lai_simulate.py','r02_lai_support_review.py','r02_structure_regions.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    def jobs=campaign.jobs.collect{job ->
        def path=descriptor(job.contract)
        def c=new groovy.json.JsonSlurper().parse(path.toFile())
        if(c.chromosome!=job.chromosome) error 'Job chromosome differs'
        code.each{p -> if(c.code_sha256[p.name]!=hash(p)) error 'Frozen source code changed before cache'}
        def deps=[descriptor(c.original_roles),descriptor(c.genetic_map),descriptor(c.donor_full_audit)]
        c.sources.values().each{v -> v.values().each{d -> deps.add(descriptor(d))}}
        tuple(job.chromosome,path,job.contract.sha256,deps)
    }
    R02_LAI_REAL_MATERIAL(Channel.fromList(jobs),Channel.value(code))
}
