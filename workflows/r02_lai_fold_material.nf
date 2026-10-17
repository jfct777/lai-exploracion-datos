nextflow.enable.dsl=2
params.campaign=null
params.contract=null
params.max_forks=1

process R02_LAI_FOLD_MATERIAL {
    tag "chr${chromosome}"
    cpus 1
    memory { "${limits.memory_gib} GB" }
    time { "${limits.timeout_seconds} sec" }
    maxForks params.max_forks
    cache 'deep'
    publishDir { campaign ? "${params.output_dir}/chr${chromosome}" : params.output_dir }, mode:'copy', overwrite:false
    input:
    tuple val(chromosome), path(contract,stageAs:'contract.json'), val(digest), val(limits), path(dependencies,stageAs:'inputs/input????????'), val(campaign)
    path code_files
    output:
    path 'folds', emit: folds
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_fold_material.py --contract contract.json --contract-sha256 '$digest' --output-dir folds
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    def campaignMode=params.campaign!=null
    if(![params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if(campaignMode && params.contract!=null) error 'Choose one contract or campaign, not both'
    if(params.max_forks!=null && (!(params.max_forks instanceof Number) || params.max_forks.toInteger()!=params.max_forks || params.max_forks<1 || params.max_forks>22)) error 'max_forks must be an integer1..22'
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
    def jobs
    if(campaignMode) {
        def cp=descriptor([path:params.campaign,sha256:params.campaign_sha256]); def doc=parse(cp)
        if(doc.schema!='r02_lai_fold_campaign_v1' || !(doc.jobs instanceof List) || !doc.jobs || doc.jobs.size()>22) error 'Invalid fold campaign'
        if(doc.jobs.collect{it.chromosome}.toSet().size()!=doc.jobs.size()) error 'Duplicate campaign chromosome'
        jobs=doc.jobs
    } else jobs=[[contract:[path:params.contract,sha256:params.contract_sha256]]]
    def entries=jobs.collect { job ->
        def cpath=descriptor(job.contract); def c=parse(cpath)
        if(!(c.schema in ['r02_lai_fold_material_input_v1','r02_lai_fold_material_input_v2'])) error 'Invalid fold schema'
        if(!(c.chromosome ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Invalid autosome'
        if(campaignMode && job.chromosome!=c.chromosome) error 'Campaign/contract chromosome mismatch'
        def dependencies=[descriptor(c.original_roles)]
        ['batch','material_report','material'].each {key ->
            // Path is Iterable: add it as one file, not its path components.
            def source=descriptor(c[key]);dependencies.add(source)
            parse(source).outputs_sha256.each {name,digest ->
                if(!(name ==~ /[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)*/) || name=='manifest.json') error 'Unsafe evidence output'
                if(key!='material' || name in ["source_valid.chr${c.chromosome}.vcf.gz".toString(),"source_valid.chr${c.chromosome}.vcf.gz.csi".toString(),'genetic.map'])
                    dependencies.add(descriptor([path:source.parent.resolve(name).toString(),sha256:digest]))
            }
        }
        tuple(c.chromosome,cpath,job.contract.sha256,c.resources,dependencies,campaignMode)
    }
    def names=['r02_lai_fold_material.py','r02_lai_simulation_batch.py','r02_lai_simulation_plan.py','r02_lai_role_subset.py','r02_lai_simulate.py','r02_lai_support_review.py']
    def code=names.collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_LAI_FOLD_MATERIAL(Channel.fromList(entries),code)
}
