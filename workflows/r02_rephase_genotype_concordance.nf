nextflow.enable.dsl=2
params.campaign=null
params.campaign_sha256=null
params.source_bin=null
params.output_dir=null

process R02_REPHASE_GENOTYPE_CONCORDANCE {
    tag "chr${chromosome}"
    cpus 1
    memory '3 GB'
    time '4h'
    maxForks 2
    cache 'deep'
    publishDir "${params.output_dir}/chr${chromosome}", mode:'link', overwrite:false
    input:
    tuple val(chromosome),path(job),val(digest)
    path code
    output:
    tuple val(chromosome),path('result'),path('complete.json'),path('publication.json')
    script:
    """
    python3 -B r02_rephase_genotype_task.py --job '${job}' --sha256 '${digest}'
    """
}

workflow {
    def sha={p->java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString()}
    def cf=file(params.campaign,checkIfExists:true)
    if(sha.call(cf)!=params.campaign_sha256)error('Campaign SHA differs')
    def c=new groovy.json.JsonSlurper().parse(cf.toFile())
    if(c.schema!='r02_rephase_genotype_campaign_v1'||c.jobs.size()!=22||c.jobs.collect{it.chromosome}.toSet()!=(1..22).collect{it.toString()}.toSet())error('All 22 unique autosomes required')
    def jobs=c.jobs.collect{j->def p=file(j.path,checkIfExists:true);if(sha.call(p)!=j.sha256)error('Job SHA differs');tuple(j.chromosome,p,j.sha256)}
    def code=file("${params.source_bin}/*.py")
    R02_REPHASE_GENOTYPE_CONCORDANCE(Channel.fromList(jobs),code)
}
