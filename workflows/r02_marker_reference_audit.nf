nextflow.enable.dsl=2

params.jobs = []
params.source_bin = null
params.output_dir = null
params.python = 'python3'
params.bcftools = 'bcftools'
params.task_cpus = 1
params.task_memory = '2 GB'
params.task_time = '2h'
params.max_forks = 2

process R02_MARKER_REFERENCE_AUDIT {
    tag "${meta.kind}_chr${meta.chromosome}"
    cache 'deep'
    cpus params.task_cpus
    memory params.task_memory
    time params.task_time
    maxForks params.max_forks
    publishDir { "${params.output_dir}/${meta.kind}/chr${meta.chromosome}" }, mode:'copy', overwrite:false
    input:
    tuple val(meta), path(job,stageAs:'job.json'), val(job_sha),
          path(inventory,stageAs:'inventory.tsv.gz'), path(fasta,stageAs:'reference.fa'),
          path(fai,stageAs:'reference.fa.fai'), path(identity,stageAs:'identity.json')
    path code_files,stageAs:'code/*'
    output:
    tuple val(meta), path('result.json'), path('mismatches.tsv.gz'), emit: results
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    '${params.python}' code/r02_marker_reference_audit.py \\
      --manifest job.json --manifest-sha256 '${job_sha}' --inventory inventory.tsv.gz \\
      --fasta reference.fa --fai reference.fa.fai --identity identity.json \\
      --bcftools '${params.bcftools}' --out result.json --mismatches mismatches.tsv.gz
    """
}

workflow {
    def unsafe={v->!(v instanceof String)||!v.startsWith('/')||v.tokenize('/').contains('..')||v.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    if (!(params.jobs instanceof List) || params.jobs.size()!=44 || params.jobs.unique(false).size()!=44)
        error 'Exactly 44 distinct job manifests required (source/rare x 22 autosomes)'
    if ([params.source_bin,params.output_dir].any{unsafe.call(it)}) error 'Absolute safe source/output paths required'
    if (params.task_cpus.toString()!='1' || !(params.max_forks.toString() in ['1','2']))
        error 'Audits require 1 CPU each and at most 2 concurrent tasks'
    if (!(params.task_memory instanceof String) || !(params.task_memory ==~ /[1-9][0-9]* (MB|GB)/) ||
        !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit task memory and time required'
    if ([params.python,params.bcftools].any{!(it instanceof String)||it.find(/[\n\r'"`$\\;|&<>]/)})
        error 'Unsafe executable parameter'
    def sha={p->java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString()}
    def parse={p->new groovy.json.JsonSlurper().parse(file(p,checkIfExists:true).toFile())}
    def keys=[]
    def jobs=params.jobs.collect{jobPath->
        if(unsafe.call(jobPath))error 'Unsafe job path'
        def jp=file(jobPath,checkIfExists:true);def j=parse.call(jobPath)
        if(j.schema!='r02_marker_reference_audit_job_v1'||!(j.kind in ['source','rare'])||
           !(j.chromosome.toString()==~ /([1-9]|1[0-9]|2[0-2])/))error 'Invalid job identity'
        def key="${j.kind}:${j.chromosome}";if(keys.contains(key))error 'Duplicate kind/chromosome';keys.add(key)
        def descriptors=[j.inventory.path,j.reference.fasta_path,j.reference.fai_path,j.reference.identity_path]
        if(descriptors.any{unsafe.call(it)})error 'Unsafe job input path'
        tuple([kind:j.kind,chromosome:j.chromosome.toString()],jp,sha.call(jp),
              file(j.inventory.path,checkIfExists:true),file(j.reference.fasta_path,checkIfExists:true),
              file(j.reference.fai_path,checkIfExists:true),file(j.reference.identity_path,checkIfExists:true))
    }
    def expected=['source','rare'].collectMany{k->(1..22).collect{c->"${k}:${c}"}}.toSet()
    if(keys.toSet()!=expected)error 'Batch must contain source and rare for every autosome 1-22'
    def code=['r02_marker_reference_audit.py','r02_lai_build_verification.py','r02_lai_allele_support.py',
              'r02_genomic_pair_evidence.py'].collect{file("${params.source_bin}/${it}",checkIfExists:true)}
    R02_MARKER_REFERENCE_AUDIT(Channel.fromList(jobs),code)
}
