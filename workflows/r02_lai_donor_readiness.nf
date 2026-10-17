nextflow.enable.dsl=2

process DONOR_DEPENDENCY {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir "${params.output_dir}/dependency", mode:'copy', overwrite:false
    input:
    path contract, stageAs:'contract.json'
    val digest
    path dependencies, stageAs:'inputs/file????????'
    path code_files
    output:
    path 'dependency', emit: result
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_donor_readiness.py dependency --contract contract.json --contract-sha256 '${digest}' --output-dir dependency
    """
}

process DONOR_CHROMOSOME_READINESS {
    tag "chr${chrom}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks 1
    cache 'deep'
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(dependencies, stageAs:'inputs/file????????')
    path contract, stageAs:'contract.json'
    val digest
    path dependency, stageAs:'dependency_input'
    path code_files
    output:
    tuple val(chrom), path('readiness'), emit: result
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_donor_readiness.py chromosome --contract contract.json --contract-sha256 '${digest}' \
      --chromosome '${chrom}' --dependency-manifest dependency_input/manifest.json --output-dir readiness
    """
}

process DONOR_READINESS_AGGREGATE {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir "${params.output_dir}/aggregate", mode:'copy', overwrite:false
    input:
    path products, stageAs:'chromosomes/chr??'
    path contract, stageAs:'contract.json'
    val digest
    path dependency, stageAs:'dependency_input'
    path code_files
    output:
    path 'aggregate', emit: result
    script:
    def items=products instanceof List ? products : [products]
    def arguments=items.collect{p->"--manifest '${p}/manifest.json'"}.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_donor_readiness.py aggregate --contract contract.json --contract-sha256 '${digest}' \
      --dependency-manifest dependency_input/manifest.json ${arguments} --output-dir aggregate
    """
}

workflow {
    def safe={p->p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    if (![params.contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    if (!(params.memory_gb?.toString() in ['1','2']) || !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded memory/time required'
    def parse={p->new groovy.json.JsonSlurper().parse(file(p,checkIfExists:true).toFile())}
    def descriptor={d->
        if (!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid descriptor'
        file(d.path,checkIfExists:true)
    }
    def bundle={d,names->
        def p=descriptor(d); def m=parse(d.path)
        [p]+names.collect{name->
            if (!(m.outputs_sha256[name] ==~ /[0-9a-f]{64}/)) error 'Unsealed input artifact'
            file(p.parent.resolve(name),checkIfExists:true)
        }
    }
    def c=parse(params.contract)
    if (c.schema!='r02_lai_donor_readiness_input_v1' || c.resources.cpus!=1 || !c.jobs) error 'Invalid campaign'
    def chromosomes=c.jobs.collect{it.chromosome}
    if (chromosomes.size()!=chromosomes.toSet().size() || chromosomes.toSet()!=(1..22).collect{it.toString()}.toSet()) error 'Exactly22autosomes required'
    def code=['r02_lai_donor_readiness.py','r02_lai_material_decision.py','r02_lai_simulation_material.py','r02_lai_support_review.py'].collect{file("${params.source_bin}/${it}",checkIfExists:true)}
    def kinDeps=[descriptor(c.criteria)]+bundle(c.pcair,['summary.json','pcair_roles.private.tsv'])+bundle(c.donor_aggregate,['donors.private.tsv'])
    c.pcrelate.each{k,v->kinDeps.addAll(bundle(v,['summary.json','pcrelate_pairs.private.tsv.gz']))}
    def chromosomeJobs=c.jobs.collect{j->
        if (!(j.chromosome ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Invalid chromosome'
        tuple(j.chromosome,[descriptor(c.criteria),descriptor(j.decision_manifest),descriptor(j.source_manifest)]+
            bundle(j.donor_manifest,['donors.private.tsv','calls.private.tsv.gz','carriages.private.tsv.gz']))
    }
    def contract=file(params.contract,checkIfExists:true)
    DONOR_DEPENDENCY(contract,params.contract_sha256,kinDeps,code)
    DONOR_CHROMOSOME_READINESS(Channel.fromList(chromosomeJobs),contract,params.contract_sha256,DONOR_DEPENDENCY.out.result.first(),code)
    DONOR_READINESS_AGGREGATE(DONOR_CHROMOSOME_READINESS.out.result.map{chromosome_id,product->product}.collect(),
        contract,params.contract_sha256,DONOR_DEPENDENCY.out.result.first(),code)
}
