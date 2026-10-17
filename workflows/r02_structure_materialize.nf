nextflow.enable.dsl=2

process R02_STRUCTURE_MATERIALIZE_PLAN {
    cpus 1
    memory '1 GB'
    time params.task_time
    cache 'deep'
    input:
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    val contractHash
    output:
    path 'plan', emit: planned
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    test "\$(sha256sum bundle/materialize.json | cut -d ' ' -f1)" = '${contractHash}'
    python3 code/r02_structure_materialize.py plan --bundle bundle --out plan
    """
}

process R02_STRUCTURE_MATERIALIZE_SOURCE {
    tag "chr${chromosome}"
    cpus 1
    memory '1 GB'
    time params.task_time
    maxForks params.max_forks
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(chromosome), path(sourceFiles), val(sourceSignatures)
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    output:
    tuple val(chromosome), path("verified_chr${chromosome}"), emit: verified
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_materialize.py verify-source --bundle bundle \
      --chromosome '${chromosome}' --out 'verified_chr${chromosome}'
    """
}

process R02_STRUCTURE_MATERIALIZE_SLICE {
    tag "${taskId}"
    cpus params.task_cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(chromosome), val(taskId), path(request), path(sourceFiles), path(verification)
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    path fitCache, stageAs:'fit_cache_descriptor.json'
    path fitFiles, stageAs:'fit_cache_files/*'
    val fitCacheHash
    output:
    path "partial_${taskId}", emit: partials
    script:
    def cached = fitCacheHash ? "--fit-cache '${fitCache}'" : ''
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    ${fitCacheHash ? "test \"\$(sha256sum '${fitCache}' | cut -d ' ' -f1)\" = '${fitCacheHash}'" : 'true'}
    python3 code/r02_structure_materialize.py slice --bundle bundle \
      --request '${request}' --verified '${verification}/manifest.json' --out 'partial_${taskId}' ${cached}
    """
}

process R02_STRUCTURE_MATERIALIZE_ASSEMBLE_BLOCK {
    tag "${blockId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    input:
    tuple val(blockId), path(partials)
    path plan
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    val exclusionScope
    output:
    path "shard_${blockId}", emit: shard
    script:
    def entries=partials.collect { "'${it}'" }.join(' ')
    def scope=exclusionScope ? "--exclusion-scope '${exclusionScope}'" : ''
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_materialize.py assemble-block --bundle bundle \
      --plan '${plan}/manifest.json' --block-id '${blockId}' --partials ${entries} \
      --out 'shard_${blockId}' ${scope}
    """
}

process R02_STRUCTURE_MATERIALIZE_INDEX {
    cpus 1
    memory '1 GB'
    time params.task_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path shards
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    val exclusionScope
    output:
    path 'prepared', emit: prepared
    script:
    def entries=shards.collect { "'${it}'" }.join(' ')
    def scope=exclusionScope ? "--exclusion-scope '${exclusionScope}'" : ''
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_materialize.py assemble-index --bundle bundle \
      --shards ${entries} --out prepared ${scope}
    """
}

process R02_STRUCTURE_MATERIALIZE_ASSEMBLE {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path partials
    path plan
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    output:
    path 'prepared', emit: prepared
    script:
    def entries=partials.collect { "'${it}'" }.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_materialize.py assemble --bundle bundle \
      --plan '${plan}/manifest.json' --partials ${entries} --out prepared
    """
}

workflow {
    def safe={ p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.bundle,params.code,params.output_dir].every(safe)) error 'Absolute safe bundle/code/output paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'materialize.json SHA256 required'
    for (name in ['task_cpus','memory_gb','max_forks']) if (!(params[name]?.toString() ==~ /[1-9][0-9]*/)) error "Positive ${name} required"
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[hmd]/)) error 'Explicit positive task_time required'
    def bundle=file(params.bundle,checkIfExists:true)
    def code=file(params.code,checkIfExists:true)
    def contract=new groovy.json.JsonSlurper().parse(new File(params.bundle+'/materialize.json'))
    def phase=params.stage ?: 'legacy'
    if (!(phase in ['legacy','extract','assemble'])) error 'stage must be extract or assemble (legacy retained for compatibility tests)'
    if (contract.resources.threads != params.task_cpus.toInteger()) error 'CPU reservation must equal tool threads'
    if (params.memory_gb.toInteger()*1024 < contract.resources.max_array_mb + contract.resources.plink_memory_mb + 512) error 'Memory must cover Python, PLINK2 and 512MiB overhead'
    def extractionChromosomes=phase=='extract' ? [params.chromosome?.toString()] : contract.sources.keySet().toList()
    if (phase=='extract' && !(params.chromosome?.toString() in contract.sources.keySet())) error 'extract requires exactly one declared chromosome'
    // Assembly never resolves raw VCF paths. Extraction admits only one
    // chromosome; the external queue publishes it before staging the next.
    def extractionSources=phase=='assemble' ? [:] : contract.sources.findAll { chrom,src -> chrom in extractionChromosomes }
    def sourcePaths=extractionSources.collectEntries { chrom,src ->
        def paths=['common_vcf','common_index','rare_vcf','rare_index'].collect { k ->
            if (!safe(src[k].path)) error 'Unsafe source path'
            file(src[k].path,checkIfExists:true)
        }
        [(chrom):paths]
    }
    // Source bytes are SHA256-checked once per chromosome, not once per block.
    // ctime/inode are cache inputs so replacing bytes while restoring size and
    // mtime cannot silently reuse a SOURCE hash check from an earlier run.
    def sourceSignatures=sourcePaths.collectEntries { chrom,paths ->
        [(chrom):paths.collect { p ->
            def real=p.toRealPath()
            def attributes=java.nio.file.Files.readAttributes(real,'unix:dev,ino,size,lastModifiedTime,ctime')
            attributes.collectEntries { key,value -> [(key):value.toString()] }
        }]
    }
    R02_STRUCTURE_MATERIALIZE_PLAN(bundle,code,params.contract_sha256)
    if (phase in ['extract','legacy']) {
        def cacheDescriptor=bundle.resolve('materialize.json')
        def cacheFiles=[cacheDescriptor]
        def cacheHash=''
        if (params.fit_cache) {
            if (!safe(params.fit_cache) || !(params.fit_cache_sha256 ==~ /[0-9a-f]{64}/)) error 'Authenticated FIT cache required'
            cacheDescriptor=file(params.fit_cache,checkIfExists:true)
            def cache=new groovy.json.JsonSlurper().parse(new File(params.fit_cache))
            if (cache.keySet() != ['manifest','contract','retained'].toSet()) error 'Exact FIT cache descriptors required'
            cacheFiles=cache.values().collect { d ->
                if (!safe(d.path)) error 'Unsafe FIT artifact path'
                file(d.path,checkIfExists:true)
            }
            cacheHash=params.fit_cache_sha256
        }
        if (contract.target_selection=='AUTHENTICATED_FIT_CACHE_REQUIRED' && !cacheHash) error 'FIT cache required; do not repeat pruning'
        // Full-source hashing starts only after the authenticated plan passes.
        def admittedSources=R02_STRUCTURE_MATERIALIZE_PLAN.out.planned.flatMap {
            sourcePaths.collect { chrom,paths -> tuple(chrom,paths,sourceSignatures[chrom]) }
        }
        R02_STRUCTURE_MATERIALIZE_SOURCE(admittedSources,bundle,code)
        def requests=R02_STRUCTURE_MATERIALIZE_PLAN.out.planned.flatMap { p ->
            def plan=new groovy.json.JsonSlurper().parse(new File(p.toString()+'/manifest.json'))
            plan.tasks.findAll { taskId,request -> request.chromosome in extractionChromosomes }.collect { taskId,request ->
                tuple(request.chromosome,taskId,file(p.toString()+'/tasks/'+taskId+'.json',checkIfExists:true),sourcePaths[request.chromosome]) }
        }
        R02_STRUCTURE_MATERIALIZE_SLICE(requests.combine(R02_STRUCTURE_MATERIALIZE_SOURCE.out.verified,by:0),
            bundle,code,cacheDescriptor,cacheFiles,cacheHash)
        if (phase=='legacy') {
            R02_STRUCTURE_MATERIALIZE_ASSEMBLE(R02_STRUCTURE_MATERIALIZE_SLICE.out.partials.collect(),
                R02_STRUCTURE_MATERIALIZE_PLAN.out.planned,bundle,code)
        }
    } else {
        if (!safe(params.partials_manifest) || !(params.partials_manifest_sha256 ==~ /[0-9a-f]{64}/)) error 'Authenticated partial locations required'
        def descriptorBytes=java.nio.file.Files.readAllBytes(java.nio.file.Path.of(params.partials_manifest))
        def descriptorSha=java.security.MessageDigest.getInstance('SHA-256').digest(descriptorBytes).encodeHex().toString()
        if (descriptorSha != params.partials_manifest_sha256) error 'Partial locations SHA256 mismatch'
        def desc=new groovy.json.JsonSlurper().parseText(new String(descriptorBytes,'UTF-8'))
        if (desc.schema!='r02_structure_partial_locations_v1' || !desc.partials) error 'Invalid partial locations'
        def locations=[:]
        desc.partials.each { row ->
            if (!safe(row.path) || !(row.manifest_sha256 ==~ /[0-9a-f]{64}/)) error 'Unsafe partial descriptor'
            def directory=file(row.path,checkIfExists:true)
            def bytes=java.nio.file.Files.readAllBytes(directory.resolve('manifest.json'))
            if (java.security.MessageDigest.getInstance('SHA-256').digest(bytes).encodeHex().toString()!=row.manifest_sha256) error 'Partial receipt SHA256 mismatch'
            def receipt=new groovy.json.JsonSlurper().parseText(new String(bytes,'UTF-8'))
            def taskId=receipt.request.task_id
            if (locations.containsKey(taskId)) error 'Duplicate partial task'
            locations[taskId]=directory
        }
        def exclusionScope=params.exclusion_scope ?: contract.exclusion_scope
        if (!(exclusionScope in [contract.exclusion_scope]+(contract.additional_exclusion_scopes ?: []))) error 'Unregistered sensitivity'
        def blocks=R02_STRUCTURE_MATERIALIZE_PLAN.out.planned.flatMap { p ->
            def plan=new groovy.json.JsonSlurper().parse(new File(p.toString()+'/manifest.json'))
            if (locations.keySet()!=plan.tasks.keySet()) error 'Complete partial task coverage required'
            plan.blocks.collect { blockId,membership ->
                // The whole-chromosome sensitivity reuses all non-target-
                // chromosome features from the principal materialization.
                def targetChrom=plan.tasks[membership.target].chromosome
                def features=membership.features.findAll { t -> exclusionScope!='WHOLE_CHROMOSOME' || plan.tasks[t].chromosome!=targetChrom }
                tuple(blockId,(features+[membership.target]).collect { t -> locations[t] })
            }
        }
        R02_STRUCTURE_MATERIALIZE_ASSEMBLE_BLOCK(blocks,R02_STRUCTURE_MATERIALIZE_PLAN.out.planned,bundle,code,exclusionScope)
        R02_STRUCTURE_MATERIALIZE_INDEX(R02_STRUCTURE_MATERIALIZE_ASSEMBLE_BLOCK.out.shard.collect(),bundle,code,exclusionScope)
    }
}
