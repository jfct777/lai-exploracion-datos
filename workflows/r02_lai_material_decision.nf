nextflow.enable.dsl=2
params.selected_chromosomes=[]

process R02_LAI_MATERIAL_DECISION {
    tag "chr${chrom}"
    cpus 1
    memory "${params.memory_gb} GB"
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(contract, stageAs:'decision_contract.json'), val(contract_hash), path(dependencies, stageAs:'authenticated_inputs/input????????')
    path code, stageAs:'r02_lai_material_decision.py'
    path joint_code, stageAs:'r02_lai_simulation_material.py'
    path support_code, stageAs:'r02_lai_support_review.py'
    output:
    tuple val(chrom), path('decision'), emit: decisions
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 r02_lai_material_decision.py chromosome --contract decision_contract.json --contract-sha256 '${contract_hash}' \
      --output-dir decision --min-free-gib ${params.min_free_gib} --timeout-seconds ${params.timeout_seconds} \
      --max-output-bytes ${params.max_output_bytes}
    """
}

process R02_LAI_MATERIAL_DECISION_AGGREGATE {
    cpus 1
    memory "${params.memory_gb} GB"
    cache 'deep'
    publishDir "${params.output_dir}/aggregate", mode:'copy', overwrite:false
    input:
    path decisions, stageAs:'decisions/chr??'
    path campaign, stageAs:'campaign.json'
    val campaign_hash
    path dependencies, stageAs:'campaign_inputs/input????????'
    path code, stageAs:'r02_lai_material_decision.py'
    path joint_code, stageAs:'r02_lai_simulation_material.py'
    path support_code, stageAs:'r02_lai_support_review.py'
    output:
    path 'aggregate', emit: aggregate
    script:
    def decisionList=decisions instanceof List ? decisions : [decisions]
    def receipts=decisionList.collect { d -> "--decision-manifest '${d}/manifest.json' --decision-sha256 \"\$(sha256sum '${d}/manifest.json' | cut -d ' ' -f1)\"" }.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 r02_lai_material_decision.py aggregate --campaign-contract campaign.json --campaign-contract-sha256 '${campaign_hash}' \
      ${receipts} --output-dir aggregate --min-free-gib ${params.min_free_gib} \
      --timeout-seconds ${params.timeout_seconds} --max-output-bytes ${params.max_output_bytes}
    """
}

workflow {
    def safe={ p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.campaign_contract,params.source_bin,params.output_dir].every { safe(it) }) error 'Absolute safe paths required'
    if (!(params.campaign_contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Campaign SHA256 required'
    if (!(params.max_forks?.toString() in ['1','2']) || !(params.memory_gb?.toString() in ['1','2'])) error 'Forks/memory must be 1 or 2'
    if (!(params.min_free_gib?.toString() ==~ /[0-9]+/) || !(params.timeout_seconds?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.max_output_bytes?.toString() ==~ /[1-9][0-9]*/)) error 'Invalid resource limits'
    def parse={ p -> new groovy.json.JsonSlurper().parse(file(p,checkIfExists:true).toFile()) }
    def descriptor={ d ->
        if (!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid descriptor'
        file(d.path,checkIfExists:true)
    }
    def campaign=parse(params.campaign_contract)
    if (campaign.schema!='r02_lai_material_decision_campaign_v1' || !(campaign.expected_chromosomes instanceof List) ||
        !campaign.expected_chromosomes || campaign.expected_chromosomes.toSet().size()!=campaign.expected_chromosomes.size()) error 'Invalid campaign'
    def seen=[]
    def jobs=campaign.jobs.collect { j ->
        def chrom=j.chromosome.toString()
        if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/) || seen.contains(chrom)) error 'Invalid/duplicate chromosome'
        seen.add(chrom)
        def contractPath=descriptor(j.contract); def c=parse(j.contract.path)
        if (c.schema!='r02_lai_material_decision_input_v1' || c.chromosome!=chrom) error 'Job/contract mismatch'
        def mp=descriptor(c.joint_manifest); def manifest=parse(c.joint_manifest.path)
        def sp=descriptor(c.material_input_contract); def source=parse(c.material_input_contract.path)
        def deps=[mp,sp,descriptor(c.publication_receipt),descriptor(c.roles)]
        manifest.outputs_sha256.keySet().each { name ->
            if (!(name in ['joint_material.tsv.gz','joint_summary.tsv','definitions.json'])) error 'Unsafe joint file'
            deps.add(file(mp.parent.resolve(name),checkIfExists:true))
        }
        deps.add(file(sp.parent.resolve('role_disjointness.json'),checkIfExists:true))
        if (source.roles.keySet()!=['REF_TRAIN','SOURCE_VALID'].toSet()) error 'Only reference/development donors permitted'
        source.roles.each { role, item ->
            def base=sp.parent.resolve(role)
            def sm=file(base.resolve('source/manifest.json'),checkIfExists:true)
            deps.add(sm); deps.add(file(base.resolve('source/map_support.json'),checkIfExists:true))
            def sourceManifest=parse(sm.toString())
            def panel=sourceManifest.input_files.panel_vcf.path
            if (!safe(panel)) error 'Unsafe role projection path'
            deps.add(file(panel,checkIfExists:true))
            item.review_sha256.keySet().each { name ->
                if (!(name in ['manifest.json','support_summary.tsv','correspondence_summary.tsv','definitions.json'])) error 'Unsafe review filename'
                deps.add(file(base.resolve('review').resolve(name),checkIfExists:true))
            }
        }
        tuple(chrom,contractPath,j.contract.sha256,deps.unique())
    }
    if (seen.toSet()!=campaign.expected_chromosomes.toSet()) error 'Incomplete campaign jobs'
    def code=file("${params.source_bin}/r02_lai_material_decision.py",checkIfExists:true)
    def jointCode=file("${params.source_bin}/r02_lai_simulation_material.py",checkIfExists:true)
    def supportCode=file("${params.source_bin}/r02_lai_support_review.py",checkIfExists:true)
    if (!(params.selected_chromosomes instanceof List) || params.selected_chromosomes.toSet().size()!=params.selected_chromosomes.size() ||
        !seen.containsAll(params.selected_chromosomes)) error 'Invalid selected chromosomes'
    def selected=params.selected_chromosomes ? jobs.findAll { params.selected_chromosomes.contains(it[0]) } : jobs
    R02_LAI_MATERIAL_DECISION(Channel.fromList(selected),code,jointCode,supportCode)
    def campaignDeps=[descriptor(campaign.completion)]+campaign.jobs.collect { descriptor(it.contract) }
    if (!params.selected_chromosomes) {
        R02_LAI_MATERIAL_DECISION_AGGREGATE(R02_LAI_MATERIAL_DECISION.out.decisions.map { c,p -> p }.collect(),
            file(params.campaign_contract,checkIfExists:true),params.campaign_contract_sha256,campaignDeps,code,jointCode,supportCode)
    }
}
