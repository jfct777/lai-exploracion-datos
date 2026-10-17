nextflow.enable.dsl=2
params.selected_chromosomes=[]

process R02_LAI_DONOR_HAPLOTYPES {
    tag "chr${chrom}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(contract, stageAs:'contract.json'), val(digest), path(dependencies, stageAs:'inputs/input????????')
    path code_files
    output:
    tuple val(chrom), path('private_preparation'), emit: preparations
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_donor_haplotypes.py chromosome --contract contract.json --contract-sha256 '${digest}' \
      --output-dir private_preparation --cpus ${task.cpus}
    """
}

process R02_LAI_DONOR_HAPLOTYPES_AGGREGATE {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir "${params.output_dir}/aggregate", mode:'copy', overwrite:false
    input:
    path preparations, stageAs:'preparations/chr??'
    path campaign, stageAs:'campaign.json'
    val campaign_hash
    path contracts, stageAs:'contracts/input????????'
    path code_files
    output:
    path 'private_aggregate', emit: aggregate
    script:
    def items=preparations instanceof List ? preparations : [preparations]
    def arguments=items.collect { p -> "--manifest '${p}/manifest.json'" }.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_donor_haplotypes.py aggregate --campaign campaign.json --campaign-sha256 '${campaign_hash}' \
      ${arguments} --output-dir private_aggregate
    """
}

workflow {
    def safe={ p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.campaign_contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if (!(params.campaign_contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Campaign SHA256 required'
    if (!(params.max_forks?.toString() in ['1','2']) || !(params.memory_gb?.toString() in ['1','2'])) error 'Memory/forks must be1or2'
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit task time required'
    def parse={p->new groovy.json.JsonSlurper().parse(file(p,checkIfExists:true).toFile())}
    def descriptor={d->
        if (!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid input descriptor'
        file(d.path,checkIfExists:true)
    }
    def campaign=parse(params.campaign_contract)
    if (campaign.schema!='r02_lai_donor_haplotypes_campaign_v1' || !(campaign.jobs instanceof List) || !campaign.jobs) error 'Invalid campaign'
    def seen=[]
    def jobs=campaign.jobs.collect{j->
        def chrom=j.chromosome.toString()
        if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/) || seen.contains(chrom)) error 'Invalid/duplicate chromosome'
        seen.add(chrom)
        def cp=descriptor(j.contract);def c=parse(j.contract.path)
        if (c.schema!='r02_lai_donor_haplotypes_input_v1' || c.chromosome!=chrom || c.resources.cpus!=1) error 'Invalid job or reader CPU count'
        def dp=descriptor(c.decision_manifest);def d=parse(c.decision_manifest.path)
        def sp=descriptor(c.donor_manifest);def s=parse(c.donor_manifest.path)
        if (s.genotype_contract.panel_role!='SOURCE_VALID') error 'Reserved donor role'
        def panel=s.input_files.panel_vcf.path
        if (!safe(panel)) error 'Unsafe projected panel path'
        def deps=[dp,sp,descriptor(c.roles),file(panel,checkIfExists:true),
          file(dp.parent.resolve('candidate_alleles.tsv.gz'),checkIfExists:true),
          file(dp.parent.resolve('role_support.json'),checkIfExists:true)]
        tuple(chrom,cp,j.contract.sha256,deps)
    }
    if (!(params.selected_chromosomes instanceof List) || params.selected_chromosomes.toSet().size()!=params.selected_chromosomes.size() || !seen.containsAll(params.selected_chromosomes)) error 'Invalid selected chromosomes'
    def names=['r02_lai_donor_haplotypes.py','r02_lai_material_decision.py','r02_lai_simulation_material.py',
      'r02_lai_support_review.py','r02_lai_genotype_support.py','r02_lai_allele_support.py',
      'r02_lai_build_verification.py','r02_genomic_pair_evidence.py']
    def code=names.collect{file("${params.source_bin}/${it}",checkIfExists:true)}
    def selected=params.selected_chromosomes ? jobs.findAll{params.selected_chromosomes.contains(it[0])}:jobs
    if (!params.selected_chromosomes && seen.toSet()!=(1..22).collect{it.toString()}.toSet()) error 'Full run requires22autosomes; use selected_chromosomes for a pilot'
    R02_LAI_DONOR_HAPLOTYPES(Channel.fromList(selected),code)
    if (!params.selected_chromosomes) {
        R02_LAI_DONOR_HAPLOTYPES_AGGREGATE(R02_LAI_DONOR_HAPLOTYPES.out.preparations.map{c,p->p}.collect(),
          file(params.campaign_contract,checkIfExists:true),params.campaign_contract_sha256,
          campaign.jobs.collect{descriptor(it.contract)},code)
    }
}
