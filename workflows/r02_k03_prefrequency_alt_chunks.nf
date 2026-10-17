nextflow.enable.dsl=2

def safeAbsolute(s) { s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') && !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
def fileHash(p) { def md=java.security.MessageDigest.getInstance('SHA-256'); p.toFile().withInputStream { x -> new java.security.DigestInputStream(x,md).transferTo(java.io.OutputStream.nullOutputStream()) }; md.digest().encodeHex().toString() }
def pinned(name,pin) { if (!safeAbsolute(name)||!(pin==~/[a-f0-9]{64}/)) error 'Safe absolute pinned file required'; def p=file(name,checkIfExists:true); if (!p.toFile().isFile()||fileHash(p)!=pin) error 'Pinned file absent or changed'; p }
def readJson(p) { new groovy.json.JsonSlurper().parse(p.toFile()) }

process K03_PREFREQUENCY_CHROMOSOME {
  tag "chr${chromosome}"; cpus 1; memory "${params.memory_gb} GB"; time params.chromosome_time
  maxForks params.max_forks; cache 'deep'; container params.container
  input: tuple val(chromosome), path(campaign,stageAs:'campaign.json'), path(roster,stageAs:'persons.tsv'), path(vcf,stageAs:'source.vcf.gz')
  path code
  output: tuple val(chromosome), path('chromosome')
  script:
  """
  umask 077
  export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python3 r02_k03_prefrequency_alt_chunks.py chromosome --campaign campaign.json --chromosome ${chromosome} --source-vcf source.vcf.gz --output chromosome
  """
  stub: "mkdir -p chromosome/chunks; touch chromosome/STUB_NOT_SCIENTIFIC_OUTPUT"
}

process K03_PREFREQUENCY_ASSEMBLE {
  cpus 1; memory '2 GB'; time '2h'; cache 'deep'; container params.container
  publishDir params.output_dir, mode:'copy', overwrite:false
  input: path campaign,stageAs:'campaign.json'; path roster,stageAs:'persons.tsv'; path chromosomeDirs,stageAs:'chromosome??'; path code
  output: path 'shared_bundle'
  script:
  """
  umask 077
  python3 r02_k03_prefrequency_alt_chunks.py assemble --campaign campaign.json \
    ${chromosomeDirs.collect { '--chromosome-dir '+it }.join(' ')} --output shared_bundle
  """
  stub: "mkdir shared_bundle; touch shared_bundle/STUB_NOT_SCIENTIFIC_OUTPUT"
}

workflow {
  if (![params.campaign,params.code_dir,params.output_dir].every { safeAbsolute(it) } || !(params.container==~/[^\s]+@sha256:[a-f0-9]{64}/)) error 'Pinned campaign, code, output and container required'
  def campaign=pinned(params.campaign,params.campaign_sha256); def value=readJson(campaign)
  if (value.schema!='r02_k03_prefrequency_campaign_v1'||value.sources.size()!=22||value.sources*.chromosome.sort()!=((1..22) as List)) error 'Exact 22-autosome pre-frequency campaign required'
  def rosterPath = safeAbsolute(value.roster.path) ? value.roster.path : new File(params.campaign).parent + '/' + value.roster.path
  def roster=pinned(rosterPath,value.roster.sha256)
  def codeFiles=['r02_k03_prefrequency_alt_chunks.py','r02_k03_inner_fit_contract.py','r02_k03_inner_fit_affinities.py','r02_projection_calibration_split.py','r02_genomic_pair_evidence.py','r02_structure_affinity_io.py','r02_structure_continuous.py','r02_structure_discrete.py','r02_structure_discrete_reuse.py','r02_structure_increment.py','r02_structure_input_plan.py','r02_structure_materialize.py','r02_structure_rare_inputs.py'].collect { file(params.code_dir+'/'+it,checkIfExists:true) }
  def jobs=value.sources.collect { row -> tuple(row.chromosome,campaign,roster,pinned(row.vcf.path,row.vcf.sha256)) }
  K03_PREFREQUENCY_CHROMOSOME(Channel.fromList(jobs),codeFiles)
  K03_PREFREQUENCY_ASSEMBLE(campaign,roster,K03_PREFREQUENCY_CHROMOSOME.out.map{ c,d->d }.collect(),codeFiles)
}
