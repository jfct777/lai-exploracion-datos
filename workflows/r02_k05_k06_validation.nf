nextflow.enable.dsl=2

process K05_PAIRED_GATE {
  cpus 1; memory '1 GB'; time '30m'; cache 'deep'
  publishDir "${params.output_dir}/k05", mode:'copy', overwrite:false
  input: path contract; val digest; path code
  output: path 'k05-result'
  script:
  """
  umask 077
  python3 ${code} --contract ${contract} --contract-sha256 '${digest}' --output-dir k05-result
  """
}

process K06_OBSERVABILITY {
  cpus 1; memory '2 GB'; time '2h'; cache 'deep'
  publishDir "${params.output_dir}/k06", mode:'copy', overwrite:false
  input: path contract; val digest; path code; path inputs, stageAs:'inputs/input????????'
  output: path 'k06-result'
  script:
  """
  umask 077
  python3 ${code} --contract ${contract} --contract-sha256 '${digest}' --output-dir k06-result
  """
}

workflow {
  def safe={p->p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
  if(![params.k05_contract,params.k06_contract,params.source_bin,params.output_dir].every{safe.call(it)}) error 'Safe absolute paths required'
  def hash={p->java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString()}
  def k05=file(params.k05_contract,checkIfExists:true)
  def k06=file(params.k06_contract,checkIfExists:true)
  def k05code=file(params.source_bin+'/r02_k05_paired_comparison.py',checkIfExists:true)
  def k06code=file(params.source_bin+'/r02_k06_observability.py',checkIfExists:true)
  def c=new groovy.json.JsonSlurper().parse(k06.toFile())
  def files=[file(c.people.path,checkIfExists:true)]
  c.chromosomes.values().each{x->files.addAll([file(x.genetic_map.path),file(x.segments_contract.path),file(x.segments.path),file(x.reference_provenance.path)])}
  if(c.calibration){files.add(file(c.calibration.truth.path));files.add(file(c.calibration.detections.path))}
  K05_PAIRED_GATE(k05,hash.call(k05),k05code)
  K06_OBSERVABILITY(k06,hash.call(k06),k06code,files.unique())
}
