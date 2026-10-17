nextflow.enable.dsl=2
def sha(p){def d=java.security.MessageDigest.getInstance('SHA-256');p.toFile().withInputStream{s->new java.security.DigestInputStream(s,d).transferTo(java.io.OutputStream.nullOutputStream())};d.digest().encodeHex().toString()}
def json(p){new groovy.json.JsonSlurper().parse(p.toFile())}
process COMMON_PREPARE_ONLY {
 tag "${view}";cpus 1;memory '2 GB';time '30m';maxForks 4;cache 'deep'
 publishDir params.output_dir,mode:'copy',overwrite:false
 input: tuple val(view),path(request),val(rpin),path(admission),val(apin),path(source,stageAs:'source/*');path code
 output: tuple val(view),path("prepare_${view}")
 script:
 """
 umask 077
 export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
 python3 r02_common_search_prepare_only.py --request '${request}' --request-sha256 '${rpin}' \\
  --admission '${admission}' --admission-sha256 '${apin}' --source-directory source --output prepare_${view}
 """
 stub:
 """
 mkdir prepare_${view}
 touch prepare_${view}/statistics.json prepare_${view}/family.json prepare_${view}/manifest.json
 """
}
workflow {
 def b=json(file(params.batch,checkIfExists:true));if(b.schema!='r02_common_prepare_only_admitted_batch_v1'||b.status!='EXACT_87_PREPARES_ADMITTED'||b.view_count!=87||b.views.size()!=87||b.views.collect{it.id}.toSet().size()!=87||b.cpus_per_prepare!=1||b.memory_gb_per_prepare!=2||b.max_parallel_prepares!=4||b.fit_allowed!=false||b.select_allowed!=false||b.evaluate_allowed!=false)error 'exact prepare-only batch required'
 def prepareModule=file(params.prepare_module,checkIfExists:true);if(sha(prepareModule)!=b.prepare_module_sha256)error 'prepare module pin differs'
 def codeNames=[prepareModule]+b.implementation_sha256.collect{n,h->def p=file(params.code_dir+'/'+n,checkIfExists:true);if(sha(p)!=h)error 'frozen code pin differs';p}
 def jobs=b.views.collect{r->
  def q=file(r.request,checkIfExists:true);def a=file(r.admission,checkIfExists:true);if(sha(q)!=r.request_sha256||sha(a)!=r.admission_sha256)error 'request/admission differs'
  def x=json(q);def m=file(x.source.directory+'/manifest.json',checkIfExists:true);def mm=json(m);if(sha(m)!=x.source.manifest_sha256||mm.evaluate_targets_computed!=false)error 'source differs'
  def source=[m]+mm.outputs_sha256.collect{n,h->def p=file(x.source.directory+'/'+n,checkIfExists:true);if(sha(p)!=h)error 'source artifact differs';p}
  tuple(r.id,q,r.request_sha256,a,r.admission_sha256,source)}
 COMMON_PREPARE_ONLY(Channel.fromList(jobs),Channel.value(codeNames))
}
