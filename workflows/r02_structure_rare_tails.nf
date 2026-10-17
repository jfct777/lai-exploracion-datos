nextflow.enable.dsl=2

// Reuse the exact bounded indexed extraction process; this entry point only
// authenticates the rare-source descriptors and geometrically complete tails.
include { INDEXED_SOURCE_SLICE } from './r02_structure_source_slices'

workflow {
    def safe={s -> s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') && !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    def pinned={name,pin ->
        if(!safe(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe pinned rare-tail path required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'Rare-tail source changed before cache lookup'
        p
    }
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)}||
       params.network_source_reads_admitted!=true||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)||!(params.max_forks?.toString() ==~ /[1-9][0-9]*/)||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded rare-tail admission required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_structure_rare_tails_batch_v1'||!b.slices||b.slices.collect{it.chromosome}.toSet().size()!=b.slices.size()) error 'Unique rare-tail batch required'
    def sourceFile=pinned(b.bundle.path+'/materialize.json',b.bundle.sha256)
    def source=new groovy.json.JsonSlurper().parse(sourceFile.toFile())
    def code=b.implementation_sha256.collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe rare-tail consumer name'
        pinned(params.code_dir+'/'+n,h)
    }
    def jobs=b.slices.collect{d ->
        if(!(d.chromosome ==~ /[1-9]|1[0-9]|2[0-2]/)) error 'Rare-tail autosome required'
        def request=pinned(d.request,d.request_sha256)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema!='r02_structure_indexed_source_slice_v1'||r.implementation_sha256!=b.implementation_sha256['r02_structure_source_slice.py']||
           r.source.sha256!=source.sources[d.chromosome].rare_vcf.sha256) error 'Rare-tail source/code differs'
        def inputs=['index','regions','samples'].collect{k -> pinned(r[k].path,r[k].sha256)}
        def m=new groovy.json.JsonSlurper().parse(pinned(d.event_manifest.path,d.event_manifest.sha256).toFile())
        if(m.chromosome!=d.chromosome||m.source_contract_sha256!=b.bundle.sha256||m.audit.rare.outside_map!=d.expected_sites||
           m.audit.rare.source_rows!=m.audit.rare.retained_sites+d.expected_sites) error 'Rare-tail original catalogue differs'
        def expected=[]
        def low=m.channels.rare[0].start_bp-1
        def high=m.channels.rare[-1].end_bp+1
        if(low>=1) expected.add(source.sources[d.chromosome].contig+'\t1\t'+low)
        if(high<=source.sources[d.chromosome].length_bp) expected.add(source.sources[d.chromosome].contig+'\t'+high+'\t'+source.sources[d.chromosome].length_bp)
        if(file(r.regions.path).readLines()!=expected) error 'Rare-tail interval coverage differs'
        tuple(d.chromosome,request,d.request_sha256,inputs)
    }
    INDEXED_SOURCE_SLICE(Channel.fromList(jobs),code)
}
