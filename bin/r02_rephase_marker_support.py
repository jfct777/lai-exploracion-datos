#!/usr/bin/env python3
"""Exact variant-axis support of an existing rephased VCF; never parse GT fields."""
import argparse,base64,csv,functools,gzip,hashlib,io,json,re,resource,sys,tarfile,time
from pathlib import Path
import numpy as np

SCHEMA_V2='r02_rephase_marker_support_v2'
SCHEMA_V3='r02_rephase_marker_support_v3'
SCHEMA=SCHEMA_V2
STRICT_REFERENCE='REQUIRE_GRCH38_DECLARATION'
UNVERIFIED_REFERENCE='REFERENCE_UNVERIFIED'
BASES=b'ACGT'
def require(ok,message):
 if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
 with Path(path).open('x')as out:json.dump(value,out,indent=2);out.write('\n')
def compressed_inventory(path,lines,max_bytes,deadline=None):
 """Write deterministic gzip while hashing both stored and tabular bytes."""
 content=hashlib.sha256();rows=0
 with Path(path).open('xb')as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as target:
   for line in lines:
    require(deadline is None or time.monotonic()<deadline,'Inventory audit time bound exceeded')
    require(line.endswith(b'\n'),'Inventory rows must be newline terminated');target.write(line);content.update(line);rows+=1
    require(raw.tell()<=max_bytes,'Compressed inventory byte bound exceeded')
 require(Path(path).stat().st_size<=max_bytes,'Compressed inventory byte bound exceeded')
 return dict(path=Path(path).name,bytes=Path(path).stat().st_size,sha256=sha(path),content_sha256=content.hexdigest(),lines=rows)
@functools.lru_cache(maxsize=22)
def aliases(chromosome):
 c=str(chromosome);require(c in {str(x)for x in range(1,23)},'Only declared autosomes supported');return frozenset((c.encode(),('chr'+c).encode()))
def hash_ids(ids):return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def roster(persons,crosswalk,manifest):
 with Path(persons).open()as stream:people=list(csv.DictReader(stream,delimiter='\t'))
 ids=[x['sample_id']for x in people];require(len(ids)==len(set(ids))==manifest['n_people']and hash_ids(ids)==manifest['sample_ids_sha256'],'Analytical people/order differ')
 roles={key:sum(x['role']==key for x in people)for key in manifest['role_counts']};require(roles==manifest['role_counts']and sum(roles.values())==len(ids),'Original roles differ')
 with Path(crosswalk).open()as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
 require(len(rows)==len(ids)and {x['sample_id']for x in rows}==set(ids),'Crosswalk does not cover exactly analytical people')
 mapping={x['sample_id']:x['source_id']for x in rows};sources=[mapping[x]for x in ids];require(all(sources)and len(set(sources))==len(ids),'Source mapping is not one-to-one')
 return sources,roles

def rare_axis(archive,manifest,*,chromosome,max_chunk_bytes):
 """Opaque chunk SHA first; open only position_bp/ref/alt NPZ arrays."""
 require(manifest['schema']=='r02_genomic_event_chromosome_v1'and manifest['chromosome']==str(chromosome),'Event chromosome/schema differ')
 expected={x['path']:x for x in manifest['channels']['rare']};require(len(expected)==len(manifest['channels']['rare'])and expected,'Rare chunk names duplicate/empty')
 positions=[];codes=[];seen=set();prefix='published/chr'+str(chromosome)+'/'
 with tarfile.open(archive,'r|gz')as source:
  for entry in source:
   if not entry.name.startswith(prefix):continue
   name=entry.name[len(prefix):]
   if name not in expected:continue
   record=expected[name];require(entry.isfile()and not entry.issym()and name not in seen and entry.size==record['bytes']and entry.size<=max_chunk_bytes,'Rare chunk inventory/bound differs')
   raw=source.extractfile(entry).read();require(hashlib.sha256(raw).hexdigest()==record['sha256'],'Rare chunk SHA differs')
   with np.load(io.BytesIO(raw),allow_pickle=False)as data:
    p=data['position_bp'];r=data['ref'];a=data['alt']
    require(p.ndim==1 and len(p)==record['n_sites']and p.dtype.kind in 'iu'and r.dtype.kind in 'iu'and a.dtype.kind in 'iu'and r.shape==a.shape==p.shape,'Invalid axis arrays')
    require(np.all(p>0)and np.all(r>=0)and np.all(a>=0)and np.all(r<4)and np.all(a<4)and np.all(r!=a),'Invalid positions or alleles')
    positions.append(p.astype(np.int64));codes.append((r*4+a).astype(np.uint8))
   seen.add(name)
 require(seen==set(expected),'Missing rare chunks')
 p=np.concatenate(positions);a=np.concatenate(codes);order=np.argsort(p,kind='stable');p=p[order];a=a[order]
 require(len(p)==manifest['audit']['rare']['retained_sites']and np.all(np.diff(p)>0),'Mapped rare axis differs or positions repeat')
 return p,a,len(seen)

def marker(line,chromosome):
 # The sixth field is an opaque tail. FORMAT and per-person genotype tokens
 # are not split, decoded, counted or used by this audit.
 fields=line.split(b'\t',5);require(len(fields)==6 and fields[0]in aliases(chromosome),'Wrong contig or truncated marker')
 ref,alt=fields[3],fields[4];require(len(ref)==len(alt)==1 and ref in (b'A',b'C',b'G',b'T')and alt in (b'A',b'C',b'G',b'T')and ref!=alt,'Non-biallelic-SNV source marker')
 position=int(fields[1]);require(position>0,'Nonpositive source position')
 return position,BASES.index(ref)*4+BASES.index(alt),fields[0].decode()

def audit_stream(stream,p,a,*,chromosome,source_bytes,source_md5,expected_sources,max_line_bytes,seconds_limit,
 reference_validation_mode=STRICT_REFERENCE,inventory_dir=None,max_inventory_bytes=None):
 started=time.monotonic();whole=hashlib.sha256();md5=hashlib.md5();header=hashlib.sha256();axis=hashlib.sha256();total=0;rows=0;exact=0;same=0;last=0;header_seen=False;source_names=None;raw_contigs=set();declared_build=False;contig_header=set()
 require(reference_validation_mode in (STRICT_REFERENCE,UNVERIFIED_REFERENCE),'Unknown reference validation mode')
 inventory_enabled=reference_validation_mode==UNVERIFIED_REFERENCE
 require(not inventory_enabled or inventory_dir is not None and isinstance(max_inventory_bytes,int)and max_inventory_bytes>0,'Explicit inventory directory/byte bound required')
 reference_declarations=[];support=np.zeros(len(p),dtype=np.uint8);source_codes=np.full(len(p),255,dtype=np.uint8)
 source_inventory=None;source_gzip=None;source_content=hashlib.sha256();source_inventory_rows=0
 if inventory_enabled:
  inventory_dir=Path(inventory_dir);source_inventory=(inventory_dir/'source-marker-axis.tsv.gz').open('xb')
  source_gzip=gzip.GzipFile(filename='',mode='wb',fileobj=source_inventory,mtime=0)
  inventory_header=b'CHROM\tPOS\tREF\tALT\n';source_gzip.write(inventory_header);source_content.update(inventory_header);source_inventory_rows=1
 require(len(p)==len(a)and np.all(np.diff(p)>0),'Query axis must be unique and increasing')
 try:
  while True:
   line=stream.readline(max_line_bytes+1)
   if not line:break
   require(len(line)<=max_line_bytes and line.endswith(b'\n'),'Oversized/truncated source line')
   total+=len(line);require(total<=source_bytes and time.monotonic()-started<seconds_limit,'Source byte/time bound exceeded');whole.update(line);md5.update(line)
   if line.startswith(b'#'):
    require(rows==0,'Header after marker rows');header.update(line)
    if line.startswith(b'##reference='):
     require(not reference_declarations,'Multiple reference declarations')
     declaration=line.rstrip(b'\r\n').split(b'=',1)[1];reference_declarations.append(declaration.decode(errors='replace'))
     declared_build=bool(re.search(br'(?i)(GRCh38|Homo_sapiens_assembly38)',declaration))
     require(declared_build,'Contradictory or unsupported reference declaration')
    if line.startswith(b'##contig=<ID='):
     value=line.split(b'=',2)[2].split(b',',1)[0].split(b'>',1)[0]
     if value in aliases(chromosome):contig_header.add(value.decode())
    if line.startswith(b'#CHROM\t'):
     require(not header_seen,'Repeated VCF header');fields=line.rstrip(b'\r\n').split(b'\t');require(fields[:9]==[b'#CHROM',b'POS',b'ID',b'REF',b'ALT',b'QUAL',b'FILTER',b'INFO',b'FORMAT'],'VCF columns differ')
     source_names=[x.decode()for x in fields[9:]];require(len(source_names)==len(set(source_names))and set(expected_sources)<=set(source_names),'Source people absent/duplicated');header_seen=True
    continue
   require(header_seen and contig_header,'VCF header or chromosome declaration absent')
   require(reference_validation_mode==UNVERIFIED_REFERENCE or declared_build,'GRCh38 reference declaration absent')
   position,code,raw=marker(line,chromosome);require(position>last,'Source positions duplicate or are unsorted at '+str(position));last=position;rows+=1;raw_contigs.add(raw)
   inventory_line=(raw+'\t'+str(position)+'\t'+chr(BASES[code//4])+'\t'+chr(BASES[code%4])+'\n').encode()
   axis.update(('chr'+str(chromosome)+'\t'+str(position)+'\t'+chr(BASES[code//4])+'\t'+chr(BASES[code%4])+'\n').encode())
   if source_gzip:
    source_gzip.write(inventory_line);source_content.update(inventory_line);source_inventory_rows+=1
    require(source_inventory.tell()<=max_inventory_bytes,'Compressed inventory byte bound exceeded')
   idx=int(np.searchsorted(p,position))
   if idx<len(p)and p[idx]==position:
    same+=1;source_codes[idx]=code
    if a[idx]==code:exact+=1;support[idx]=1
    else:support[idx]=2
 finally:
  if source_gzip:source_gzip.close()
  if source_inventory:source_inventory.close()
 require(rows>0 and total==source_bytes and base64.b64encode(md5.digest()).decode()==source_md5,'Whole-stream size/MD5 mismatch or empty source')
 result=dict(schema=SCHEMA_V3 if inventory_enabled else SCHEMA_V2,status='COMPLETE_DESCRIPTIVE_MARKER_SUPPORT_REFERENCE_UNVERIFIED_NO_GT_PARSED'if inventory_enabled else'COMPLETE_EXACT_MARKER_SUPPORT_NO_GT_PARSED',chromosome=str(chromosome),source_bytes=total,source_sha256=whole.hexdigest(),source_md5_base64=source_md5,source_header_sha256=header.hexdigest(),source_marker_axis_sha256=axis.hexdigest(),source_marker_count=rows,source_person_count=len(source_names),source_person_order_sha256=hash_ids(source_names),analytical_people_present=len(expected_sources),mapped_rare_markers=len(p),same_position=same,exact_ref_alt_intersection=exact,position_matches_with_ref_alt_difference=same-exact,mapped_rare_absent_exact=len(p)-exact,raw_row_contigs=sorted(raw_contigs),raw_header_contigs=sorted(contig_header),normalized_contig='chr'+str(chromosome),genome_build_declared='GRCh38'if declared_build else None,npz_arrays_opened=['position_bp','ref','alt'],vcf_fields_used=['CHROM','POS','REF','ALT'],genotype_fields_parsed=False,source_people_presence_only_in_header=True,individual_genotype_completeness_audited=False,phase_accuracy_validated=False,allele_ancestry_attribution=False,seconds_stream=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 if inventory_enabled:
  source_path=inventory_dir/'source-marker-axis.tsv.gz';require(source_path.stat().st_size<=max_inventory_bytes,'Compressed inventory byte bound exceeded')
  def rare_lines():
   yield b'CHROM\tPOS\tREF\tALT\tSUPPORT_STATUS\tSOURCE_REF\tSOURCE_ALT\n'
   labels=(b'POSITION_ABSENT',b'EXACT_REF_ALT',b'SAME_POSITION_REF_ALT_DIFFERENT')
   for position,code,state,source_code in zip(p,a,support,source_codes):
    source_ref=b'.'if source_code==255 else bytes((BASES[source_code//4],));source_alt=b'.'if source_code==255 else bytes((BASES[source_code%4],))
    yield b'chr'+str(chromosome).encode()+b'\t'+str(int(position)).encode()+b'\t'+bytes((BASES[code//4],))+b'\t'+bytes((BASES[code%4],))+b'\t'+labels[state]+b'\t'+source_ref+b'\t'+source_alt+b'\n'
  rare_descriptor=compressed_inventory(inventory_dir/'rare-marker-support.tsv.gz',rare_lines(),max_inventory_bytes-source_path.stat().st_size,started+seconds_limit)
  require(time.monotonic()-started<seconds_limit,'Inventory audit time bound exceeded')
  result.update(reference_validation_mode=reference_validation_mode,reference_declaration_present=bool(reference_declarations),reference_declarations=reference_declarations,mapped_rare_exact_ref_alt=exact,mapped_rare_same_position_ref_alt_difference=same-exact,mapped_rare_position_absent=len(p)-same,inventories={'source_marker_axis':dict(path=source_path.name,bytes=source_path.stat().st_size,sha256=sha(source_path),content_sha256=source_content.hexdigest(),lines=source_inventory_rows),'rare_marker_support':rare_descriptor},inventory_compressed_bytes=source_path.stat().st_size+rare_descriptor['bytes'],limits={'source_bytes':source_bytes,'max_line_bytes':max_line_bytes,'stream_seconds_limit':seconds_limit,'max_inventory_bytes':max_inventory_bytes},seconds_total_inventory_audit=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 return result

def portable_axis_identity(spec):
 def clean(value):
  if isinstance(value,dict):return {k:clean(v)for k,v in value.items()if k not in ('path','local_path')}
  if isinstance(value,list):return [clean(v)for v in value]
  return value
 return clean({key:spec[key]for key in ('chromosome','events_manifest','persons','crosswalk','events_archive','max_chunk_bytes')})

def admit_prepared_axis(prepared,current_spec,current_spec_sha):
 if prepared['spec_sha256']==current_spec_sha:return dict(mode='SAME_SPEC',original_spec_sha256=current_spec_sha)
 reuse=current_spec.get('axis_reuse');require(current_spec['schema']=='r02_rephase_marker_support_input_v2'and isinstance(reuse,dict)and reuse.get('mode')=='AUTHENTICATED_EQUIVALENT_AXIS_INPUTS','Prepared axis belongs to a different spec')
 original_record=reuse['original_spec'];script_record=reuse['original_script'];require(sha(original_record['path'])==original_record['sha256']==prepared['spec_sha256'],'Original axis spec authentication differs')
 original=read(original_record['path']);require(sha(script_record['path'])==script_record['sha256']==original['script_sha256'],'Original axis reader authentication differs')
 require(portable_axis_identity(original)==portable_axis_identity(current_spec),'Axis-producing inputs or bounds differ')
 return dict(mode=reuse['mode'],original_spec_sha256=original_record['sha256'],original_script_sha256=script_record['sha256'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['axis','stream']);ap.add_argument('--spec',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--axis',type=Path);args=ap.parse_args();spec=read(args.spec)
 require(spec['schema']in ('r02_rephase_marker_support_input_v1','r02_rephase_marker_support_input_v2')and spec['script_sha256']==sha(__file__),'Frozen input/code differs')
 mode=STRICT_REFERENCE if spec['schema']=='r02_rephase_marker_support_input_v1'else spec.get('reference_validation_mode')
 require(mode==STRICT_REFERENCE if spec['schema']=='r02_rephase_marker_support_input_v1'else mode==UNVERIFIED_REFERENCE,'Reference mode/input schema combination differs')
 for key in ['events_manifest','persons','crosswalk']:
  require(sha(spec[key]['path'])==spec[key]['sha256'],'Input SHA differs: '+key)
 manifest=read(spec['events_manifest']['path']);sources,roles=roster(spec['persons']['path'],spec['crosswalk']['path'],manifest)
 args.output.mkdir(exist_ok=False)
 if args.stage=='axis':
  archive=Path(spec['events_archive']['local_path']);require(archive.stat().st_size==spec['events_archive']['bytes']and sha(archive)==spec['events_archive']['sha256'],'Event archive differs')
  started=time.monotonic();p,a,chunks=rare_axis(archive,manifest,chromosome=spec['chromosome'],max_chunk_bytes=spec['max_chunk_bytes'])
  np.savez(args.output/'rare-axis.npz',position_bp=p,allele_code=a)
  result=dict(schema=SCHEMA_V3 if mode==UNVERIFIED_REFERENCE else SCHEMA_V2,stage='METADATA_AXES_ONLY',chromosome=spec['chromosome'],mapped_rare_markers=len(p),rare_chunks=chunks,axis_sha256=sha(args.output/'rare-axis.npz'),spec_sha256=sha(args.spec),npz_arrays_opened=['position_bp','ref','alt'],dosage_or_observed_opened=False,roles=roles,seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 else:
  require(args.axis is not None,'Prepared axis required');prepared=read(args.axis/'manifest.json');reuse=admit_prepared_axis(prepared,spec,sha(args.spec));require(prepared['axis_sha256']==sha(args.axis/'rare-axis.npz'),'Prepared axis differs')
  with np.load(args.axis/'rare-axis.npz',allow_pickle=False)as data:p=data['position_bp'];a=data['allele_code']
  result=audit_stream(sys.stdin.buffer,p,a,chromosome=spec['chromosome'],source_bytes=spec['rephased_vcf']['bytes'],source_md5=spec['rephased_vcf']['md5_base64'],expected_sources=sources,max_line_bytes=spec['max_line_bytes'],seconds_limit=spec['stream_seconds_limit'],reference_validation_mode=mode,inventory_dir=args.output,max_inventory_bytes=spec.get('max_inventory_bytes'))
  result.update(spec_sha256=sha(args.spec),axis_manifest_sha256=sha(args.axis/'manifest.json'),axis_reuse_admission=reuse,source_uri=spec['rephased_vcf']['uri'],source_generation=spec['rephased_vcf'].get('generation'),roles=roles)
 save(args.output/'manifest.json',result);print(json.dumps(result))
if __name__=='__main__':main()
