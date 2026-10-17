#!/usr/bin/env python3
"""Dose concordance at every exact rare-catalogue/rephased-VCF intersection.

Reads an authenticated event-chunk tar sequentially and a complete VCF on
stdin. Only exact CHROM/POS/REF/ALT rows cause GT decoding. Stored dosage is for
the frozen source-minor allele; VCF ALT dosage is complemented when REF is the
counted allele. Results contain aggregate locus/role counts, never identities or
individual genotypes. Separators are descriptive and do not validate phase.
"""
import argparse,base64,csv,gzip,hashlib,io,json,re,resource,shutil,sys,tarfile,time,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import r02_rephase_marker_support as support

SCHEMA='r02_rephase_genotype_concordance_input_v1';OUT='r02_rephase_genotype_concordance_v1';BASES=b'ACGT';MISSING=255
def require(x,m):support.require(x,m)
def read(p):return json.loads(Path(p).read_text())
def receipt(d):
 p=Path(d['path']);require(p.is_file()and support.sha(p)==d['sha256'],'Input SHA256 differs: '+str(p));return p
def snapshot(paths):return {str(p.resolve()):(p.stat().st_size,support.sha(p))for p in paths}
def unchanged(before):
 for name,value in before.items():
  p=Path(name);require((p.stat().st_size,support.sha(p))==tuple(value),'Input changed: '+name)

def load_protocol(path,digest):
 require(support.sha(path)==digest,'Protocol SHA256 differs');c=read(path)
 keys={'schema','script_sha256','chromosome','events_manifest','events_archive','persons','crosswalk','marker_support','rephased_vcf','limits'}
 require(set(c)==keys and c['schema']==SCHEMA and c['script_sha256']==support.sha(__file__),'Protocol schema/code differs')
 chrom=str(c['chromosome']);require(chrom in {str(i)for i in range(1,23)}and c['chromosome']==chrom,'Autosome string required')
 for k in ('events_manifest','events_archive','persons','crosswalk','marker_support'):
  require(set(c[k])>={'path','sha256'},'Incomplete descriptor: '+k)
 r=c['rephased_vcf'];require(set(r)=={'bytes','md5_base64','sha256'}and all(type(r[x])is int and r[x]>0 for x in ('bytes',))and re.fullmatch(r'[0-9a-f]{64}',r['sha256'])and r['md5_base64'],'Invalid VCF receipt')
 lim=c['limits'];require(set(lim)=={'max_chunk_bytes','max_array_bytes','max_line_bytes','max_rows_per_position','max_output_bytes','min_free_bytes','timeout_seconds'}and all(type(v)is int and v>0 for v in lim.values())and lim['max_array_bytes']<=1024**3,'Explicit positive limits and <=1 GiB decoded chunk required')
 return c

def roster(persons,crosswalk,manifest):
 sources,role_counts=support.roster(persons,crosswalk,manifest)
 with Path(persons).open()as f:rows=list(csv.DictReader(f,delimiter='\t'))
 require(len(rows)==len(sources)and all(r['role']in role_counts for r in rows),'Roster roles differ')
 return sources,np.asarray([r['role']for r in rows]),role_counts

class VCF:
 def __init__(self,stream,c,source_ids):self.s=stream;self.c=c;self.ids=source_ids;self.sha=hashlib.sha256();self.md5=hashlib.md5();self.nbytes=0;self.rows=0;self.samples=None;self.selected=None;self.pending=None;self.seen_header=False
 def line(self):
  x=self.s.readline(self.c['limits']['max_line_bytes']+1)
  require(len(x)<=self.c['limits']['max_line_bytes']and(not x or x.endswith(b'\n')),'VCF line oversized/truncated')
  self.nbytes+=len(x);require(self.nbytes<=self.c['rephased_vcf']['bytes'],'VCF byte bound exceeded');self.sha.update(x);self.md5.update(x);return x
 def header(self):
  while True:
   x=self.line();require(x,'VCF header absent')
   if x.startswith(b'##'):continue
   require(x.startswith(b'#CHROM\t'),'VCF column header absent');f=x.rstrip(b'\r\n').split(b'\t')
   require(f[:9]==[b'#CHROM',b'POS',b'ID',b'REF',b'ALT',b'QUAL',b'FILTER',b'INFO',b'FORMAT'],'VCF fixed columns differ')
   names=[v.decode()for v in f[9:]];require(len(names)==len(set(names)),'Duplicate VCF samples');index={v:i for i,v in enumerate(names)}
   require(set(self.ids)<=set(names),'Analytical sources absent from VCF');self.samples=names;self.selected=np.asarray([9+index[x]for x in self.ids]);self.seen_header=True;return
 def groups(self):
  if not self.seen_header:self.header()
  current=None;group=[];last=0
  while True:
   x=self.line()
   if not x:break
   require(not x.startswith(b'#'),'Header after VCF records');require(x.count(b'\t')==8+len(self.samples),'VCF field count differs from header');head=x.split(b'\t',5);require(len(head)==6 and head[0]in support.aliases(self.c['chromosome'])and head[1].isdigit(),'Malformed/wrong-contig VCF row')
   pos=int(head[1]);require(pos>=last and pos>0,'VCF positions decrease/nonpositive');last=pos;self.rows+=1
   if current is not None and pos!=current:yield current,group;group=[]
   current=pos;group.append(x);require(len(group)<=self.c['limits']['max_rows_per_position'],'Repeated-position bound exceeded')
  if group:yield current,group
  require(self.nbytes==self.c['rephased_vcf']['bytes']and self.sha.hexdigest()==self.c['rephased_vcf']['sha256']and base64.b64encode(self.md5.digest()).decode()==self.c['rephased_vcf']['md5_base64'],'Whole VCF size/hash/MD5 differs')

def decode_gt(line,selected):
 f=line.rstrip(b'\r\n').split(b'\t');fmt=f[8].split(b':');require(b'GT'in fmt,'FORMAT_GT_ABSENT')
 gi=fmt.index(b'GT');tokens=[f[i]for i in selected];dose=np.full(len(tokens),MISSING,np.uint8);cats=np.zeros(len(tokens),np.uint8);sep=np.zeros(len(tokens),np.uint8)
 # Common GT-first fixed-width route uses vector comparisons, not one parser call per person.
 if gi==0 and all(len(x)>=3 and x[1:2]in(b'|',b'/')and(len(x)==3 or x[3:4]==b':')for x in tokens):
  gt=np.frombuffer(b''.join(x[:3]for x in tokens),dtype='S3');sep[np.char.find(gt,b'|')>=0]=1;sep[np.char.find(gt,b'/')>=0]=2
  for key,val in ((b'0|0',0),(b'0/0',0),(b'0|1',1),(b'1|0',1),(b'0/1',1),(b'1/0',1),(b'1|1',2),(b'1/1',2)):dose[gt==key]=val
  partial=np.char.find(gt,b'.')>=0;cats[partial]=1;cats[(dose==MISSING)&~partial]=2
 else:
  for i,token in enumerate(tokens):
   pieces=token.split(b':');gt=pieces[gi]if gi<len(pieces)else b'.'
   if b'.'in gt:cats[i]=1;continue
   if len(gt)!=3 or gt[1:2]not in(b'|',b'/')or gt[0:1]not in(b'0',b'1')or gt[2:3]not in(b'0',b'1'):cats[i]=2;continue
   dose[i]=(gt[0]-48)+(gt[2]-48);sep[i]=1 if gt[1:2]==b'|'else 2
 return dose,cats,sep

def load_chunk(raw,record,n_people,max_bytes,max_array_bytes):
 require(len(raw)==record['bytes']and len(raw)<=max_bytes and hashlib.sha256(raw).hexdigest()==record['sha256'],'Corrupt/oversized rare chunk')
 with zipfile.ZipFile(io.BytesIO(raw))as archive:require(sum(x.file_size for x in archive.infolist())<=max_array_bytes,'Decoded NPZ exceeds max-array-bytes')
 with np.load(io.BytesIO(raw),allow_pickle=False)as z:
  need={'position_bp','ref','alt','counted_allele','dosage','observed'};require(need<=set(z.files),'Rare chunk arrays absent')
  p=z['position_bp'];ref=z['ref'];alt=z['alt'];ca=z['counted_allele'];d=z['dosage'];o=z['observed']
  require(p.shape==ref.shape==alt.shape==ca.shape==(record['n_sites'],)and d.shape==o.shape==(n_people,record['n_sites'])and d.dtype==np.uint8 and o.dtype==np.bool_,'Rare chunk axes/dtypes differ')
  require(np.all(np.diff(p)>0)and np.all((ref<4)&(alt<4)&(ref!=alt))and np.all((ca==0)|(ca==1))and np.all(d[o]<=2)and np.all(d[~o]==255),'Rare chunk values differ')
  return tuple(np.array(x,copy=True)for x in(p,ref,alt,ca,d,o))

FIELDS=['CHROM','POS','REF','ALT','COUNTED_ALLELE','ROLE','RARE_OBSERVED','VCF_OBSERVED','JOINT_OBSERVED','DOSAGE_MATCH','DOSAGE_MISMATCH','VCF_HETEROZYGOUS','VCF_HOMOZYGOUS_REF','VCF_HOMOZYGOUS_ALT','RARE_CARRIER','VCF_COUNTED_CARRIER','CARRIER_PRESENCE_MISMATCH','VCF_PARTIAL_OR_COMPLETE_MISSING','VCF_UNSUPPORTED_NONBIALLELIC_OR_PLOIDY','VCF_PHASED_PIPE','VCF_UNPHASED_SLASH']
def locus_rows(chrom,pos,ref,alt,counted,rare_d,rare_o,vcf_d,cats,seps,roles):
 out=[]
 for role in [*sorted(set(roles)), 'ALL']:
  mask=np.ones(len(roles),bool)if role=='ALL'else roles==role;ro=rare_o&mask;vo=(vcf_d!=MISSING)&mask;joint=ro&vo;vminor=np.where(vcf_d==MISSING,0,vcf_d if counted==1 else 2-vcf_d)
  row=[chrom,int(pos),chr(BASES[ref]),chr(BASES[alt]),'ALT'if counted==1 else'REF',role,int(ro.sum()),int(vo.sum()),int(joint.sum()),int(np.count_nonzero(joint&(rare_d==vminor))),int(np.count_nonzero(joint&(rare_d!=vminor))),int(np.count_nonzero(vo&(vcf_d==1))),int(np.count_nonzero(vo&(vcf_d==0))),int(np.count_nonzero(vo&(vcf_d==2))),int(np.count_nonzero(ro&(rare_d>0))),int(np.count_nonzero(vo&(vminor>0))),int(np.count_nonzero(joint&((rare_d>0)!=(vminor>0)))),int(np.count_nonzero(mask&(cats==1))),int(np.count_nonzero(mask&(cats==2))),int(np.count_nonzero(mask&(seps==1))),int(np.count_nonzero(mask&(seps==2)))]
  out.append(row)
 return out

def run(spec_path,digest,output,stream):
 started=time.monotonic();c=load_protocol(spec_path,digest);paths={k:receipt(c[k])for k in('events_manifest','events_archive','persons','crosswalk','marker_support')};require(paths['events_archive'].stat().st_size==c['events_archive']['bytes'],'Event archive byte count differs');before=snapshot([Path(spec_path),*paths.values(),Path(__file__)])
 manifest=read(paths['events_manifest']);require(manifest['schema']=='r02_genomic_event_chromosome_v1'and manifest['chromosome']==c['chromosome'],'Event manifest chromosome differs')
 source_ids,roles,role_counts=roster(paths['persons'],paths['crosswalk'],manifest);marker=read(paths['marker_support']);expected=marker['mapped_rare_exact_ref_alt'];require(marker['chromosome']==c['chromosome']and expected>=0,'Marker-support summary differs')
 output=Path(output);require(not output.exists()and output.parent.is_dir()and shutil.disk_usage(output.parent).free>=c['limits']['min_free_bytes'],'Fresh output/free-space required');output.mkdir(mode=0o700)
 vcf=VCF(stream,c,source_ids);groups=iter(vcf.groups());current=next(groups,None);overlap=0;totals={k:Counter()for k in[*role_counts,'ALL']};seen=[];prefix='published/chr'+c['chromosome']+'/';ordered=[prefix+x['path']for x in manifest['channels']['rare']];lookup={prefix+x['path']:x for x in manifest['channels']['rare']};require(len(lookup)==len(ordered)and ordered,'Rare chunk manifest duplicates/empty');last_rare_position=0
 with gzip.open(output/'locus_role_counts.tsv.gz','wt',newline='')as target,tarfile.open(paths['events_archive'],'r|gz')as archive:
  writer=csv.writer(target,delimiter='\t',lineterminator='\n');writer.writerow(FIELDS)
  for member in archive:
   if member.name not in lookup:continue
   require(member.isfile()and member.name not in seen and member.name==ordered[len(seen)],'Rare chunks are repeated, nonregular, or out of authenticated manifest order');record=lookup[member.name];raw=archive.extractfile(member).read(c['limits']['max_chunk_bytes']+1);p,refs,alts,counted,dose,observed=load_chunk(raw,record,len(source_ids),c['limits']['max_chunk_bytes'],c['limits']['max_array_bytes']);require(int(p[0])>last_rare_position,'Rare positions decrease/overlap between chunks');last_rare_position=int(p[-1]);seen.append(member.name)
   for j,pos in enumerate(p):
    while current and current[0]<pos:current=next(groups,None)
    if not current or current[0]>pos:continue
    exact=[]
    for line in current[1]:
     f=line.split(b'\t',5)
     if f[3]==bytes((BASES[refs[j]],))and f[4]==bytes((BASES[alts[j]],)):exact.append(line)
    require(len(exact)<=1,'REPEATED_EXACT_KEY')
    if not exact:continue
    vd,cats,seps=decode_gt(exact[0],vcf.selected);rows=locus_rows(c['chromosome'],pos,refs[j],alts[j],counted[j],dose[:,j],observed[:,j],vd,cats,seps,roles);overlap+=1
    for row in rows:
     writer.writerow(row);r=totals[row[5]]
     for name,value in zip(FIELDS[6:],row[6:]):r[name]+=value
    require(time.monotonic()-started<=c['limits']['timeout_seconds'],'Concordance timeout exceeded')
    require((output/'locus_role_counts.tsv.gz').stat().st_size<=c['limits']['max_output_bytes'],'Output byte bound exceeded')
  require(seen==ordered,'Rare archive chunks missing or out of order')
 for _ in groups:pass
 require(overlap==expected,'Exact-overlap count differs from authenticated marker-support summary')
 unchanged(before);table=output/'locus_role_counts.tsv.gz';result=dict(schema=OUT,status='COMPLETE_DIPLOID_DOSAGE_CONCORDANCE_QC_ONLY',chromosome=c['chromosome'],n_analytical_people=len(source_ids),role_counts=role_counts,rare_catalogue_loci=marker['mapped_rare_markers'],exact_ref_alt_overlap=overlap,position_absent=marker['mapped_rare_position_absent'],position_ref_alt_different=marker['mapped_rare_same_position_ref_alt_difference'],vcf_rows=vcf.rows,vcf_bytes=vcf.nbytes,vcf_sha256=vcf.sha.hexdigest(),vcf_md5_base64=base64.b64encode(vcf.md5.digest()).decode(),counts_by_role={k:dict(v)for k,v in totals.items()},locus_role_table=dict(path=table.name,sha256=support.sha(table),bytes=table.stat().st_size,rows=overlap*(len(role_counts)+1)),limits=c['limits'],seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,phase_separator_is_descriptive_only=True,phase_accuracy_validated=False,homolog_msp_link_validated=False,ancestry_attribution_enabled=False,individual_ids_or_genotypes_emitted=False)
 (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');require(sum(p.stat().st_size for p in output.iterdir())<=c['limits']['max_output_bytes'],'Combined output bound exceeded');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(run(a.spec,a.spec_sha256,a.output_dir,sys.stdin.buffer)))
if __name__=='__main__':main()
