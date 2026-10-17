#!/usr/bin/env python3
"""Bounded indexed extraction from an immutable GCS object with bcftools.

The local HTTP bridge gives HTSlib ordinary byte ranges while enforcing a total
transfer budget, an absolute deadline and a fixed object generation. Credentials
exist only in memory in the parent; neither child environment nor files contain
them. A slice hash is not claimed to be the hash of the complete original VCF.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import quote,urlsplit
import urllib.request


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


class TransferBudget:
    def __init__(self,maximum,seconds):
        require(type(maximum)is int and maximum>0 and type(seconds)is int and seconds>0,'Positive transfer/deadline bounds required')
        self.maximum=maximum;self.deadline=time.monotonic()+seconds;self.used=0;self.lock=threading.Lock();self.records=[]

    def reserve(self,start,end):
        with self.lock:
            require(time.monotonic()<self.deadline,'Original range-transfer deadline exhausted')
            require(self.used+end-start+1<=self.maximum,'Range-transfer byte budget exhausted')
            self.used+=end-start+1

    def record(self,start,end,data):
        with self.lock:self.records.append(dict(start=start,end=end,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))


class GcsObject:
    def __init__(self,descriptor,budget,token_provider):
        require(set(descriptor)=={'uri','generation','bytes','sha256','md5_base64'}
                and re.fullmatch('[0-9]+',descriptor['generation'])
                and re.fullmatch('[a-f0-9]{64}',descriptor['sha256'])
                and type(descriptor['bytes'])is int and descriptor['bytes']>0,'Exact immutable source descriptor required')
        uri=urlsplit(descriptor['uri'])
        require(uri.scheme=='gs' and uri.netloc=='projects-usp' and uri.path.startswith('/dnaBr-lai/datalake/')
                and not uri.query and not uri.fragment,'Source outside project datalake')
        name=quote(uri.path.lstrip('/'),safe='');query='?generation='+descriptor['generation']
        self.metadata_url='https://storage.googleapis.com/storage/v1/b/'+uri.netloc+'/o/'+name+query
        self.media_url='https://storage.googleapis.com/download/storage/v1/b/'+uri.netloc+'/o/'+name+query+'&alt=media'
        self.descriptor=descriptor;self.budget=budget;self.token_provider=token_provider

    def open(self,url,headers=None):
        seconds=self.budget.deadline-time.monotonic();require(seconds>0,'Original transfer deadline exhausted')
        request=urllib.request.Request(url,headers={'Authorization':'Bearer '+self.token_provider(),
              'Accept-Encoding':'identity',**(headers or {})})
        return urllib.request.urlopen(request,timeout=min(30,seconds))

    def verify(self):
        with self.open(self.metadata_url)as response:value=json.load(response)
        d=self.descriptor
        require(value['generation']==d['generation'] and int(value['size'])==d['bytes']
                and value['md5Hash']==d['md5_base64'] and not value.get('contentEncoding'),
                'Immutable source generation/size/MD5 or stored encoding differs')
        return {k:value[k]for k in ('generation','size','md5Hash')}

    def read(self,start,end):
        require(0<=start<=end<self.descriptor['bytes'],'Source byte range outside object')
        self.budget.reserve(start,end)
        with self.open(self.media_url,{'Range':f'bytes={start}-{end}'})as response:
            require(response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{self.descriptor["bytes"]}',
                    'Server did not honor exact original byte range')
            require(response.headers.get('x-goog-generation')==self.descriptor['generation'],'Media object generation changed')
            data=response.read(end-start+2)
        require(len(data)==end-start+1,'Truncated or excessive source range')
        self.budget.record(start,end,data);return data


@contextlib.contextmanager
def range_bridge(size,read_range,chunk_bytes):
    """One anonymous loopback path; no upstream URL/token exposed to HTSlib."""
    require(type(size)is int and size>0 and type(chunk_bytes)is int and chunk_bytes>0,'Positive object/window bytes required')
    failures=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):self.handle_body(False)
        def do_GET(self):self.handle_body(True)
        def handle_body(self,body):
            if self.path!='/source.vcf.gz':self.send_error(404);return
            start,end=0,size-1;header=self.headers.get('Range')
            if header:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
                if not match:self.send_error(416);return
                start=int(match[1]);end=min(int(match[2])if match[2]else end,end)
                if not 0<=start<=end<size:self.send_error(416);return
            self.send_response(206 if header else 200)
            self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not body:return
            try:
                while start<=end:
                    last=min(end,start+chunk_bytes-1);data=read_range(start,last)
                    require(len(data)==last-start+1,'Range bridge received wrong byte length')
                    self.wfile.write(data);self.wfile.flush();start=last+1
            except (BrokenPipeError,ConnectionResetError):pass
            except BaseException as exc:failures.append(type(exc).__name__+': '+str(exc));self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=False
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:yield f'http://127.0.0.1:{server.server_port}/source.vcf.gz',failures
    finally:server.shutdown();server.server_close();worker.join(timeout=5)


def metadata_token():
    req=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                               headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(req,timeout=10)as response:result=json.load(response)
    require(isinstance(result.get('access_token'),str),'Metadata did not return a service identity token')
    return result['access_token']


def extract(request,digest,output):
    request=Path(request).resolve();require(sha(request)==digest,'Slice request changed');r=json.loads(request.read_text())
    require(set(r)=={'schema','source','index','regions','samples','bcftools','resources','implementation_sha256'}
            and r['schema']=='r02_structure_indexed_source_slice_v1'
            and r['implementation_sha256']==sha(__file__),'Exact pinned source-slice specification required')
    limits=r['resources'];require(set(limits)=={'max_transfer_bytes','max_runtime_seconds','http_chunk_bytes','max_output_bytes','min_free_bytes'}
                and all(type(v)is int and v>0 for v in limits.values()),'Explicit positive source-slice bounds required')
    for key in ('index','regions','samples','bcftools'):
        require(set(r[key])=={'path','sha256'} and Path(r[key]['path']).is_absolute()
                and sha(r[key]['path'])==r[key]['sha256'],'Pinned '+key+' differs')
    output=Path(output).resolve();require(not output.exists(),'Preserve existing source slice')
    import shutil
    require(shutil.disk_usage(output.parent).free>=limits['min_free_bytes']+limits['max_output_bytes'],'Source-slice disk reserve insufficient')
    output.mkdir(mode=0o700);budget=TransferBudget(limits['max_transfer_bytes'],limits['max_runtime_seconds'])
    source=GcsObject(r['source'],budget,metadata_token);before=source.verify();result=output/'selected.vcf.gz'
    with range_bridge(r['source']['bytes'],source.read,limits['http_chunk_bytes'])as (url,failures):
        command=[r['bcftools']['path'],'view','--no-update','--no-version','--regions-overlap','pos','-R',r['regions']['path'],
                 '-S',r['samples']['path'],'-Oz','-o',str(result),url+'##idx##'+r['index']['path']]
        with (output/'bcftools.log').open('x')as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll()is None:
                    require(time.monotonic()<budget.deadline,'Original slice deadline exhausted')
                    require(not result.exists()or result.stat().st_size<=limits['max_output_bytes'],'Source slice output budget exceeded')
                    time.sleep(.1)
                require(process.returncode==0 and not failures,'Indexed source extraction failed; preserve bounded transfer records')
            finally:
                if process.poll()is None:process.kill();process.wait(timeout=10)
                (output/'transfer-attempt.json').write_text(json.dumps(dict(source=r['source'],bytes_reserved=budget.used,
                    range_records=budget.records,bridge_failures=failures),indent=2)+'\n')
    require(source.verify()==before and sha(request)==digest,'Original source/request changed during slicing')
    (output/'transfer-attempt.json').write_text(json.dumps(dict(source=r['source'],bytes_reserved=budget.used,
        range_records=budget.records,bridge_failures=failures),indent=2)+'\n')
    require(result.stat().st_size<=limits['max_output_bytes'],'Final source slice output budget exceeded')
    for key in ('index','regions','samples','bcftools'):require(sha(r[key]['path'])==r[key]['sha256'],'Slice input changed during read')
    receipt=dict(schema=r['schema'],stage='INDEXED_SOURCE_SLICE_NOT_FIT',request_sha256=digest,source=r['source'],
        object_metadata=before,source_full_sha256_recomputed=False,authentication='Pinned immutable generation,size,MD5 and original byte-hash receipt; bounded ranges recorded',
        transfer_bytes_reserved=budget.used,range_records=budget.records,implementation_sha256=sha(__file__),
        output_sha256=sha(result),output_bytes=result.stat().st_size,credentials_persisted=False)
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();os.umask(0o077);r=extract(a.request,a.request_sha256,a.output)
    print(json.dumps({k:r[k]for k in ('stage','transfer_bytes_reserved','output_sha256','output_bytes')}))
