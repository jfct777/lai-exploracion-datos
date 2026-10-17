#!/usr/bin/env python3
"""One reviewed temporary VM, a fixed stop timestamp, and existing retirement gate.

The caller supplies its pinned package and completion contract. No scientific
algorithm or retirement implementation is duplicated here. This controller never
restarts an existing VM, deletes GCS, or extends its original deadline.
"""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import r02_publish_evidence as pub
import r02_retirement_gate as gate
pipe=pub.pipeline;require=pub.require;read=pub.read_unique_json;save=pub.write_receipt_new


class Campaign:
    def __init__(self, directory, *, runner=subprocess.run):
        self.directory=Path(directory).resolve();self.runner=runner
        self.prepared=read(self.directory/'ops-prepared.json')
        self.launch=read(self.directory/'package/launch.json')
        self.vm=self.launch['vm'];self.limits=self.launch['resources']
        self.prefix=self.launch['output_prefix']

    def cli(self,args,*,check=True):
        return self.runner(['gcloud',*args,'--project='+self.vm['project'],
          '--account='+self.vm['account']],capture_output=True,text=True,check=check,timeout=240)

    def inspect(self,kind,*,optional=False):
        p=self.cli(['compute',kind,'describe',self.vm['name'],'--zone='+self.vm['zone'],'--format=json'],check=False)
        if p.returncode:
            require(optional and 'was not found' in p.stderr and
              not any(x in p.stderr.lower() for x in ['403','permission','forbidden','unauthorized']),
              'Cannot establish resource absence: '+p.stderr[-400:])
            return None
        return json.loads(p.stdout)

    def preserve(self,path,name):
        path=Path(path);record=dict(name=path.name,path=str(path),uri=self.prefix+'/provenance/vm/'+name,
                                  **pipe.publication_digests(path))
        out=self.directory/'publication-receipts';out.mkdir(exist_ok=True)
        # Identical describe responses may be retained at two provenance URIs.
        # Bind receipts to both destination and content, preserving old receipts.
        uri_key=hashlib.sha256(record['uri'].encode()).hexdigest()
        receipt=out/(uri_key+'-'+record['sha256']+'.json')
        legacy=out/(record['sha256']+'.json')
        if not receipt.exists() and legacy.exists() and read(legacy).get('uri')==record['uri']:
            receipt=legacy
        if receipt.exists():
            old=read(receipt);require(all(old[k]==record[k] for k in ('uri','bytes','sha256')),'Provenance changed')
            pub.verify_remote(old,pinned=True);return old
        pub.upload_one(record);pub.verify_remote(record,pinned=True);save(receipt,record);return record

    def verify(self,vm,disk,*,original=None,stop=None):
        require(vm['name']==disk['name']==self.vm['name'] and
          vm.get('labels',{}).get('team')==disk.get('labels',{}).get('team')=='frank','Name/labels differ')
        require(vm['machineType'].endswith('/'+self.limits['machine_type']) and
          disk['type'].endswith('/'+self.limits['disk_type']) and int(disk['sizeGb'])==self.limits['disk_gib'],'Size/type differs')
        require(len(vm['disks'])==1 and vm['disks'][0]['source']==disk['selfLink'] and
          vm['disks'][0].get('autoDelete') is False,'Retained original boot disk required')
        require(vm['scheduling']['instanceTerminationAction']=='STOP' and not vm['scheduling'].get('automaticRestart',True),
          'STOP/no-restart contract differs')
        if stop is not None:
            actual=vm['scheduling'].get('terminationTime')
            require(actual is not None and epoch(actual)==epoch(stop),'Absolute native stop differs')
        if original is not None:
            require(str(vm['id'])==str(original['vm_id']) and str(disk['id'])==str(original['disk_id']),
              'Original resource ID changed')

    def commands(self,archive,stop):
        tag=self.launch['resource_tag'];name=self.vm['name'];zone='--zone='+self.vm['zone']
        disk=['compute','disks','create',name,zone,'--size='+str(self.limits['disk_gib'])+'GB',
          '--type='+self.limits['disk_type'],'--image='+self.launch['boot_image'],
          '--image-project=debian-cloud','--labels=team=frank,r02-task='+tag,'--format=json','--quiet']
        metadata={'r02-run-dir':self.launch['runtime_root'],'r02-archive-uri':archive['uri']+'#'+archive['generation'],
          'r02-archive-sha256':archive['sha256'],'r02-launch-sha256':self.prepared['launch_sha256'],
          'r02-native-stop':stop}
        vm=['compute','instances','create',name,zone,'--machine-type='+self.limits['machine_type'],
          '--disk=name='+name+',boot=yes,auto-delete=no','--service-account='+self.vm['account'],'--scopes=cloud-platform',
          '--no-restart-on-failure','--maintenance-policy=MIGRATE','--provisioning-model=STANDARD',
          '--termination-time='+stop,'--instance-termination-action=STOP','--labels=team=frank,r02-task='+tag,
          '--metadata='+','.join(k+'='+v for k,v in metadata.items()),
          '--metadata-from-file=startup-script='+str(self.directory/'startup.sh'),'--format=json','--quiet']
        return disk,vm

    def validate_prepared(self):
        for name,pin in self.prepared['operational_files_sha256'].items():
            require(pipe.sha(self.directory/pub.safe_relative(name))==pin,'Frozen operational source changed: '+name)
        require(pipe.sha(self.directory/'package/launch.json')==self.prepared['launch_sha256'],'Launch changed')
        for name,pin in self.launch['files_sha256'].items():
            require(pipe.sha(self.directory/'package'/pub.safe_relative(name))==pin,'Package changed: '+name)
        subprocess.run(['bash','-n',str(self.directory/'startup.sh')],check=True)

    def create(self,review_sha):
        with (self.directory/'create.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.validate_prepared()
            review=self.directory/'ROOT_OPERATIONAL_REVIEW.json'
            require(pipe.sha(review)==review_sha,'Pinned root admission required');a=read(review)
            require(a['status']=='ADMITTED_EXACT_TEMPORARY_CAMPAIGN_AND_VERIFIED_RETIREMENT' and
              a['ops_prepared_sha256']==pipe.sha(self.directory/'ops-prepared.json'),'Admission/package differs')
            require(not (self.directory/'create-intent.json').exists(),'Creation already attempted; inspect before recovery')
            before={k:self.inspect(k,optional=True) for k in ('instances','disks')}
            require(not any(before.values()),'Original names must be absent before create')
            save(self.directory/('before-create-'+str(time.time_ns())+'.json'),before)
            archive=self.preserve(self.directory/'bundle.tar.gz','bundle.tar.gz')
            for n in ['ops-prepared.json','ROOT_OPERATIONAL_REVIEW.json','OPERATOR_PRE.md',*self.prepared['operational_files_sha256']]:
                if n!='bundle.tar.gz':self.preserve(self.directory/n,n)
            stop=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=self.limits['native_stop_seconds'])).isoformat(timespec='seconds').replace('+00:00','Z')
            disk_cmd,vm_cmd=self.commands(archive,stop)
            save(self.directory/'create-intent.json',dict(native_stop=stop,disk_command=disk_cmd,vm_command=vm_cmd,review_sha256=review_sha))
            disk=json.loads(self.cli(disk_cmd).stdout)[0];save(self.directory/'created-disk.json',disk)
            require(disk.get('labels',{}).get('team')=='frank','Created disk missing label')
            vm=json.loads(self.cli(vm_cmd).stdout)[0];save(self.directory/'created-vm.json',vm)
            identity=dict(vm_id=str(vm['id']),disk_id=str(disk['id']))
            self.verify(vm,disk,original=identity,stop=stop)
            actual_vm=self.inspect('instances');actual_disk=self.inspect('disks')
            self.verify(actual_vm,actual_disk,original=identity,stop=stop)
            save(self.directory/'verified-vm.json',actual_vm);save(self.directory/'verified-disk.json',actual_disk)
            for n in ['create-intent.json','created-disk.json','created-vm.json','verified-vm.json','verified-disk.json']:
                self.preserve(self.directory/n,n)
            return dict(status='VM_CREATED_ORIGINAL_IDS_LABELS_AND_ABSOLUTE_STOP_VERIFIED',**identity,native_stop=stop)

    def watch(self):
        with (self.directory/'watch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            original=dict(vm_id=str(read(self.directory/'created-vm.json')['id']),disk_id=str(read(self.directory/'created-disk.json')['id']))
            stop=read(self.directory/'create-intent.json')['native_stop'];end=epoch(stop)+1800
            while time.time()<end:
                result=self.cli(['storage','objects','describe',self.prefix+'/complete.json','--format=json'],check=False)
                complete=None
                if result.returncode==0:
                    meta=json.loads(result.stdout);target=self.directory/'completed-from-gcs.json'
                    raw=self.cli(['storage','cat',self.prefix+'/complete.json#'+str(meta['generation'])]).stdout
                    if target.exists():require(target.read_text()==raw,'Completion changed')
                    else:
                        with target.open('x') as out:out.write(raw)
                    record=dict(uri=self.prefix+'/complete.json',generation=str(meta['generation']),**pipe.publication_digests(target))
                    pub.verify_remote(record,pinned=True);complete=read(target)
                    require(complete['status']==self.launch['completion_status'],'Unexpected campaign complete status')
                    resources=complete['resource_plan']['resources']
                    require(len(resources)==2 and resources[0]['id']==original['vm_id'] and resources[1]['id']==original['disk_id'],
                      'Different retirement resource IDs')
                else:
                    require('not found: 404' in result.stderr and not any(x in result.stderr.lower() for x in ['403','permission','forbidden']),
                            'Cloud error is not absent completion: '+result.stderr[-300:])
                vm=self.inspect('instances',optional=complete is not None)
                if vm is not None:require(str(vm['id'])==original['vm_id'],'Same-name replacement VM; no action')
                if complete is not None and (vm is None or vm['status']=='TERMINATED'):
                    record=complete['retirement_manifest'];pub.verify_remote(record,pinned=True)
                    p=self.directory/'retirement-manifest.json'
                    if not p.exists():self.cli(['storage','cp',record['uri']+'#'+record['generation'],str(p)])
                    require(pipe.sha(p)==record['sha256'],'Retirement manifest differs')
                    state=self.directory/'retirement-state';state.mkdir(exist_ok=True)
                    value=gate.retire(p,record['sha256'],record,state)
                    save(self.directory/'retirement-complete.json',value)
                    self.preserve(self.directory/'retirement-complete.json','retirement-complete.json');return value
                if vm and vm['status']=='TERMINATED' and complete is None:
                    value=dict(status='STOPPED_DISK_RETAINED_NO_COMPLETE_MARKER',**original,utc=pipe.now())
                    save(self.directory/'stopped-retained.json',value);return value
                time.sleep(30)
            value=dict(status='WATCH_DEADLINE_NO_RESOURCE_DELETED',**original,utc=pipe.now())
            save(self.directory/'watch-deadline.json',value);return value


def epoch(value):
    return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
