"""Apply one reviewed absolute lease to an exact already-stopped original VM.

Caller supplies the existing identity/cloud helpers and publishes immutable
sources before calling. No creation, deletion, disk detach or automatic retry.
"""
import datetime,json
from pathlib import Path
import r02_publish_evidence as pub


def apply(directory, *, cloud, vm_id, disk_id, name, phase_seconds, metadata_prefix,
          archive, launch_sha256, review_sha256, startup, now=None, absolute_stop=None,
          metadata_additions=None):
    directory=Path(directory);startup=Path(startup)
    pub.require(type(phase_seconds)is int and 0<phase_seconds<=10800,'Invalid admitted lease duration')
    pub.require(metadata_prefix.startswith('r02-')and metadata_prefix.replace('-','').isalnum(),'Unsafe metadata prefix')
    pub.require(not(directory/'start-intent.json').exists(),'Start intent exists: no duplicate boot or clock reset')
    for pin in (launch_sha256,review_sha256,archive['sha256']):pub.validate_sha(pin)
    pub.verify_remote(archive,pinned=True)
    def checked():
        v,d=cloud.identity()
        pub.require(str(v['id'])==vm_id and str(d['id'])==disk_id and v['name']==d['name']==name,
                    'Original VM/disk identity changed')
        pub.require(v['status']=='TERMINATED'and v['labels'].get('team')==d['labels'].get('team')=='frank',
                    'Original VM must be stopped and both labels verified')
        pub.require(len(v['disks'])==1 and not v['disks'][0]['autoDelete']and v['disks'][0]['source']==d['selfLink'],
                    'Retained original disk attachment differs')
        return v,d
    vm,disk=checked()
    clock=now or datetime.datetime.now(datetime.timezone.utc)
    if absolute_stop is None:
        end=(clock+datetime.timedelta(seconds=phase_seconds)).replace(microsecond=0).isoformat().replace('+00:00','Z')
    else:
        # Explicit continuation: the caller authenticates this preexisting bound.
        # This branch never derives a fresh lease by adding phase_seconds.
        original_end=datetime.datetime.fromisoformat(absolute_stop.replace('Z','+00:00'))
        pub.require(original_end.tzinfo is not None and clock<original_end<=clock+datetime.timedelta(hours=4),
                    'Preexisting absolute stop expired or outside bounded continuation')
        end=original_end.isoformat().replace('+00:00','Z')
    additions={metadata_prefix+'-archive-uri':archive['uri']+'#'+archive['generation'],
               metadata_prefix+'-archive-sha256':archive['sha256'],metadata_prefix+'-launch-sha256':launch_sha256,
               metadata_prefix+'-termination-time':end}
    extra=metadata_additions or {}
    pub.require(isinstance(extra,dict)and all(isinstance(k,str)and k.startswith(metadata_prefix+'-')and k not in additions
                and isinstance(v,str)and v and not any(c in v for c in (',','\n','\r','\0'))for k,v in extra.items()),
                'Unsafe or conflicting continuation metadata')
    additions.update(extra)
    old={x['key']:x['value']for x in vm.get('metadata',{}).get('items',[])}
    pub.write_receipt_new(directory/'start-intent.json',dict(utc=pub.pipeline.now(),vm=vm,disk=disk,additions=additions,
                          absolute_stop=end,approval_sha256=review_sha256,phase_seconds=phase_seconds))
    checked()
    cloud.gc(['instances','add-metadata',name,'--metadata='+','.join(k+'='+v for k,v in additions.items()),
              '--metadata-from-file=startup-script='+str(startup)])
    checked()
    cloud.gc(['instances','set-scheduling',name,'--clear-max-run-duration','--termination-time='+end,'--instance-termination-action=STOP'])
    vm,disk=checked()
    actual={x['key']:x['value']for x in vm['metadata']['items']}
    pub.require(actual=={**old,**additions,'startup-script':startup.read_text()},'Unrelated metadata changed')
    pub.require(vm['scheduling']['instanceTerminationAction']=='STOP'and not vm['scheduling'].get('maxRunDuration')
                and datetime.datetime.fromisoformat(vm['scheduling']['terminationTime'].replace('Z','+00:00'))==datetime.datetime.fromisoformat(end.replace('Z','+00:00')),
                'Native absolute STOP differs')
    pub.write_receipt_new(directory/'amended-before-start.json',dict(vm=vm,disk=disk))
    checked()
    operation=cloud.gc(['instances','start',name,'--async'])
    pub.write_receipt_new(directory/'start-operation.json',operation)
    return dict(status='START_REQUESTED_NOT_YET_OBSERVED',vm_id=vm_id,disk_id=disk_id,absolute_stop=end)
