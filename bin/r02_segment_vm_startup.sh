#!/usr/bin/env bash
# One authenticated M14.2 chromosome on a disposable VM. No FUSE or new science.
# Sourceable functions permit offline contract tests without metadata or shutdown.
set -euo pipefail

metadata() {
  curl --connect-timeout 5 --max-time 20 --retry 3 -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/$1"
}

validate_assignment() {
  [[ "$worker" =~ ^chr(0[1-9]|1[0-8])$ ]] || return 1
  [[ "$instance_name" == "dnabr-m142-1003-c${worker#chr}" ]] || return 1
  [[ "$instance_id" =~ ^[0-9]+$ ]] || return 1
  [[ "$project_id" == uspbr-242713 && "$zone" == us-central1-a ]] || return 1
  [[ "$run_dir" == "/home/jose.tantalean/projects/lai-exploracion-datos/.claude/runs/r02-m142-fleet-20261003a/$worker" ]] || return 1
  for checksum in "$bundle_sha" "$runtime_sha" "$campaign_sha"; do
    [[ "$checksum" =~ ^[a-f0-9]{64}$ ]] || return 1
  done
  [[ "$runtime_sha" == 8c1f85de9e192be2fcc19e5a278d29ae28fc586a5b76c40eec54ef5f0b384916 ]] || return 1
  [[ "$runtime_uri" == gs://projects-usp/dnaBr-lai/datalake/transient/DNABR_QC/R02_20260930/r02-autosomes-20261001b/parallel-launch-20261001/runtime.tar.gz ]] || return 1
  [[ "$bundle_uri" == gs://projects-usp/dnaBr-lai/datalake/transient/DNABR_QC/R02_20260930/r02-m142-fleet-20261003a/* ]] || return 1
  [[ "$log_uri" == gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-m142-fleet-20261003a/"$worker"/00_vm_logs ]] || return 1
  for object in "$bundle_uri" "$runtime_uri" "$log_uri"; do
    [[ "$object" != *..* && "$object" != *$'\n'* && "$object" != *$'\r'* && "$object" != *' '* ]] || return 1
  done
}

safe_extract() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import os, pathlib, shutil, sys, tarfile
archive, destination, required_root, ceiling = sys.argv[1:]
destination = pathlib.Path(destination)
if destination.resolve() != destination or not destination.is_dir():
    raise ValueError('Noncanonical extraction directory')
with tarfile.open(archive, 'r:gz') as source:
    members, seen, total = source.getmembers(), set(), 0
    for member in members:
        raw = member.name
        name = pathlib.PurePosixPath(raw)
        if name.is_absolute() or '..' in name.parts or '\\' in raw:
            raise ValueError('Unsafe archive path')
        if name == pathlib.PurePosixPath('.') and member.isdir():
            continue
        if not name.parts or (required_root and name.parts[0] != required_root):
            raise ValueError('Unexpected archive root')
        if name in seen or not (member.isdir() or member.isfile()):
            raise ValueError('Duplicate, link or special archive member')
        seen.add(name)
        total += member.size
        if member.size < 0 or total > int(ceiling):
            raise ValueError('Archive exceeds bounded extracted bytes')
        target = destination.joinpath(*name.parts)
        if target.exists() or target.is_symlink():
            raise ValueError('Refuse to overwrite an extracted path')
    for member in members:
        name = pathlib.PurePosixPath(member.name)
        if name == pathlib.PurePosixPath('.'):
            continue
        target = destination.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            target.mkdir(exist_ok=True)
        else:
            with source.extractfile(member) as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
            os.chmod(target, 0o700 if member.mode & 0o111 else 0o600)
PY
}

verify_own_instance() {
  python3 - "$1" "$instance_name" "$instance_id" "$project_id" "$zone" <<'PY'
import json, sys
path, name, identity, project, zone = sys.argv[1:]
record = json.load(open(path))
assert record['name'] == name and str(record['id']) == identity
assert record['zone'].endswith('/projects/'+project+'/zones/'+zone)
assert all(record.get('labels', {}).get(k) == v for k,v in
           {'role':'m142-worker','team':'frank','round':'r02'}.items())
disks = record.get('disks', [])
assert len(disks) == 1 and disks[0].get('boot') is True
assert disks[0]['source'] == 'https://www.googleapis.com/compute/v1/projects/'+project+'/zones/'+zone+'/disks/'+name
PY
}

publish_private_file() {
  # Reuse the existing create-only publisher and its generation/MD5 checks.
  runuser -u jose.tantalean -- python3 - "$run_dir/frozen/bin" "$1" "$2" <<'PY'
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication
path, uri = pathlib.Path(sys.argv[2]), sys.argv[3]
record = dict(name=path.name, path=str(path), uri=uri, **pipeline.publication_digests(path))
publication.upload_one(record)
publication.verify_remote(record, pinned=True)
print(json.dumps(record))
PY
}

validate_campaign_assignment() {
  python3 - "$run_dir" "$campaign_sha" "$worker" <<'PY'
import hashlib, json, pathlib, sys
root, expected, worker = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = root/'campaign.json'
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
record = json.loads(path.read_text())
assert record['mode'] == 'PRODUCTION' and len(record['jobs']) == 1
assert record['resources']['max_concurrent'] == 1 and not record.get('external_reservations')
assert record['resources']['memory_budget_gib'] <= 12
job = record['jobs'][0]
assert job['geometry']['expected']['chrom'] == str(int(worker[3:]))
assert job['geometry']['expected']['n_samples'] == 2619
assert pathlib.Path(job['run_dir']) == root/'job'
PY
}

verify_complete() {
  runuser -u jose.tantalean -- python3 - "$run_dir" "$campaign_sha" <<'PY'
import json, pathlib, sys
root, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root/'frozen/bin'))
import r02_segment_campaign as campaign
import r02_publish_evidence as publication
config, _ = campaign.validate_config(root/'campaign.json', expected)
status = campaign.read_json(root/'status.json')
assert status['stage'] == 'COMPLETE_PUBLISHED_ALL_JOBS'
assert status['campaign_sha256'] == expected
assert len(config['jobs']) == 1 and config['resources']['max_concurrent'] == 1
assert not config.get('external_reservations')
job = config['jobs'][0]
campaign.authenticate_job(config, job)
manifest = campaign.accept_annotation(job)
receipt_path = pathlib.Path(job['run_dir'])/'publication.json'
receipt = campaign.verify_publication_receipt(receipt_path, manifest, job['destination'])
_, _, records = publication.authenticated_bundle(manifest.parent, campaign.operations.sha(manifest))
local = {item['name']:item for item in records}
assert len(receipt['files']) == len(local)
seen = set()
for item in receipt['files']:
    name = item['name']
    assert name in local and name not in seen
    assert all(item[k] == local[name][k] for k in ('bytes','sha256','md5_base64'))
    assert item['uri'] == job['destination'].rstrip('/')+'/'+name
    seen.add(name)
    publication.verify_remote(item, pinned=True)
print(json.dumps(dict(status='COMPLETE_PUBLISHED_REMOTE_REVERIFIED',campaign_sha256=expected,
    manifest_sha256=campaign.operations.sha(manifest),destination=job['destination'],
    publication=receipt,biological_validation_complete=False)))
PY
}

write_diagnostics() {
  python3 - "$run_dir" "$bootstrap_dir" "$instance_name" "$instance_id" "$campaign_sha" "$1" <<'PY'
import json, pathlib, sys, tarfile
root, dest = map(pathlib.Path, sys.argv[1:3])
name, instance_id, campaign_sha, code = sys.argv[3:]
files = []
if root.is_dir():
    for pattern in ('status.json','campaign.json','prepared.json','job/status.json',
                    'job/geometry.complete.json','job/annotation.complete.json','job/publication.json',
                    'job/attempt*.log','job/attempt*.trace.tsv'):
        files.extend(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
seen, total = set(), 0
with tarfile.open(dest/'diagnostics.tar.gz','w:gz') as archive:
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        if path.stat().st_size > 4*1024**2 or total + path.stat().st_size > 24*1024**2:
            continue
        total += path.stat().st_size
        archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
(dest/'startup-final.json').write_text(json.dumps(dict(instance=name,instance_id=instance_id,
    campaign_sha256=campaign_sha,exit_code=int(code),diagnostic_bytes=total,
    status='STARTUP_EXIT_NOT_SCIENTIFIC_VALIDATION'))+'\n')
PY
}

finish() {
  local result=$? complete=0 published=0
  trap - EXIT
  set +e
  if [[ "$assignment_verified" == 1 && "$result" == 0 ]]; then
    timeout 900s bash -c verify_complete > "$bootstrap_dir/remote-completion.json" 2> "$bootstrap_dir/completion-error.log"
    [[ $? == 0 ]] && complete=1
    [[ "$complete" == 1 ]] || result=1
  fi
  if [[ "$assignment_verified" == 1 && -d "$bootstrap_dir" ]]; then
    write_diagnostics "$result"
    tail -c 2097152 /var/log/r02-segment-startup.log > "$bootstrap_dir/startup-final.log"
    chown 1017:1020 "$bootstrap_dir"/diagnostics.tar.gz "$bootstrap_dir"/startup-final.json "$bootstrap_dir"/startup-final.log 2>/dev/null
    published=1
    for name in diagnostics.tar.gz startup-final.json startup-final.log; do
      if [[ -f "$run_dir/frozen/bin/r02_publish_evidence.py" ]] && id jose.tantalean >/dev/null 2>&1; then
        timeout 300s bash -c 'publish_private_file "$1" "$2"' -- "$bootstrap_dir/$name" "$log_uri/$name" > "$bootstrap_dir/$name.publication.json" || published=0
      else
        # Failed bootstrap may not yet have the authenticated publisher/user.
        # Preserve diagnostics if possible, but never authorize disk deletion.
        timeout 300s gcloud storage cp --if-generation-match=0 "$bootstrap_dir/$name" "$log_uri/$name"
        published=0
      fi
    done
    if [[ "$complete" == 1 && "$published" == 1 ]]; then
      chown 1017:1020 "$bootstrap_dir/remote-completion.json"
      timeout 300s bash -c 'publish_private_file "$1" "$2"' -- "$bootstrap_dir/remote-completion.json" "$log_uri/completed.json" > "$bootstrap_dir/completed.publication.json" || published=0
    fi
  fi
  if [[ "$complete" == 1 && "$published" == 1 ]]; then
    timeout 120s gcloud compute instances describe "$instance_name" --project "$project_id" --zone "$zone" --format=json > "$bootstrap_dir/instance-before-delete.json"
    if [[ $? == 0 ]] && verify_own_instance "$bootstrap_dir/instance-before-delete.json"; then
      echo "All scientific outputs and receipts published; deleting this exact temporary VM and its boot disk."
      timeout 180s gcloud compute instances delete "$instance_name" --project "$project_id" --zone "$zone" --quiet --delete-disks=boot
    fi
  fi
  # Any failed verification or deletion retains the disk for explicit recovery.
  echo "Startup finished: exit=$result completion_verified=$complete diagnostics_published=$published; stopping instance."
  if [[ "$assignment_verified" == 1 ]]; then shutdown -h now; fi
  exit "$result"
}

startup_main() {
  umask 077
  exec 9>/run/r02-segment-startup.lock
  flock -n 9 || return 0
  exec > >(tee -a /var/log/r02-segment-startup.log) 2>&1
  assignment_verified=0
  worker='' instance_name='' instance_id='' project_id='' zone='' run_dir='' log_uri='' campaign_sha=''
  bootstrap_dir=/opt/r02-segment-bootstrap
  trap finish EXIT
  worker=$(metadata instance/attributes/r02-worker)
  instance_name=$(metadata instance/name)
  instance_id=$(metadata instance/id)
  project_id=$(metadata project/project-id)
  zone=$(metadata instance/zone); zone=${zone##*/}
  bundle_uri=$(metadata instance/attributes/r02-bundle-uri)
  bundle_sha=$(metadata instance/attributes/r02-bundle-sha256)
  runtime_uri=$(metadata instance/attributes/r02-runtime-uri)
  runtime_sha=$(metadata instance/attributes/r02-runtime-sha256)
  campaign_sha=$(metadata instance/attributes/r02-campaign-sha256)
  run_dir=$(metadata instance/attributes/r02-run-dir)
  log_uri=$(metadata instance/attributes/r02-log-uri)
  validate_assignment
  assignment_verified=1
  install -d -m 711 "$bootstrap_dir"
  export run_dir bootstrap_dir log_uri campaign_sha
  export -f publish_private_file verify_complete
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q docker.io openjdk-17-jre-headless python3 curl ca-certificates gnupg
  if ! command -v gcloud >/dev/null; then apt-get install -y -q google-cloud-cli; fi
  dpkg-query -W docker.io openjdk-17-jre-headless python3 google-cloud-cli
  systemctl enable --now docker
  [[ ! -e "$run_dir" ]]  # No implicit replay after a reboot or a failed attempt.
  gcloud storage cp "$runtime_uri" "$bootstrap_dir/runtime.tar.gz"
  printf '%s  %s\n' "$runtime_sha" "$bootstrap_dir/runtime.tar.gz" | sha256sum -c -
  safe_extract "$bootstrap_dir/runtime.tar.gz" "$bootstrap_dir" runtime 8589934592
  install -m 755 "$bootstrap_dir/runtime/nextflow" /usr/local/bin/nextflow
  groupadd -g 1020 jose.tantalean
  useradd -m -u 1017 -g 1020 -s /bin/bash jose.tantalean
  usermod -aG docker jose.tantalean
  install -d -m 700 -o 1017 -g 1020 /home/jose.tantalean/.nextflow \
    /home/jose.tantalean/.nextflow/framework /home/jose.tantalean/.nextflow/framework/26.04.6
  runuser -u jose.tantalean -- python3 - <<'PY'
import pathlib, tempfile
for directory in ('/home/jose.tantalean/.nextflow',
                  '/home/jose.tantalean/.nextflow/framework',
                  '/home/jose.tantalean/.nextflow/framework/26.04.6'):
    path = pathlib.Path(directory)
    assert path.stat().st_uid == 1017 and path.stat().st_gid == 1020
    with tempfile.TemporaryFile(dir=path) as probe:
        probe.write(b'cache-write-probe')
        probe.flush()
PY
  install -m 600 -o 1017 -g 1020 "$bootstrap_dir/runtime/nextflow-26.04.6-one.jar" /home/jose.tantalean/.nextflow/framework/26.04.6/
  docker load -i "$bootstrap_dir/runtime/images.tar"
  image_id=sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9
  [[ "$(docker image inspect --format '{{.Id}}' "$image_id")" == "$image_id" ]]
  gcloud storage cp "$bundle_uri" "$bootstrap_dir/worker.tar.gz"
  printf '%s  %s\n' "$bundle_sha" "$bootstrap_dir/worker.tar.gz" | sha256sum -c -
  install -d -m 700 -o 1017 -g 1020 "$run_dir"
  safe_extract "$bootstrap_dir/worker.tar.gz" "$run_dir" '' 12884901888
  chown -R 1017:1020 "$run_dir"
  runuser -u jose.tantalean -- python3 "$run_dir/frozen/bin/preprocess_storage_guard.py" --directory "$run_dir" --minimum-free-gib 12
  [[ "$(stat -f -c %T "$run_dir")" == ext2/ext3 ]]  # Linux reports ext4 as ext2/ext3.
  [[ "$(sha256sum "$run_dir/campaign.json" | cut -d' ' -f1)" == "$campaign_sha" ]]
  validate_campaign_assignment
  export run_dir bootstrap_dir log_uri campaign_sha
  export -f publish_private_file verify_complete
  local remaining=$((165600 - SECONDS))
  [[ "$remaining" -gt 0 ]]
  runuser -u jose.tantalean -- env NXF_VER=26.04.6 NXF_OFFLINE=true NXF_DISABLE_CHECK_LATEST=true \
    PYTHONDONTWRITEBYTECODE=1 python3 "$run_dir/frozen/bin/r02_segment_campaign.py" \
    --campaign "$run_dir/campaign.json" --expected-sha256 "$campaign_sha" --validate-only
  timeout --signal=TERM --kill-after=180s "${remaining}s" runuser -u jose.tantalean -- \
    env NXF_VER=26.04.6 NXF_OFFLINE=true NXF_DISABLE_CHECK_LATEST=true PYTHONDONTWRITEBYTECODE=1 \
    python3 "$run_dir/frozen/bin/r02_segment_campaign.py" --campaign "$run_dir/campaign.json" --expected-sha256 "$campaign_sha"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then startup_main "$@"; fi
