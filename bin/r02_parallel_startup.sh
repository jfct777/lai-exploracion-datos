#!/usr/bin/env bash
# Start one immutable chromosome assignment on a temporary Compute Engine VM.
# All scientific commands remain in the authenticated Nextflow snapshot.
set -euo pipefail
umask 077
exec > >(tee -a /var/log/r02-worker-startup.log) 2>&1
log_uri=''
run_dir=''
finish() {
  result=$?
  trap - EXIT
  # Preserve diagnostic logs even on an unsuccessful worker; never delete data.
  if [[ "$log_uri" == gs://projects-usp/dnaBr-lai/datalake/* ]] && command -v gcloud >/dev/null; then
    gcloud storage cp --if-generation-match=0 /var/log/r02-worker-startup.log "$log_uri/startup-final.log" || true
    if [[ -d "$run_dir" ]]; then
      tar --exclude=source --exclude=work --exclude=preprocess --exclude=common --exclude=anchor --exclude=sensitivity \
          --exclude='*.npz' --exclude='*.vcf.gz*' --exclude='*.bcf*' --exclude='*.pgen' \
          --exclude='*.grm.*' -czf /tmp/r02-diagnostics.tar.gz -C "$run_dir" . || true
      gcloud storage cp --if-generation-match=0 /tmp/r02-diagnostics.tar.gz "$log_uri/diagnostics.tar.gz" || true
    fi
  fi
  echo "Worker finished: exit=$result; shutting down. Outputs retained in GCS."
  shutdown -h now
  exit "$result"
}
trap finish EXIT
metadata() { curl --retry 3 -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
worker=$(metadata instance/attributes/r02-worker)
[[ "$worker" =~ ^worker(0[1-9]|1[0-2])$ ]]
bundle_uri=$(metadata instance/attributes/r02-bundle-uri)
bundle_sha=$(metadata instance/attributes/r02-bundle-sha256)
runtime_uri=$(metadata instance/attributes/r02-runtime-uri)
runtime_sha=$(metadata instance/attributes/r02-runtime-sha256)
log_uri=$(metadata instance/attributes/r02-log-uri)
for checksum in "$bundle_sha" "$runtime_sha"; do [[ "$checksum" =~ ^[a-f0-9]{64}$ ]]; done
for object in "$bundle_uri" "$runtime_uri" "$log_uri"; do
  [[ "$object" == gs://projects-usp/dnaBr-lai/datalake/* ]]
done
run_dir="/home/jose.tantalean/projects/lai-exploracion-datos/.claude/runs/r02-autosomes-20261001b/parallel/$worker"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q docker.io openjdk-17-jre-headless python3 curl ca-certificates gnupg
if ! command -v gcloud >/dev/null; then apt-get install -y -q google-cloud-cli; fi
systemctl enable --now docker
mkdir -p /opt/r02-bootstrap
gcloud storage cp "$runtime_uri" /opt/r02-bootstrap/runtime.tar.gz
echo "$runtime_sha  /opt/r02-bootstrap/runtime.tar.gz" | sha256sum -c -
tar -xzf /opt/r02-bootstrap/runtime.tar.gz -C /opt/r02-bootstrap
install -m 644 /opt/r02-bootstrap/runtime/gcsfuse.list /etc/apt/sources.list.d/r02-gcsfuse.list
install -m 644 /opt/r02-bootstrap/runtime/cloud.google.asc /usr/share/keyrings/cloud.google.asc
apt-get update -q
apt-get install -y -q gcsfuse=3.11.2
gcsfuse --version
install -m 755 /opt/r02-bootstrap/runtime/nextflow /usr/local/bin/nextflow
groupadd -g 1020 jose.tantalean
useradd -m -u 1017 -g 1020 -s /bin/bash jose.tantalean
usermod -aG docker jose.tantalean
install -d -o 1017 -g 1020 /home/jose.tantalean/.nextflow /home/jose.tantalean/.nextflow/framework /home/jose.tantalean/.nextflow/framework/26.04.6
install -m 600 -o 1017 -g 1020 /opt/r02-bootstrap/runtime/nextflow-26.04.6-one.jar /home/jose.tantalean/.nextflow/framework/26.04.6/
docker load -i /opt/r02-bootstrap/runtime/images.tar
for digest in sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9 sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7; do
  test "$(docker image inspect --format '{{.Id}}' "$digest")" = "$digest"
done
install -d -m 700 -o 1017 -g 1020 /home/jose.tantalean/gcs-dnabr
# The authenticated code uses the same paths as the development VM. No new ACL,
# credentials, public access, or raw-input relocation is introduced here.
gcsfuse --implicit-dirs --only-dir dnaBr-lai/datalake --uid 1017 --gid 1020 \
  --file-mode 600 --dir-mode 700 -o allow_other projects-usp /home/jose.tantalean/gcs-dnabr
gcloud storage cp "$bundle_uri" /opt/r02-bootstrap/worker.tar.gz
echo "$bundle_sha  /opt/r02-bootstrap/worker.tar.gz" | sha256sum -c -
install -d -m 700 -o 1017 -g 1020 "$run_dir"
tar -xzf /opt/r02-bootstrap/worker.tar.gz --no-same-owner -C "$run_dir"
chown -R 1017:1020 "$run_dir"
runuser -u jose.tantalean -- env NXF_VER=26.04.6 NXF_OFFLINE=true NXF_DISABLE_CHECK_LATEST=true \
  PYTHONDONTWRITEBYTECODE=1 python3 "$run_dir/worker.py" run --run-dir "$run_dir"
