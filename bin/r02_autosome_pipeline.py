#!/usr/bin/env python3
"""Private, resumable R02 execution on the existing VM, one autosome at a time.

The checkpoint records commands, hashes, source generations and status. It does
not turn exploratory communities into validated biological populations.
"""
from __future__ import annotations

import argparse
import base64
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MOUNT = Path('/home/jose.tantalean/gcs-dnabr')
BUCKET = 'gs://projects-usp/dnaBr-lai/datalake/'
PREP_IMAGE = 'sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9'
KEEP = ROOT / '.claude/runs/m14-chr22-original-bi-minor-20260922a/m14.keep2619.txt'
KEEP_HASH = 'e6e5e23ffd032dcdaf725ee861efdf42ad7079e606157b7556dea431967a5b17'
OLD = ROOT / '.claude/runs/m14-chr22-original-bi-minor-20260922a/results'
SWEEP22 = ROOT / '.claude/runs/m14-segment-sensitivity-20260923a/results/sensitivity'
AUTOSOMES = list(range(1, 23))
H_SETTINGS = dict(expected_samples=2619, n_seeds=25, seed=42, min_community_size=3,
                  consensus_resolution=1.0, resolutions=[.5, .8, 1., 1.2, 1.5, 2., 3.],
                  configurations=[dict(length_bp=l, gap_bp=50000, min_shared=20,
                                       min_edge_bp=t, min_max_segment_bp=0)
                                  for l, t in [(250000,250000), (250000,500000), (250000,1000000),
                                               (500000,500000), (500000,750000), (500000,1000000)]])


def all_m14_diagnostic_settings(grid, expected_samples):
    """Describe every effective detector, not a new set of Leiden experiments.

    The 1 bp edge threshold is only a schema-compatible no-additional-filter
    sentinel: this settings file is consumed by aggregation, never by clustering.
    """
    configurations = {}
    for length in grid['lengths']:
        for gap in grid['gaps']:
            for count in grid['counts']:
                if any(type(x) is not int or x <= 0 for x in (length, gap, count)):
                    raise ValueError('M14 geometry must contain positive integer values')
                effective = max(count, math.ceil((length-1)/gap)+1)
                configurations[(length, gap, effective)] = dict(
                    length_bp=length, gap_bp=gap, min_shared=effective,
                    min_edge_bp=1, min_max_segment_bp=0)
    if not configurations or type(expected_samples) is not int or expected_samples <= 0:
        raise ValueError('A nonempty M14 grid and positive cohort size are required')
    return dict(expected_samples=expected_samples, n_seeds=1, seed=42,
                min_community_size=3, consensus_resolution=1., resolutions=[1.],
                configurations=[configurations[k] for k in sorted(configurations)])


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def publication_digests(path):
    """Local SHA256 for provenance and GCS-compatible MD5 for transfer checks."""
    h = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return dict(bytes=size, sha256=h.hexdigest(),
                md5_base64=base64.b64encode(md5.digest()).decode('ascii'))


def write_new(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def write_fixed(path,obj):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=obj: raise ValueError('Existing task settings changed: '+str(path))
    else:
        write_new(path,obj)


def status(run, stage, **fields):
    temp = run / 'status.pending.json'
    temp.write_text(json.dumps(dict(updated_utc=now(), stage=stage, **fields), indent=2)+'\n')
    temp.replace(run / 'status.json')


def uri(path):
    return BUCKET + str(Path(path).relative_to(MOUNT))


def object_metadata(path):
    return json.loads(subprocess.check_output([
        'gcloud', 'storage', 'objects', 'describe', uri(path), '--format=json'], text=True, timeout=120))


def ensure_new_or_empty(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f'Refuse to overwrite existing directory: {path}')


@contextmanager
def execution_lock(run):
    """One live controller per immutable run; never steal a live lock."""
    with (Path(run)/'.runner.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another controller already owns this run') from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(dict(pid=os.getpid(), started_utc=now()))+'\n')
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def files_manifest(paths):
    records = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise ValueError('Expected output is missing: '+str(path))
        selected = sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in selected:
            if item.is_file() and item.suffix != '.sqlite':
                records.append(dict(path=str(item), bytes=item.stat().st_size, sha256=sha(item)))
    if paths and not records:
        raise ValueError('Expected output directory contains no files')
    return records


def verify_files(records):
    for record in records:
        path=Path(record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Checkpoint output changed or is missing: '+str(path))


def publication_files(folder):
    """Explicitly omit computation caches and reconstructible large genotypes."""
    excluded_dirs={'work','.nextflow','.matplotlib','02_filter','01_norm'}
    excluded_names={'common.filtered.bcf','common.filtered.bcf.csi','common.filtered.bcf.tbi',
                    'common.pgen','common.pvar','common.psam'}
    return sorted(p for p in folder.rglob('*') if p.is_file()
                  and p.suffix != '.sqlite' and p.name not in excluded_names
                  and not any(part in excluded_dirs for part in p.relative_to(folder).parts)
                  and not p.name.startswith('.nextflow'))


def validate_raw_record_count(folder, chrom, *, source_index=None):
    helper = runpy.run_path(str(ROOT/'bin/preprocess_count_validation.py'))
    return helper['validate_raw_record_count'](folder, chrom, source_index=source_index)


def preprocessing_resources(configuration, source_bytes):
    """Versioned operational settings; never reinterpret frozen legacy runs.

    Three input sizes plus 32 GiB is a conservative planning allowance for
    overlapping BCF/VCF intermediates, not a guaranteed compression bound.
    The low-disk monitor remains necessary while each task runs.
    """
    enabled = configuration.get('preprocess_checkpointed_m01', False)
    if type(enabled) is not bool:
        raise ValueError('preprocess_checkpointed_m01 must be boolean')
    if not enabled:
        return {}
    cpus = configuration.get('container_cpus')
    # M02/M02.1 retain legacy truthy thread defaults: passing zero there would
    # select two extra threads. Do not advertise a one-CPU whole-pipeline mode
    # until those legacy processes support zero explicitly. M01 alone does.
    if type(cpus) is not int or cpus < 2:
        raise ValueError('At least two total CPUs required for the complete preprocessing pipeline')
    if type(source_bytes) is not int or source_bytes < 1:
        raise ValueError('Authenticated input size required for scratch budget')
    multiplier = configuration.get('preprocess_scratch_input_multiplier', 3)
    reserve = configuration.get('preprocess_scratch_reserve_gib', 32)
    if type(multiplier) is not int or multiplier < 1 or type(reserve) is not int or reserve < 12:
        raise ValueError('Invalid preprocessing scratch budget')
    return dict(preprocess_checkpointed_m01=True,
                preprocess_minimum_free_gib=configuration.get('free_disk_min_gib', 12),
                preprocess_required_free_bytes=source_bytes * multiplier + reserve * 1024**3,
                annotation_cpus=cpus)


def prepare(run, analysis_image):
    """No cohort computation; freeze source, exact settings, metadata and paths."""
    if not (ROOT/'.git').exists():
        raise ValueError('Prepare must run from the project checkout, not a frozen source copy')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{2,100}',run.name):
        raise ValueError('Unsafe run name')
    if sha(KEEP) != KEEP_HASH:
        raise ValueError('The fixed 2619-person list changed')
    ensure_new_or_empty(run)
    os.chmod(run, 0o700)
    (run / 'source').mkdir()
    for directory, suffixes in [('bin', ('.py', '.sh')), ('modules', ('.nf',)), ('workflows', ('.nf',))]:
        for src in (ROOT/directory).rglob('*'):
            if src.is_file() and src.suffix in suffixes and '__pycache__' not in src.parts and not src.is_symlink():
                dst = run/'source'/src.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    source_files = {str(p.relative_to(run/'source')): sha(p) for p in (run/'source').rglob('*') if p.is_file()}
    write_new(run/'source.sha256.json', source_files)
    shutil.copy2(KEEP, run/'samples.txt')
    write_new(run/'h_settings.json', H_SETTINGS)
    write_new(run/'weighted_settings_primary.json', {})
    write_new(run/'weighted_settings_sensitivity.json', {'include_rare_graph':False})
    raw = MOUNT/'raw/DNABR_QC/decrypted'
    ref = MOUNT/'trusted/DNABR_QC/reference/hg38_ucsc/Homo_sapiens_assembly38.fasta'
    objects = {}
    for c in AUTOSOMES:
        for suffix in ('.vcf.gz', '.vcf.gz.tbi'):
            p = raw/f'dnabr.hg38.2723.chr{c}{suffix}'
            objects[str(p)] = object_metadata(p)
    for p in (ref, Path(str(ref)+'.fai')):
        objects[str(p)] = object_metadata(p)
    metadata=MOUNT/'trusted/DNABR_QC/metadata/metadata_cleaned.txt'
    if sha(metadata)!='b7be61a6b84a47923279cd319c22c84247ea759b9b558659a048deda9a61f3e7':
        raise ValueError('Existing metadata version changed; inspect before use')
    objects[str(metadata)]=object_metadata(metadata)
    pcrelate=MOUNT/'refined/DNABR_QC/qc_pcrelate/dnabr.pcrelate.kin.tsv'
    objects[str(pcrelate)]=object_metadata(pcrelate)
    historical=MOUNT/'refined/DNABR_QC/presentacion/biologico/M14_chr22__A01_20260922'
    reuse_base=historical/'01_base_bialelicos_originales_alelo_menor/paquete_original'
    reuse=dict(filtered=str(reuse_base/'02_filter/dnabr.hg38.2723.chr22.snv.bi.pass.vcf.gz'),
               rare=str(reuse_base/'lai_rare/dnabr.hg38.2723.chr22.rare.minor.vcf.gz'),
               sensitivity=str(historical/'03_barrido_de_01/paquete_original/results/sensitivity'))
    reuse_paths=[Path(reuse['filtered']),Path(reuse['filtered']+'.tbi'),Path(reuse['rare']),
                 Path(reuse['rare']+'.tbi'),Path(reuse['rare']).with_name('dnabr.hg38.2723.chr22.rare.minor.contract.json')]
    reuse_paths += [Path(reuse['sensitivity'])/name for name in
                    ('pair_configuration_summary.tsv.gz','configuration_summary.tsv','summary.json')]
    for path in reuse_paths:
        objects[str(path)]=object_metadata(path)
    reuse['authenticated_objects']=[str(p) for p in reuse_paths]
    write_new(run/'input_objects.json', objects)
    destination = MOUNT/'refined/DNABR_QC/presentacion/biologico/R02_20260930'/run.name
    # New runs use provisioned disk, never the bucket mount, for heavy scratch.
    # Frozen runs retain their original paths and require a recorded transition.
    bulk = run/'local_bulk'
    ensure_new_or_empty(bulk)
    largest_input = max(int(objects[str(raw/f'dnabr.hg38.2723.chr{c}.vcf.gz')]['size'])
                        for c in AUTOSOMES if c != 22)
    subprocess.run([sys.executable, str(run/'source/bin/preprocess_storage_guard.py'),
                    '--directory', str(bulk), '--required-free-bytes',
                    str(3 * largest_input + 32 * 1024**3)], check=True)
    # Do not create the publication prefix before scratch capacity is checked.
    ensure_new_or_empty(destination)
    config = dict(run_id=run.name, created_utc=now(), prep_image=PREP_IMAGE,
                  analysis_protocol_version=2, m14_skip_legacy_windows=True,
                  evaluate_all_m14_configurations=True,
                  m14_diagnostic_edge_thresholds_bp=[0,250000,500000,750000,1000000],
                  m14_diagnostic_kinship_thresholds=[.0221,.0442],
                  analysis_image=analysis_image, destination=str(destination), bulk=str(bulk),
                  raw_dir=str(raw), ref=str(ref), samples_sha256=KEEP_HASH,
                  reuse_chr22=reuse,
                  metadata=str(metadata), metadata_color_column='finestructure_clusters',
                  pcrelate=str(pcrelate),
                  pcrelate_sha256='c42e0cb215d62e73773dbdaabc34f52b07584e3e95d6a9d762bb571912ccd6b6',
                  chromosomes=AUTOSOMES, processing_order=[22]+list(range(21,0,-1)),
                  free_disk_min_gib=12, container_memory_gib=24, container_cpus=6,
                  preprocess_checkpointed_m01=True,
                  preprocess_scratch_input_multiplier=3, preprocess_scratch_reserve_gib=32,
                  source_frequency_samples=2723, analytical_samples=2619,
                  rare_definition='MAC>=2 and MAF<=0.01 in all2723; original allele count exactly2',
                  rare_mask='complete diploid GT only; unknown is not zero',
                  common_definition='original biallelic PASS SNVs; MAF>=.05 and missing<=.02 in selected2619',
                  common_ld={'primary':[500,1,.2], 'sensitivity':[200,1,.5]},
                  m14={'gaps':[25000,50000,100000], 'lengths':[100000,250000,500000,1000000,2000000],
                       'counts':[5,10,20,50,100,200], 'max_pair_events':750000000,
                       'max_output_rows':100000000, 'max_memory_mb':20480},
                  weighted_graphs=['R', 'C_primary', 'R_plus_C_primary', 'C_sensitivity', 'R_plus_C_sensitivity'],
                  weighted_resolutions=[.5,1.,2.], weighted_seeds=25,
                  omitted_conditional=['RC discordance filtering: no calibrated threshold',
                      'local window/stratum sensitivity: not needed to compute whole-catalogue J',
                      'neural training, LAI simulation, new phasing and new kinship: outside this execution'],
                  interpretation='Full-cohort descriptive evidence for all M14 detectors and graphs; not held-out validation or certified populations',
                  council='Auxiliary adversarial review; not qualified formal approval',
                  notebooklm='Current access timed out; dated primary plan/settings take precedence',
                  origin_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    write_new(run/'run.json', config)
    write_new(run/'m14_diagnostic_settings.json',
              all_m14_diagnostic_settings(config['m14'], config['analytical_samples']))
    write_new(run/'frozen.sha256.json', {p.name:sha(p) for p in
              [run/'run.json',run/'source.sha256.json',run/'h_settings.json',run/'weighted_settings_primary.json',
               run/'weighted_settings_sensitivity.json',run/'input_objects.json',run/'samples.txt',
               run/'m14_diagnostic_settings.json']})
    (run/'logs').mkdir()
    (run/'checkpoints').mkdir()
    status(run, 'PREPARED_NOT_STARTED', destination=uri(destination))
    print(run)


class Runner:
    def __init__(self, run):
        self.run = run.resolve()
        self.c = json.loads((run/'run.json').read_text())
        self.source = run/'source'
        self.bin = self.source/'bin'
        self.samples = run/'samples.txt'
        self.dest = Path(self.c['destination'])
        self.objects=json.loads((run/'input_objects.json').read_text())
        self.current_stage = 'INITIALIZING'
        for p,h in json.loads((run/'frozen.sha256.json').read_text()).items():
            if sha(run/p) != h: raise ValueError('Frozen run input changed: '+p)
        for p,h in json.loads((run/'source.sha256.json').read_text()).items():
            if sha(self.source/p) != h: raise ValueError('Frozen source changed: '+p)
        self.env = dict(os.environ, NXF_SYNTAX_PARSER='v1', NXF_OFFLINE='true',
                        NXF_DISABLE_CHECK_LATEST='true', OPENBLAS_NUM_THREADS='1',
                        OMP_NUM_THREADS='6', MKL_NUM_THREADS='1', MPLBACKEND='Agg')

    def verify_inputs(self, paths):
        """A remembered filename is not a frozen object: check its generation."""
        for path in paths:
            name=str(path)
            expected=self.objects.get(name)
            if expected is None:
                raise ValueError('Input was not authenticated at preparation: '+name)
            current=object_metadata(path)
            for field in ('generation','size'):
                if field not in expected or str(current.get(field)) != str(expected[field]):
                    raise ValueError('GCS input changed since preparation: '+name+' '+field)
            for field in ('crc32c_hash','md5_hash','crc32c','md5Hash'):
                if field in expected and current.get(field)!=expected[field]:
                    raise ValueError('GCS input digest changed: '+name)

    def stop_own_containers(self):
        ids = subprocess.check_output(['docker','ps','-q','--filter','label=dnabr-r02='+self.c['run_id']], text=True).split()
        if ids:
            subprocess.run(['docker','stop','--time','30',*ids], timeout=120, check=False)

    def command(self, stage, command, cwd=None, expected_outputs=()):
        self.current_stage = stage
        checkpoint = self.run/'checkpoints'/f'{stage}.json'
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            if old['command'] != [str(x) for x in command]: raise ValueError('Checkpoint command changed')
            if expected_outputs and not old.get('outputs'):
                raise ValueError('Checkpoint does not authenticate expected outputs')
            verify_files(old.get('outputs',[]))
            return
        free = shutil.disk_usage(self.run).free/1024**3
        if free < self.c['free_disk_min_gib']: raise RuntimeError('Insufficient local disk')
        command = [str(x) for x in command]
        log = self.run/'logs'/f'{stage}.{int(time.time())}.log'
        status(self.run, stage, state='RUNNING', command=command, log=str(log), free_disk_gib=free)
        with log.open('x') as f:
            p = subprocess.Popen(command, cwd=cwd or self.run, env=self.env,
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    free = shutil.disk_usage(self.run).free/1024**3
                    status(self.run, stage, state='RUNNING', pid=p.pid, log=str(log), free_disk_gib=round(free,2))
                    if free < self.c['free_disk_min_gib']:
                        self.stop_own_containers()
                        os.killpg(p.pid, signal.SIGTERM)
                        raise RuntimeError('Stopped own tasks before local disk exhaustion')
                    time.sleep(20)
                if p.returncode: raise RuntimeError(f'{stage} failed with exit {p.returncode}; see {log}')
            except BaseException:
                if p.poll() is None:
                    self.stop_own_containers()
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=60)
                raise
        write_new(checkpoint, dict(command=command, completed_utc=now(), log=str(log), returncode=0,
                                  outputs=files_manifest(expected_outputs)))

    def docker(self, stage, args, image=None, cwd=None, expected_outputs=None):
        """Nextflow owns the pinned Docker analysis task, not an ad-hoc shell."""
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',stage): raise ValueError('Unsafe analysis stage name')
        args=[str(x) for x in args]
        if expected_outputs is None:
            if '--output-prefix' in args:
                prefix=args[args.index('--output-prefix')+1]
                expected_outputs=[Path(prefix+'.npz'),Path(prefix+'.manifest.json')]
            elif '--output' in args:
                expected_outputs=[Path(args[args.index('--output')+1])]
            elif '--output-dir' in args:
                expected_outputs=[Path(args[args.index('--output-dir')+1])]
            else:
                raise ValueError('Every stage must declare verifiable outputs: '+stage)
        taskdir=self.run/'stages'/stage
        taskdir.mkdir(parents=True,exist_ok=True)
        command_file=taskdir/'command.json'
        write_fixed(command_file,dict(command=args,tool_sha256=sha(args[1])))
        parameters=taskdir/'parameters.json'
        write_fixed(parameters,dict(r02_stage=stage,r02_command_file=str(command_file),
                    r02_exec_tool=str(self.bin/'r02_exec_task.py'),r02_source_bin=str(self.bin)))
        cpus=self.c['container_cpus']; memory=self.c['container_memory_gib']
        text=f"""process.executor='local'
process.container='{image or self.c['analysis_image']}'
process.cpus={cpus}
process.memory='{memory} GB'
process.time='72h'
process.stageInMode='symlink'
process.errorStrategy='terminate'
process.maxRetries=0
docker.enabled=true
docker.runOptions='--network none --user {os.getuid()}:{os.getgid()} --memory={memory}g --cpus={cpus} --label dnabr-r02={self.c['run_id']} -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS={cpus} -e MPLBACKEND=Agg -e MPLCONFIGDIR={self.run}/.matplotlib -v /home/jose.tantalean:/home/jose.tantalean'
executor.queueSize=1
executor.cpus={cpus}
executor.memory='{memory+1} GB'
"""
        config=taskdir/'runtime.config'
        if config.exists():
            if config.read_text()!=text: raise ValueError('Existing task runtime configuration changed')
        else: config.write_text(text)
        self.command(stage,['/usr/local/bin/nextflow','-log',taskdir/'nextflow.log','-C',config,
                    'run',self.source/'workflows/r02_analysis_task.nf','-params-file',parameters,
                    '-work-dir',taskdir/'work','-ansi-log','false','-with-trace',taskdir/'trace.tsv'],
                    cwd=taskdir,expected_outputs=expected_outputs)

    def publish(self, folder, relative):
        """Verify through the GCS API, not potentially stale FUSE stat entries."""
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe relative publication directory')
        self.current_stage = 'publish_'+relative.replace('/','_')
        status(self.run, self.current_stage, state='RUNNING', operation='VERIFY_OR_PUBLISH')
        target = self.dest/relative
        receipts = self.run/'checkpoints'/('publish_'+relative.replace('/','_')+'.json')
        if receipts.exists():
            for record in json.loads(receipts.read_text())['files']:
                remote=MOUNT/record['uri'].removeprefix(BUCKET)
                metadata=object_metadata(remote)
                if str(metadata.get('generation')) != str(record['generation']):
                    raise ValueError('Published object changed since checkpoint: '+record['uri'])
            return
        files = publication_files(folder)
        records = []
        for index, p in enumerate(files, 1):
            rel = p.relative_to(folder)
            dst = target/rel
            digest = publication_digests(p)
            status(self.run, self.current_stage, state='RUNNING',
                   file_index=index, file_count=len(files), file=str(rel))
            # generation=0 forbids overwriting, including during a race. An
            # already existing object yields HTTP 412; only matching API bytes
            # below allow it to be reused. gcloud forbids combining this with
            # no-clobber. Never consult the FUSE negative-stat cache here.
            uploaded = subprocess.run(['gcloud','storage','cp','--if-generation-match=0',
                                       '--content-md5='+digest['md5_base64'],str(p),uri(dst)],
                                      capture_output=True, text=True, timeout=7200,
                                      env={**os.environ, 'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED':'false'})
            if uploaded.returncode and 'HTTPError 412:' not in uploaded.stderr:
                raise subprocess.CalledProcessError(uploaded.returncode, uploaded.args,
                                                    output=uploaded.stdout, stderr=uploaded.stderr)
            metadata=object_metadata(dst)
            remote_md5=metadata.get('md5_hash', metadata.get('md5Hash'))
            if (int(metadata.get('size', -1)) != digest['bytes']
                    or remote_md5 != digest['md5_base64'] or not metadata.get('generation')):
                raise ValueError('Published GCS bytes differ from source or lack checksum '+str(dst))
            if publication_digests(p) != digest:
                raise ValueError('Local source changed during publication '+str(p))
            records.append(dict(path=str(p), uri=uri(dst), **digest,
                                generation=str(metadata['generation']),
                                verification='GCS API size + MD5; local SHA256 recorded'))
        write_new(receipts, dict(completed_utc=now(),files=records))

    def preprocess(self, c):
        folder = self.run/f'chr{c:02d}'
        folder.mkdir(exist_ok=True)
        if c == 22:
            reuse_spec=self.c['reuse_chr22']
            self.verify_inputs(reuse_spec['authenticated_objects'])
            filtered=Path(reuse_spec['filtered'])
            rare=Path(reuse_spec['rare'])
            contract = json.loads(rare.with_name('dnabr.hg38.2723.chr22.rare.minor.contract.json').read_text())
            if contract['counts']['rare'] != 424573 or contract['counts']['rare_ref'] != 385:
                raise ValueError('Wrong chr22 version')
            reuse = folder/'reuse.json'
            if not reuse.exists(): write_new(reuse, dict(filtered=str(filtered),rare=str(rare),rare_sha256=sha(rare),
                                                        sensitivity=reuse_spec['sensitivity']))
            return folder, filtered, rare
        raw_path=Path(self.c['raw_dir'])/f'dnabr.hg38.2723.chr{c}.vcf.gz'
        self.verify_inputs([raw_path,Path(str(raw_path)+'.tbi'),Path(self.c['ref']),Path(self.c['ref']+'.fai')])
        params = dict(outdir=str(folder/'preprocess'), cpus=6, memory='24 GB',time='72h',
                      resources={name:{'threads':6} for name in
                      ('preprocess_norm_leftalign','preprocess_filter_snv_biallelic_pass','lai_rare_bialelic_only')},
                      r02_chrom=c, r02_raw_vcf=str(Path(self.c['raw_dir'])/f'dnabr.hg38.2723.chr{c}.vcf.gz'),
                      r02_bin_dir=str(self.bin),ref_fasta=self.c['ref'],
                      preprocess_large_temp_dir=self.c['bulk'], bcftools_min_alleles=2,
                      plink_max_alleles=2,plink_snps_only=True,max_maf=None,keep_pass=True,
                      lai_rare_max_maf=.01,lai_rare_min_mac=2,lai_rare_keep_format='GT',lai_rare_remove_info=False)
        paramfile=folder/'parameters.json'
        operational = preprocessing_resources(self.c, int(self.objects[str(raw_path)]['size']))
        if operational:
            annotation_cpus = operational.pop('annotation_cpus')
            params.update(operational)
            params['resources']['preprocess_original_alleles'] = dict(cpus=annotation_cpus)
            params['cpus'] = annotation_cpus
            for name in ('preprocess_norm_leftalign', 'preprocess_filter_snv_biallelic_pass',
                         'lai_rare_bialelic_only'):
                params['resources'][name]['threads'] = max(0, annotation_cpus - 1)
            # Initial capacity includes all intermediates. A resumed invocation
            # already has some of those bytes on disk; each incomplete Nextflow
            # stage performs its own remaining-capacity check.
            required_now = 0 if paramfile.exists() else operational['preprocess_required_free_bytes']
            check = subprocess.run([sys.executable, str(self.bin/'preprocess_storage_guard.py'),
                '--directory', self.c['bulk'], '--required-free-bytes', str(required_now),
                '--minimum-free-gib', str(operational['preprocess_minimum_free_gib'])], check=True,
                capture_output=True, text=True)
            # A new observation is timestamped; do not overwrite an earlier preflight.
            write_new(folder/f'storage_preflight_{time.time_ns()}.json', json.loads(check.stdout))
        write_fixed(paramfile,params)
        task_cpus = params['cpus']
        cfg=folder/'runtime.config'
        if not cfg.exists():
            cfg.write_text(f"""process.executor='local'
process.container='{self.c['prep_image']}'
process.stageInMode='symlink'
process.errorStrategy='terminate'
process.maxRetries=0
docker.enabled=true
docker.runOptions='--network none --user {os.getuid()}:{os.getgid()} --memory=24g --cpus={task_cpus} --label dnabr-r02={self.c['run_id']}'
executor.queueSize=1
executor.cpus={task_cpus}
executor.memory='25 GB'
process {{
 withName:PREPROCESS_NORM_LEFTALIGN {{ publishDir=[enabled:false] }}
 withName:PREPROCESS_FILTER_SNV_BIALLELIC_PASS {{ publishDir=[path:'{folder}/preprocess/02_filter',mode:'symlink',overwrite:false] }}
 withName:LAI_RARE_BIALELIC_ONLY {{ publishDir=[path:'{folder}/preprocess/lai_rare',mode:'link',overwrite:false] }}
}}
""")
        base=f'dnabr.hg38.2723.chr{c}'
        filtered=folder/'preprocess/02_filter'/f'{base}.snv.bi.pass.vcf.gz'
        rare=folder/'preprocess/lai_rare'/f'{base}.rare.minor.vcf.gz'
        # Only the durable small rare outputs belong in the completion hash;
        # filtered full genotypes are explicitly temporary and later removed.
        expected=[rare,Path(str(rare)+'.tbi'),rare.with_name(f'{base}.rare.minor.contract.json'),
                  rare.with_name(f'{base}.rare.minor.counts.tsv')]
        self.command(f'chr{c:02d}_M01_M02_M021', ['/usr/local/bin/nextflow','-log',folder/'nextflow.log',
            '-C',cfg,'run',self.source/'workflows/r02_preprocess_autosome.nf','-params-file',paramfile,
            '-work-dir',folder/'work','-ansi-log','false','-with-trace',folder/'trace.tsv','-resume'],
            cwd=folder,expected_outputs=expected)
        audit=validate_raw_record_count(folder,c,source_index=Path(str(raw_path)+'.tbi'))
        if not (folder/'sequential_input_counts.json').exists():
            write_new(folder/'sequential_input_counts.json',audit)
        if not (folder/'preprocessing_counts.tsv').exists():
            shutil.copy2(folder/'preprocess/02_filter'/f'{base}.counts.tsv',folder/'preprocessing_counts.tsv')
        contract=json.loads(rare.with_name(f'{base}.rare.minor.contract.json').read_text())
        if (contract['cohort_samples_before']!=2723 or contract['cohort_samples_after']!=2723
                or contract['cohort_sha256']!='11e5482eded751cad8ae0df9b8bda3c0b861c04ad6be107383065936e684fa0a'
                or contract['criteria']['original_allele_count']!=2
                or contract['criteria']['min_mac_inclusive']!=2
                or float(contract['criteria']['max_maf_inclusive'])!=.01
                or not contract['criteria']['GT_REF_ALT_preserved']): raise ValueError('Invalid rare contract')
        if operational:
            subprocess.run([sys.executable, str(self.bin/'preprocess_audit.py'),
                            '--folder', str(folder), '--chrom', str(c)], check=True)
        return folder,filtered,rare

    def analyze_chromosome(self,c):
        done=self.run/'checkpoints'/f'chr{c:02d}_complete.json'
        if done.exists():
            verify_files(json.loads(done.read_text())['outputs'])
            return
        science=self.run/'checkpoints'/f'chr{c:02d}_science_complete.json'
        if science.exists():
            verify_files(json.loads(science.read_text())['outputs'])
            self.finalize_chromosome(self.run/f'chr{c:02d}',c)
            return
        folder,filtered,rare=self.preprocess(c)
        prefix=folder/'rare_evidence'
        self.docker(f'chr{c:02d}_rare_J', ['python3',self.bin/'r02_genomic_pair_evidence.py','rare',
            '--vcf',rare,'--samples',self.samples,'--expected-source-samples','2723','--expected-samples','2619',
            '--chrom',str(c),'--chunk-sites','2048','--output-prefix',prefix],image=self.c['prep_image'])
        common=folder/'common'
        self.docker(f'chr{c:02d}_common_GRM', ['bash',self.bin/'r02_common_grm.sh','--vcf',filtered,
            '--samples',self.samples,'--chrom',str(c),'--expected-source-samples','2723','--expected-samples','2619',
            '--threads','6','--memory-mb','8192','--min-maf','.05','--max-missing','.02',
            '--ld-configs','primary:500:0.2,sensitivity:200:0.5','--output-dir',common],image=self.c['prep_image'])
        for panel in ('primary','sensitivity'):
            self.docker(f'chr{c:02d}_common_{panel}_import', ['python3',self.bin/'r02_genomic_pair_evidence.py','common',
                '--grm-prefix',common/panel/'grm','--method-manifest',common/panel/'method.json',
                '--samples',self.samples,'--expected-samples','2619','--chrom',str(c),
                '--output-prefix',folder/f'common_{panel}_evidence'])
        if c != 22:
            anchor=folder/'anchor'
            anchor.mkdir(exist_ok=True)
            window_args = (['--skip-windows', 'true']
                           if self.c.get('m14_skip_legacy_windows', False) else [])
            self.docker(f'chr{c:02d}_M14_anchor', ['python3',self.bin/'rare_allele_sharing_painter.py',
                '--mode','scan','--input',rare,'--input-format','vcf_rare','--chr',str(c),
                '--carrier-allele-mode','source_minor','--expected-samples','2619','--sample-ids-file',self.samples,
                '--window-size-bp','500000','--step-size-bp','250000','--min-shared-variants','10','--min-jaccard','.1',
                '--max-gap-bp','50000','--min-segment-bp','1000000','--n-jobs','1','--skip-plots','true', *window_args,
                '--out-sharing-windows',anchor/'windows.tsv.gz','--out-pairwise-segments',anchor/'segments.tsv.gz',
                '--out-summary-json',anchor/'summary.json','--output-dir',anchor],image=self.c['prep_image'])
            grid=self.c['m14']
            self.docker(f'chr{c:02d}_M14_sensitivity', ['python3',self.bin/'rare_segment_sensitivity.py','--input',rare,
                '--chr',str(c),'--sample-ids-file',self.samples,'--expected-samples','2619',
                '--anchor-segments',anchor/'segments.tsv.gz','--anchor-summary',anchor/'summary.json',
                '--output-dir',folder/'sensitivity','--gaps-bp',','.join(map(str,grid['gaps'])),
                '--lengths-bp',','.join(map(str,grid['lengths'])),'--min-shared',','.join(map(str,grid['counts'])),
                '--max-pair-events',str(grid['max_pair_events']),'--max-memory-mb',str(grid['max_memory_mb']),
                '--max-output-rows',str(grid['max_output_rows'])],image=self.c['prep_image'])
        write_new(science,dict(completed_utc=now(),outputs=files_manifest(publication_files(folder))))
        self.finalize_chromosome(folder,c)

    def finalize_chromosome(self,folder,c):
        self.publish(folder,f'01_estructura/desarrollo/por_cromosoma/chr{c:02d}')
        self.cleanup_bulk(folder,c)
        # These local genotype intermediates have already been consumed by
        # both LD/GRM configurations. Keep all resulting matrices and manifests.
        common_removed=[]
        common=folder/'common'
        for name in ('common.filtered.bcf','common.filtered.bcf.csi','common.filtered.bcf.tbi',
                     'common.pgen','common.pvar','common.psam'):
            p=common/name
            if p.exists():
                if p.is_symlink() or not p.is_file(): raise ValueError('Unsafe common temporary file')
                common_removed.append(dict(path=str(p),bytes=p.stat().st_size))
                p.unlink()
        if not (folder/'local_common_cleanup.json').exists():
            write_new(folder/'local_common_cleanup.json',dict(removed_generated_intermediates=common_removed,
                       utc=now(),recovery='Regenerate from authenticated M02 input; derived GRMs retained'))
        done=self.run/'checkpoints'/f'chr{c:02d}_complete.json'
        write_new(done,dict(completed_utc=now(),chromosome=c,outputs=files_manifest(publication_files(folder))))

    def cleanup_bulk(self,folder,c):
        """Delete only exact new large files after validated publication/consumers."""
        receipt=folder/'bulk_cleanup.json'
        if c==22 or receipt.exists(): return
        relative=f'01_estructura/desarrollo/por_cromosoma/chr{c:02d}'
        if not (self.run/'checkpoints'/('publish_'+relative.replace('/','_')+'.json')).exists():
            raise ValueError('Refuse bulk cleanup before successful publication')
        base=f'dnabr.hg38.2723.chr{c}'
        allowed={base+'.original.bcf',base+'.norm.vcf.gz',base+'.norm.vcf.gz.tbi',base+'.norm.log',
                 base+'.snv.bi.vcf.gz',base+'.snv.bi.vcf.gz.tbi',base+'.snv.bi.pass.vcf.gz',
                 base+'.snv.bi.pass.vcf.gz.tbi',base+'.counts.tsv'}
        plan_path=folder/'bulk_cleanup_plan.json'
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
        else:
            plan=[]
            for marker in (folder/'work').glob('*/*/*.large_temp_dir.txt'):
                d=Path(marker.read_text().strip())
                if d.parent!=Path(self.c['bulk']) or not re.fullmatch(f'm0[12]_chr{c}\\.[A-Za-z0-9]+',d.name) or d.is_symlink():
                    raise ValueError('Unsafe bulk cleanup target')
                entries=[]
                for p in sorted(d.iterdir()):
                    if p.name not in allowed or not p.is_file() or p.is_symlink():
                        raise ValueError('Unexpected generated file')
                    entries.append(dict(path=str(p),
                                        uri=uri(p) if p.is_relative_to(MOUNT) else None,
                                        bytes=p.stat().st_size))
                plan.append(dict(directory=str(d),files=entries))
            if not plan: raise ValueError('No authenticated bulk-directory marker found')
            write_new(plan_path,plan)
        records=[]
        for group in plan:
            d=Path(group['directory'])
            if d.parent!=Path(self.c['bulk']) or not re.fullmatch(f'm0[12]_chr{c}\\.[A-Za-z0-9]+',d.name) or d.is_symlink():
                raise ValueError('Unsafe saved cleanup plan')
            for item in group['files']:
                p=Path(item['path'])
                if p.parent!=d or p.name not in allowed or p.is_symlink():
                    raise ValueError('Unsafe saved cleanup file')
                if p.exists():
                    if not p.is_file() or p.stat().st_size!=item['bytes']:
                        raise ValueError('Generated intermediate changed before cleanup')
                    p.unlink()
                records.append(item)
            if d.exists(): d.rmdir()
        write_new(receipt,dict(removed_new_intermediates_only=records,utc=now(),
                              recovery='Regenerate from immutable raw input; no historical source removed'))

    def aggregate(self):
        manifest={'schema_version':1,'chromosomes':[]}
        for c in AUTOSOMES:
            s=Path(self.c['reuse_chr22']['sensitivity']) if c==22 else self.run/f'chr{c:02d}/sensitivity'
            files=dict(pair_summary=s/'pair_configuration_summary.tsv.gz',
                       configuration_summary=s/'configuration_summary.tsv', summary=s/'summary.json')
            manifest['chromosomes'].append(dict(chromosome=str(c),**{k:str(v) for k,v in files.items()},
                                               sha256={k:sha(v) for k,v in files.items()}))
        mp=self.run/'autosome_inputs.json'
        write_fixed(mp,manifest)
        # The frozen JSON is the authority for BOTH aggregation and clustering.
        # A Python constant must not silently override a reviewed configuration.
        h_settings=json.loads((self.run/'h_settings.json').read_text())
        evaluate_all=self.c.get('evaluate_all_m14_configurations',False)
        aggregation_settings=self.run/('m14_diagnostic_settings.json' if evaluate_all else 'h_settings.json')
        if evaluate_all:
            expected=all_m14_diagnostic_settings(self.c['m14'],self.c['analytical_samples'])
            if json.loads(aggregation_settings.read_text()) != expected:
                raise ValueError('Diagnostic settings omit or alter an effective M14 configuration')
        out=self.run/('aggregate_M14_all' if evaluate_all else 'aggregate_H')
        self.docker('all22_H_aggregate',['python3',self.bin/'m165_autosome_sweep.py','aggregate','--manifest',mp,
            '--settings',aggregation_settings,'--sample-ids',self.samples,'--output',out])
        if evaluate_all:
            self.verify_inputs([self.c['pcrelate']])
            self.docker('all22_M14_configuration_diagnostics',[
                'python3',self.bin/'r02_m14_configuration_diagnostics.py',
                '--pair-summary',out/'pair_configuration_summary.tsv.gz',
                '--configuration-summary',out/'configuration_summary.tsv',
                '--aggregation-receipt',out/'aggregation.json', '--sample-ids',self.samples,
                '--expected-samples',str(self.c['analytical_samples']),
                '--expected-configurations',str(len(expected['configurations'])),
                '--pcrelate-file',self.c['pcrelate'], '--expected-pcrelate-sha256',self.c['pcrelate_sha256'],
                '--thresholds',','.join(map(str,self.c['m14_diagnostic_kinship_thresholds'])),
                '--edge-thresholds-bp',','.join(map(str,self.c['m14_diagnostic_edge_thresholds_bp'])),
                '--output-dir',self.run/'diagnostics_M14_all'])
        b64=base64.b64encode(json.dumps(h_settings).encode()).decode()
        prepared=self.run/'prepared_H'
        self.docker('all22_H_prepare',['python3',self.bin/'m165_chr22_sweep.py','prepare',
            '--pair-summary',out/'pair_configuration_summary.tsv.gz','--configuration-summary',out/'configuration_summary.tsv',
            '--sample-ids',self.samples,'--output',prepared,'--settings-base64',b64,
            '--chromosomes',','.join(map(str,AUTOSOMES)),'--aggregation-receipt',out/'aggregation.json'])
        hresults=self.run/'communities_H'
        hresults.mkdir(exist_ok=True)
        for d in sorted(prepared.glob('L*')):
            self.docker('all22_H_'+d.name,['python3',self.bin/'m165_chr22_sweep.py','run','--configuration-dir',d,
                '--output',hresults/d.name,'--core-script',self.bin/'ibd_community_enhanced.py'])
        # All 6 graphs × 7 resolutions, with the same recorded projection settings.
        ids=[d.name for d in sorted(prepared.glob('L*'))]
        settings=dict(prefix='R02_all22_H',config_ids=ids,resolutions=h_settings['resolutions'],
                      expected_samples=self.c['analytical_samples'])
        self.verify_inputs([self.c['metadata']])
        self.docker('all22_H_plots',['python3',self.bin/'m165_spectral_figures.py','--results-dir',hresults,
            '--output-dir',self.run/'plots_H','--parameters-json-text',json.dumps(settings),
            '--metadata-file',self.c['metadata'],'--metadata-sample-column','ID',
            '--metadata-color-column',self.c['metadata_color_column']])
        self.docker('all22_H_diagnostics',['python3',self.bin/'m165_sweep_figures.py','--results-dir',hresults,
            '--output-dir',self.run/'diagnostics_H','--prefix','R02_all22_H','--all-networks'])
        self.verify_inputs([self.c['pcrelate']])
        for label, threshold in [('0221','.0221'),('0442','.0442')]:
            self.docker('all22_H_kinship_'+label,['python3',self.bin/'m165_graph_kinship.py',
                '--results-dir',hresults,'--pcrelate-file',self.c['pcrelate'],
                '--expected-sha256',self.c['pcrelate_sha256'],'--threshold',threshold,
                '--output-dir',self.run/f'diagnostics_H_kinship_{label}'])
        for panel in ('primary','sensitivity'):
            evidence_dir=self.run/f'aggregate_R_C_{panel}'
            evidence_dir.mkdir(exist_ok=True)
            combined=evidence_dir/'evidence'
            self.docker('all22_R_C_'+panel,['python3',self.bin/'r02_genomic_pair_evidence.py','aggregate',
                '--samples',self.samples,'--expected-samples','2619','--chromosomes',','.join(map(str,AUTOSOMES)),
                '--rare',*[self.run/f'chr{c:02d}/rare_evidence.npz' for c in AUTOSOMES],
                '--common',*[self.run/f'chr{c:02d}/common_{panel}_evidence.npz' for c in AUTOSOMES],
                '--output-prefix',combined])
            # The weighted runner uses real dimensionless weights, never fake base pairs.
            self.docker('all22_R_C_communities_'+panel,['python3',self.bin/'r02_weighted_communities.py',
                '--evidence',combined.with_suffix('.npz'),'--output',self.run/f'communities_R_C_{panel}',
                '--samples',self.samples,'--settings',self.run/f'weighted_settings_{panel}.json',
                '--metadata',self.c['metadata'],'--metadata-sample-column','ID',
                '--metadata-color-column',self.c['metadata_color_column']])
            self.docker('all22_R_C_kinship_'+panel,['python3',self.bin/'r02_weighted_kinship.py',
                '--results-dir',self.run/f'communities_R_C_{panel}','--samples',self.samples,
                '--expected-samples','2619','--pcrelate-file',self.c['pcrelate'],
                '--expected-sha256',self.c['pcrelate_sha256'],'--thresholds','0.0221,0.0442',
                '--output-dir',self.run/f'diagnostics_R_C_{panel}_kinship'])
        for name in (out.name,'communities_H','plots_H','diagnostics_H','aggregate_R_C_primary',
                     'aggregate_R_C_sensitivity','communities_R_C_primary','communities_R_C_sensitivity',
                     'diagnostics_H_kinship_0221','diagnostics_H_kinship_0442',
                     'diagnostics_R_C_primary_kinship','diagnostics_R_C_sensitivity_kinship'):
            self.publish(self.run/name, '01_estructura/desarrollo/'+name)
        if evaluate_all:
            self.publish(self.run/'diagnostics_M14_all','01_estructura/desarrollo/diagnostics_M14_all')

    def execute(self):
        self.current_stage='VERIFYING_INPUTS'
        status(self.run, self.current_stage, state='RUNNING')
        self.verify_inputs(self.objects)
        for c in self.c['processing_order']:
            self.analyze_chromosome(c)
        self.aggregate()
        status(self.run,'COMPUTE_COMPLETE_PUBLICATION_PENDING',state='RUNNING')
        provenance=self.run/'provenance'
        provenance.mkdir(exist_ok=True)
        for f in ('run.json','source.sha256.json','frozen.sha256.json','samples.txt','input_objects.json','h_settings.json',
                  'weighted_settings_primary.json','weighted_settings_sensitivity.json','autosome_inputs.json'):
            shutil.copy2(self.run/f,provenance/f)
        if self.c.get('evaluate_all_m14_configurations',False):
            shutil.copy2(self.run/'m14_diagnostic_settings.json',provenance/'m14_diagnostic_settings.json')
        cleanup=provenance/'cleanup'
        cleanup.mkdir(exist_ok=True)
        for c in AUTOSOMES:
            for name in ('bulk_cleanup_plan.json','bulk_cleanup.json','local_common_cleanup.json'):
                item=self.run/f'chr{c:02d}'/name
                if item.exists(): shutil.copy2(item,cleanup/f'chr{c:02d}_{name}')
        if (self.run/'repairs').is_dir():
            shutil.copytree(self.run/'repairs', provenance/'operational_repairs', dirs_exist_ok=True)
        self.publish(provenance,'00_datos_y_diseno')
        status(self.run,'COMPLETE_PUBLISHED',state='COMPLETE',destination=uri(self.dest),
               interpretation=self.c['interpretation'],
               biological_validation_complete=False,
               scope='DESCRIPTIVE_NOT_CONFIRMATORY')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--analysis-image')
    args=p.parse_args()
    os.umask(0o077)
    if args.mode=='prepare':
        if not args.analysis_image or not re.fullmatch('sha256:[a-f0-9]{64}',args.analysis_image):
            p.error('An existing pinned analysis image is required')
        prepare(args.run_dir.resolve(),args.analysis_image)
    else:
        with execution_lock(args.run_dir.resolve()):
            runner=Runner(args.run_dir.resolve())
            try: runner.execute()
            except BaseException as e:
                status(args.run_dir,'FAILED',state='FAILED',failed_stage=runner.current_stage,error=str(e))
                raise


if __name__=='__main__': main()
