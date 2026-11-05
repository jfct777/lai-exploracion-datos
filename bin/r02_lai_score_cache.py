"""Reuse completed LAI prediction regions after interruption during scoring."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from r02_context_checkpoint import _atomic_bytes, _canonical, _require, _sha_file, _fsync_directory
from r02_context_update_checkpoint import _state_digest


class LaiScoreCache:
    def __init__(self, output, model, contract, resolved, stage, arm):
        self.output = Path(output)
        self.root = self.output / '.scoring'
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = dict(schema='r02_lai_score_regions_v1',
            model_sha256=_state_digest(model.state_dict()),
            contract_sha256=hashlib.sha256(_canonical(contract)).hexdigest(),
            resolved_sha256=hashlib.sha256(_canonical(resolved)).hexdigest(),
            stage=stage, arm=arm, code_sha256={name: _sha_file(Path(__file__).with_name(name))
                for name in ('r02_lai_score_cache.py', 'r02_context_train.py',
                             'genomic_context_runtime.py', 'r02_lai_common_context.py',
                             'r02_lai_query_event_attention.py', 'r02_local_attention_blocks.py',
                             'r02_lai_person_microbatch.py')})
        self.digest = hashlib.sha256(_canonical(self.identity)).hexdigest()
        path = self.root / 'IDENTITY.json'
        if path.exists():
            _require(json.loads(path.read_text()) == self.identity, 'scoring identity changed')
        else:
            _atomic_bytes(path, _canonical(self.identity))

    def load(self, index, entry):
        receipt = self.root / f'batch-{index:05d}.json'
        if not receipt.exists():
            return None
        row = json.loads(receipt.read_text())
        path = self.output / f'predictions-{index:05d}.npy'
        _require(row['identity_sha256'] == self.digest and row['input'] == entry
                 and _sha_file(path) == row['sha256'], 'cached scoring source or bytes changed')
        return np.load(path, allow_pickle=False)

    def save(self, index, entry, value):
        path = self.output / f'predictions-{index:05d}.npy'
        fd, temporary = tempfile.mkstemp(prefix='.prediction-', dir=self.output)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            digest = _sha_file(temporary)
            if path.exists():
                if _sha_file(path) != digest:
                    conflict = self.output / f'.conflict-{index:05d}-{digest}.npy'
                    if conflict.exists():
                        _require(_sha_file(conflict) == digest, 'conflict copy corrupted')
                    else:
                        os.replace(temporary, conflict)
                        _fsync_directory(self.output)
                    raise ValueError('unreceipted prediction differs; both preserved for diagnosis')
            else:
                os.replace(temporary, path)
                _fsync_directory(self.output)
            _atomic_bytes(self.root / f'batch-{index:05d}.json', _canonical(dict(
                identity_sha256=self.digest, input=entry, sha256=digest)))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
