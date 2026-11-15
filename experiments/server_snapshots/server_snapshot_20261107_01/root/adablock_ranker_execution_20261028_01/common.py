"""Read-only reuse of frozen experiments; all artifacts belong to this directory."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/root/adablock_ruler_autonomous_20261022_01')
SMALL = Path('/root/adablock_small_ranker_20261028_01')
sys.path.insert(0, str(SOURCE))
import experiment as e
import torch

ORIGINAL = e.baseline.forward_packed
NAMES = ['original_S', 'original_B', 'H64_S', 'H64_B', 'P32H64_S', 'P32H64_B']


def dump(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(e.clean(value), indent=2), encoding='utf-8')
    tmp.replace(path)


def status(stage, **extra):
    dump('status.json', dict(stage=stage, time_unix=time.time(), **extra))


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_pids():
    out = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(x.strip()) for x in out.splitlines() if x.strip().isdigit()}


def exclusive_context():
    assert not gpu_pids(), 'GPU 0 is occupied; no competing measurement is allowed'
    probe = torch.zeros(1, device='cuda')
    owners = gpu_pids()
    assert len(owners) == 1, owners
    dump('gpu_identity.json', dict(container_pid=os.getpid(), nvml_pids=sorted(owners)))
    del probe
    return owners


def check_exclusive(owners):
    assert gpu_pids() == owners, 'GPU exclusivity lost; discard the measurement'


def load():
    model, tok, old, eos = e.load()
    models = {'original_' + n: old[n] for n in ['S', 'B']}
    del old
    for arch in ['H64', 'P32H64']:
        for branch in ['S', 'B']:
            models[arch + '_' + branch] = e.c.load_pair(
                SMALL / arch / 'branches' / branch / 'best_pair.pt', 'cuda')[:2]
    e.c.run.DynamicCache = e.BufferCache
    e.c.run.AdaptiveLlama = e.gpu.Controller
    e.baseline.compact_digest = e.fast_digest.compact_digest
    return model, tok, models, eos


def protect():
    assert json.loads((SMALL / 'status.json').read_text())['stage'] == 'complete'
    paths = list(SOURCE.glob('*.py')) + [SOURCE / 'selected.json', SOURCE / 'comparison.json',
        SMALL / 'comparison.json', SMALL / 'verification.json', Path(__import__('inspect').getfile(ORIGINAL))]
    paths += [e.OLD / 'branches' / n / 'best_pair.pt' for n in ['S', 'B']]
    paths += [SMALL / a / 'branches' / b / 'best_pair.pt' for a in ['H64', 'P32H64'] for b in ['S', 'B']]
    hashes = {str(p): sha(p) for p in paths}
    dump('protected.json', hashes)
    return hashes


def verify():
    hashes = json.loads((ROOT / 'protected.json').read_text())
    assert all(sha(p) == h for p, h in hashes.items()), 'Protected source or result changed'
    return dict(old_files_unchanged=True, protected_files=len(hashes))
