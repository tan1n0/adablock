"""Create a frozen second-order COBS selector checkpoint for the same Llama protocol.

This is the paper's fixed mean + half covariance block-mass estimator, using
this repository's post-RoPE full-space rank-4 summary. It is NOT the paper's
complete NSA/NoPE/query-subspace/FP4 system or its original model checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT = ROOT / 'results/cobs_paper_core_same100_20261117/checkpoint'
COEFFICIENTS = [1.0, 0.5, 0.0, 0.0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'cobs_best.pt'
    meta = out / 'protocol.json'
    protocol = {
        'method_label': 'COBS paper-core fixed second-order selector, Llama adaptation',
        'paper': 'https://arxiv.org/abs/2607.09052',
        'formula': 'log mass ≈ log(16) + q_scaled^T mean_K + 0.5 q_scaled^T Sigma_rank4 q_scaled',
        'coefficients': COEFFICIENTS,
        'training': 'none; four coefficients frozen',
        # The evaluator checks this legacy protocol tag for selector compatibility.
        'variance': 'diagonal of rank-4 covariance approximation; same at train and inference',
        'index_summary': 'post-RoPE K mean, top-four covariance eigenvectors/eigenvalues, FP32 full 128D space',
        'gqa': 'per-query-head softmax over complete blocks, then mean across four query heads per KV head',
        'position_terms': 'zero; all complete blocks have 16 tokens so log(16) cancels',
        'not_paper_full_reproduction': 'frozen Llama rather than NSA model; post-RoPE rather than NoPE; no NSA compression/local branch, query-subspace projection, or FP4 descriptor',
        'code_source_sha256': sha(ROOT / 'models/cacheable_selectors.py'),
    }
    if target.exists() or meta.exists():
        if not (target.exists() and meta.exists()):
            raise FileExistsError('Checkpoint and protocol must either both exist or both be absent')
        old = json.loads(meta.read_text())
        if old != protocol:
            raise ValueError('Existing paper-core checkpoint protocol differs; use a new output directory')
        payload = torch.load(target, map_location='cpu', weights_only=True)
        if payload.get('model') != 'cobs' or payload.get('state_dict', {}).get('coefficients') is None:
            raise ValueError('Existing checkpoint is not COBS')
        if payload['state_dict']['coefficients'].tolist() != COEFFICIENTS:
            raise ValueError('Existing checkpoint coefficients differ')
        print(json.dumps({'checkpoint': str(target), 'sha256': sha(target), 'reused': True}))
        return
    from models.cacheable_selectors import SimpleCOBS
    model = SimpleCOBS().eval().requires_grad_(False)
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor(COEFFICIENTS, dtype=torch.float32))
    payload = {'model': 'cobs', 'config': {'coefficients': COEFFICIENTS},
               'state_dict': model.state_dict(), 'protocol': protocol}
    torch.save(payload, target)
    meta.write_text(json.dumps(protocol, indent=2))
    print(json.dumps({'checkpoint': str(target), 'sha256': sha(target), 'reused': False}))


if __name__ == '__main__':
    main()
