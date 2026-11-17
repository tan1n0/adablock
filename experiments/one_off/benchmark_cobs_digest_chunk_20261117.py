"""Measure exact rank-4 digest batching for one Llama-3.1-8B layer at 32K.

The digest implementation and tensor shape match the deployed COBS index path.
Runs on one GPU, records synchronized wall times, peak allocated memory, and
max absolute difference from the existing 1024-block batch size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DIGEST_DIR = ROOT / "experiments/server_snapshots/ruler_autonomous_20261022_01"
sys.path.insert(0, str(DIGEST_DIR))
from fast_digest import compact_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.repeats < 1 or not args.chunks:
        parser.error("CUDA, positive repeats, and at least one chunk are required")
    torch.manual_seed(20261117)
    keys = torch.randn((8, 32768, 128), device="cuda", dtype=torch.bfloat16)
    reference = compact_digest(keys, chunk=1024)
    torch.cuda.synchronize()
    rows = []
    for chunk in args.chunks:
        torch.cuda.reset_peak_memory_stats()
        warmup = compact_digest(keys, chunk=chunk)
        torch.cuda.synchronize()
        errors = {name: (warmup[name] - reference[name]).abs().max().item()
                  for name in ("mean", "basis", "eigenvalues", "counts")}
        durations = []
        for _ in range(args.repeats):
            begin = time.perf_counter()
            result = compact_digest(keys, chunk=chunk)
            torch.cuda.synchronize()
            durations.append(1000 * (time.perf_counter() - begin))
        rows.append({"chunk": chunk, "median_ms_per_layer": statistics.median(durations),
                     "samples_ms": durations, "max_abs_diff_from_1024": errors,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576})
        del warmup, result
    output = {"seed": 20261117, "keys_shape": list(keys.shape), "keys_dtype": str(keys.dtype),
              "digest_source": str(DIGEST_DIR / "fast_digest.py"),
              "digest_sha256": hashlib.sha256((DIGEST_DIR / "fast_digest.py").read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(0), "repeats": args.repeats, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
