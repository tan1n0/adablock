"""Versioned document/layer shards with prefix-safe shared block summaries."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from utils.block_oracle import fixed_and_middle_blocks


SCHEMA = "adablock_cobs_ranking_v1"
COMPATIBILITY_KEYS = ("schema", "model_name", "model_signature", "block_size", "sink_blocks",
                      "local_window_blocks", "cobs_rank", "layers", "query_scaling", "rope", "gqa")


def model_signature(config):
    return {k: getattr(config, k, None) for k in (
        "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "rope_theta", "rope_scaling", "max_position_embeddings",
    )}


def check_metadata(expected, observed, keys=COMPATIBILITY_KEYS):
    mismatch = {k: (expected.get(k), observed.get(k)) for k in keys if expected.get(k) != observed.get(k)}
    if mismatch:
        raise ValueError(f"Incompatible ranking metadata: {mismatch}")


def document_identity(row, text):
    source, doc_id = str(row.get("source", "unknown")), str(row.get("doc_id", ""))
    components = list(row.get("component_doc_ids", []))
    if not components:
        components = [f"{source}::{doc_id or hashlib.sha256(text.encode()).hexdigest()}"]
    if any("longbench" in str(x).lower() for x in [source, *components]):
        raise ValueError("LongBench is reserved for downstream testing, not ranker training data.")
    return {"source": source, "doc_id": doc_id, "components": components,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


class RankingWriter:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Use a new output directory: {self.directory}")
        if metadata["local_window_blocks"] < 1:
            raise ValueError("v1 keeps the unfinished history block fixed: local_window_blocks >= 1 required.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata, schema=SCHEMA)
        self.entries = []
        self.documents = []
        self._identities = set()
        self.current = None

    def begin_document(self, identity):
        key = (identity["source"], identity["doc_id"], identity["text_sha256"])
        if key in self._identities:
            raise ValueError(f"Duplicate document: {key[:2]}")
        self._identities.add(key)
        self.documents.append(identity)

    def start_layer(self, layer_id, digests):
        if self.current is not None:
            raise RuntimeError("Previous ranking layer was not flushed.")
        tensors = {name: getattr(digests, name).detach().float().cpu() for name in (
            "counts", "mean", "minimum", "maximum", "covariance_basis", "covariance_eigenvalues",
        )}
        if not all(bool(t.isfinite().all()) for t in tensors.values()):
            raise FloatingPointError("Nonfinite ranking block summaries.")
        self.current = {"schema": SCHEMA, "layer_id": layer_id, "digests": tensors, "records": []}

    def add_query(self, position, query, block_mass, self_mass):
        meta = self.metadata
        bs = meta["block_size"]
        history_blocks = (position + bs - 1) // bs
        sink, local, middle = fixed_and_middle_blocks(
            history_blocks, meta["sink_blocks"], meta["local_window_blocks"], history_blocks - 1)
        ids = torch.tensor(sorted(middle), dtype=torch.long)
        # All stored/reused candidate descriptors must be for completed blocks.
        if ids.numel() and not bool(((ids + 1) * bs <= position).all()):
            raise ValueError("A candidate summary contains future/current-token keys.")
        if not ids.numel():
            return
        mass = block_mass[:, :history_blocks].detach().float().cpu().clone()
        self_mass = self_mass.detach().float().cpu()
        if position // bs < history_blocks:
            mass[:, position // bs] -= self_mass
        mass.clamp_min_(0)
        fixed = self_mass + mass[:, sorted(sink | local)].sum(-1)
        teacher = mass[:, ids]
        if not torch.allclose(fixed + teacher.sum(-1), torch.ones_like(fixed), atol=2e-4, rtol=0):
            raise ValueError("Fixed + candidate attention mass must equal one (self counted once).")
        query = query.detach().float().cpu()
        if not bool(query.isfinite().all()):
            raise FloatingPointError("Nonfinite ranking query.")
        self.current["records"].append({"token_position": int(position), "query": query,
                                         "block_ids": ids, "teacher_mass": teacher,
                                         "fixed_mass": fixed})

    def end_layer(self):
        shard = self.current
        self.current = None
        if not shard["records"]:
            return
        heads = int(shard["digests"]["mean"].shape[0])
        filename = f"doc_{len(self.documents)-1:06d}_layer_{shard['layer_id']:03d}.pt"
        target = self.directory / filename
        torch.save(shard, target.with_suffix(".tmp"))
        target.with_suffix(".tmp").replace(target)
        self.entries.append({"file": filename, "document_index": len(self.documents) - 1,
                             "layer_id": shard["layer_id"], "records": len(shard["records"]),
                             "heads": heads, "scenes": len(shard["records"]) * heads})

    def finish(self):
        if self.current is not None or not self.entries:
            raise ValueError("No complete ranking scenes; use longer texts or a smaller fixed window.")
        result = {"metadata": self.metadata, "documents": self.documents, "shards": self.entries}
        path = self.directory / "manifest.json"
        path.with_suffix(".tmp").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        path.with_suffix(".tmp").replace(path)
        return result


class RankingDataset(Dataset):
    def __init__(self, directory, cache_size=2):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.metadata = self.manifest["metadata"]
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("Unsupported ranking schema.")
        self.shards = self.manifest["shards"]
        self.ends = []
        total = 0
        for entry in self.shards:
            total += entry["scenes"]
            self.ends.append(total)
        self.cache = OrderedDict()
        self.cache_size = max(1, cache_size)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        offset = index - (self.ends[shard_index - 1] if shard_index else 0)
        entry = self.shards[shard_index]
        if shard_index not in self.cache:
            self.cache[shard_index] = torch.load(self.directory / entry["file"], weights_only=True, map_location="cpu")
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard_index)
        shard = self.cache[shard_index]
        record_index, head = divmod(offset, entry["heads"])
        record, d = shard["records"][record_index], shard["digests"]
        ids, t = record["block_ids"], record["token_position"]
        bs = self.metadata["block_size"]
        if not bool(((ids + 1) * bs <= t).all()):
            raise ValueError("Noncausal candidate in ranking shard.")
        return {"query": record["query"][head], "mean": d["mean"][head, ids],
                "basis": d["covariance_basis"][head, ids],
                "eigenvalues": d["covariance_eigenvalues"][head, ids],
                "minimum": d["minimum"][head, ids], "maximum": d["maximum"][head, ids],
                "counts": d["counts"][ids], "block_ids": ids, "block_size": bs,
                "teacher_mass": record["teacher_mass"][head], "fixed_mass": record["fixed_mass"][head],
                "token_position": t, "layer_id": entry["layer_id"], "kv_head_id": head,
                "document_index": entry["document_index"]}


def assert_disjoint(train, validation):
    check_metadata(train.metadata, validation.metadata)
    def identities(dataset):
        ids = set()
        for doc in dataset.manifest["documents"]:
            ids.add("text::" + doc["text_sha256"])
            ids.update("component::" + str(x) for x in doc["components"])
        return ids
    overlap = identities(train) & identities(validation)
    if overlap:
        raise ValueError(f"Training/validation document or component overlap: {sorted(overlap)[:5]}")


class ShardShuffleSampler(Sampler):
    """Shuffle scenes within shuffled shards, keeping IO bounded by shard size."""
    def __init__(self, dataset, generator):
        self.dataset, self.generator = dataset, generator

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for s in torch.randperm(len(self.dataset.shards), generator=self.generator).tolist():
            start = self.dataset.ends[s - 1] if s else 0
            for offset in torch.randperm(self.dataset.shards[s]["scenes"], generator=self.generator).tolist():
                yield start + offset
