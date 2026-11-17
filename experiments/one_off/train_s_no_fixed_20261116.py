"""Rebuild no-sink/no-local Llama ranking labels and retrain the S ranker.

Uses the original 12 non-RULER train and 6 disjoint validation documents,
original pilot ranker warm start, 16-token blocks, two rank phases of 192
optimizer steps each, and S's mass-weighted RankLoss (rho=0.90). Every complete
historical block is a ranking candidate. Only the currently incomplete block
is fixed; no sink or completed local block is excluded. The budget classifier
is not trained because deployment uses fixed 512 historical tokens per head.

Commands: check, selftest, collect --split train|validation, train. Nothing is
launched in the background; collect/train are long GPU jobs for the user.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "results/llama_budget_rank_pilot_20261015_01"
OUTPUT = ROOT / "results/s_no_fixed_20261116"
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-359efdbb")
SEED = 20261019
BLOCK_SIZE = 16
SINK_BLOCKS = 0
LOCAL_BLOCKS = 0
COVERAGE = 0.90
SELECTED_K = 32
GROUPS_PER_STEP = 8
POSITIONS_PER_LAYER = 4
PHASES = 2
INITIAL_SHA256 = "376a0e09edb450de8bc0b4e891bcd7d38fe3c04d414baf8fa2369ccbe3aa81aa"
TRAIN_SHA256 = "19a96d43ba8940d091c678fd8f97c6c785bf04008cc6e837c795e6f9b9eb1473"
VALIDATION_SHA256 = "cc15ade45a15f6c7b5264f8a5d917d445626657cd2bb85b48090330d89aceb4e"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.block_ranker import load_ranker, save_ranker
from scripts.build_ranking_dataset import collect_document, ranking_metadata
from utils.ranking_batch import forward_packed, loss_from_pairs, pack_scenes, sample_batch_pairs
from utils.ranking_data import RankingDataset, RankingWriter, SCHEMA, assert_disjoint, check_metadata, document_identity
from utils.ranking_loss import PairSamplingConfig


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def source_path(split: str) -> Path:
    return PILOT / ("rank_train.jsonl" if split == "train" else "validation.jsonl")


def read_rows(split: str) -> list[dict]:
    return [json.loads(line) for line in source_path(split).read_text().splitlines() if line]


def check_inputs(args) -> dict:
    required = [source_path("train"), source_path("validation"), PILOT / "ranker/best.pt",
                args.model / "config.json"]
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing source: " + ", ".join(absent))
    if sha(PILOT / "ranker/best.pt") != INITIAL_SHA256:
        raise ValueError("Pilot ranker warm-start SHA-256 changed")
    if sha(source_path("train")) != TRAIN_SHA256 or sha(source_path("validation")) != VALIDATION_SHA256:
        raise ValueError("Original non-RULER train/validation JSONL changed")
    groups = {split: read_rows(split) for split in ("train", "validation")}
    if (len(groups["train"]), len(groups["validation"])) != (12, 6):
        raise ValueError("Expected the original 12/6 document split")
    identities = {}
    for split, rows in groups.items():
        ids = set()
        for row in rows:
            if not row.get("pilot_input_ids") or row["pilot_max_tokens"] not in (2048, 4096, 8192):
                raise ValueError(f"Invalid pilot tokens in {split}")
            if len(row["pilot_input_ids"]) != row["pilot_max_tokens"]:
                raise ValueError(f"Pilot token length mismatch in {split}")
            identity = document_identity(row, row["text"])
            ids.update(identity["components"])
        identities[split] = ids
    if identities["train"] & identities["validation"]:
        raise ValueError("Train/validation component overlap")
    return {"train_docs": 12, "validation_docs": 6,
            "train_sha256": sha(source_path("train")),
            "validation_sha256": sha(source_path("validation")),
            "pilot_ranker_sha256": INITIAL_SHA256,
            "lengths": [2048, 4096, 8192],
            "note": "No RULER test prompt enters training or checkpoint selection",
            "warm_start_note": "Pilot ranker was previously trained with sink/local; this controlled run changes the two S fine-tuning phases, not initialization"}


class NoFixedRankingWriter(RankingWriter):
    """Same shard schema, with all completed blocks eligible for ranking."""

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Use a new document directory: {self.directory}")
        if metadata["sink_blocks"] != 0 or metadata["local_window_blocks"] != 0:
            raise ValueError("This writer requires sink=local=0")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata, schema=SCHEMA)
        self.entries = []
        self.documents = []
        self._identities = set()
        self.current = None

    def add_query(self, position, query, block_mass, self_mass):
        block_size = self.metadata["block_size"]
        complete = position // block_size
        ids = torch.arange(complete, dtype=torch.long)
        if not len(ids):
            return
        mass = block_mass.detach().float().cpu()
        self_mass = self_mass.detach().float().cpu()
        if complete >= mass.shape[-1]:
            raise ValueError("Missing current block attention mass")
        teacher = mass[:, :complete].clone()
        # Current token and earlier tokens in its unfinished block are fixed.
        # At a block boundary this is just the current token; the last fully
        # completed block remains eligible, unlike local_window_blocks=1.
        fixed = mass[:, complete].clone()
        if not bool((fixed + 2e-4 >= self_mass).all()):
            raise ValueError("Current-block mass smaller than self attention")
        if not torch.allclose(fixed + teacher.sum(-1), torch.ones_like(fixed), atol=2e-4, rtol=0):
            raise ValueError("Fixed plus complete-block mass must equal one")
        if not bool(((ids + 1) * block_size <= position).all()):
            raise ValueError("Noncausal ranking candidate")
        q = query.detach().float().cpu()
        if not bool(q.isfinite().all()) or not bool(teacher.isfinite().all()):
            raise ValueError("Nonfinite ranking scene")
        self.current["records"].append({"token_position": int(position), "query": q,
                                         "block_ids": ids, "teacher_mass": teacher,
                                         "fixed_mass": fixed})


def selftest() -> None:
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as folder:
        metadata = {"block_size": 16, "sink_blocks": 0, "local_window_blocks": 0,
                    "head_dim": 4, "num_layers": 1, "num_kv_heads": 1,
                    "model_name": "synthetic", "model_signature": {}, "cobs_rank": 1,
                    "layers": [0], "query_scaling": "attention_scale_applied_once",
                    "rope": "post_rope", "gqa": "mean_of_per_query_softmax_block_mass"}
        writer = NoFixedRankingWriter(Path(folder) / "data", metadata)
        writer.begin_document({"source": "synthetic", "doc_id": "one", "components": ["synthetic::one"],
                               "text_sha256": "synthetic"})
        digests = SimpleNamespace(counts=torch.full((3,), 16.),mean=torch.zeros(1,3,4),
                                  minimum=torch.zeros(1,3,4),maximum=torch.zeros(1,3,4),
                                  covariance_basis=torch.zeros(1,3,1,4),
                                  covariance_eigenvalues=torch.zeros(1,3,1))
        writer.start_layer(0, digests)
        writer.add_query(18, torch.zeros(1, 1, 4), torch.tensor([[.25,.75,0.]]), torch.tensor([.5]))
        writer.add_query(32, torch.zeros(1, 1, 4), torch.tensor([[.2,.3,.5]]), torch.tensor([.5]))
        writer.end_layer()
        writer.finish()
        data = RankingDataset(writer.directory)
        first, second = data[0], data[1]
        assert first["block_ids"].tolist() == [0] and abs(float(first["fixed_mass"])-.75)<1e-6
        assert second["block_ids"].tolist() == [0,1] and abs(float(second["fixed_mass"])-.5)<1e-6
        print(json.dumps({"selftest": "passed", "partial_current_fixed": True,
                          "completed_recent_block_eligible": True}))


def collection_config(args, split: str, metadata: dict) -> dict:
    return {"split": split, "source_sha256": sha(source_path(split)),
            "model": str(args.model.resolve()), "seed": SEED, "block_size": BLOCK_SIZE,
            "sink_blocks": 0, "local_window_blocks": 0, "sample_stride": 256,
            "query_chunk_size": 8, "cobs_rank": 4, "metadata": metadata,
            "candidate_rule": "all complete blocks; current incomplete block fixed"}


def collect(args) -> None:
    check_inputs(args)
    from transformers import AutoModel
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for full-attention label collection")
    split = args.split
    root = args.output / "data" / split
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                     device_map={"": "cuda:0"}, attn_implementation="sdpa",
                                     local_files_only=True).eval()
    if model.config.model_type != "llama":
        raise ValueError("Expected the same Llama model as the pilot")
    metadata = ranking_metadata(model, str(args.model.resolve()), list(range(32)),
                                BLOCK_SIZE, 0, 0, 4)
    metadata.update(schema=SCHEMA, sample_stride=256, attention_backend="sdpa", base_dtype="bfloat16")
    planned = collection_config(args, split, metadata)
    config_path = root / "collection_protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != planned:
        raise ValueError("Collection protocol differs; use a new --output")
    if not config_path.exists():
        write_json(config_path, planned)
    rows = read_rows(split)
    for index, row in enumerate(rows):
        doc_dir = root / f"doc_{index:03d}"
        manifest_path = doc_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            check_metadata(metadata, existing["metadata"])
            if existing["documents"][0] != document_identity(row, row["text"]):
                raise ValueError(f"Existing document {index} identity differs")
            print(json.dumps({"split": split, "document": index, "status": "skip complete"}), flush=True)
            continue
        writer = NoFixedRankingWriter(doc_dir, metadata)
        identity = document_identity(row, row["text"])
        ids = torch.tensor([row["pilot_input_ids"]], device="cuda:0")
        collect_document(model, ids, writer, identity, sample_stride=256, query_chunk_size=8)
        result = writer.finish()
        print(json.dumps({"split": split, "document": index,
                          "tokens": len(row["pilot_input_ids"]),
                          "shards": len(result["shards"])}), flush=True)
    documents, shards = [], []
    for index, row in enumerate(rows):
        doc_name = f"doc_{index:03d}"
        info = json.loads((root / doc_name / "manifest.json").read_text())
        check_metadata(metadata, info["metadata"])
        documents.extend(info["documents"])
        for item in info["shards"]:
            shards.append(dict(item, file=f"{doc_name}/{item['file']}", document_index=index))
    write_json(root / "manifest.json", {"metadata": dict(metadata, schema=SCHEMA),
                                        "documents": documents, "shards": shards})
    print(json.dumps({"split": split, "status": "complete", "documents": len(documents),
                      "groups": sum(item["records"] for item in shards)}), flush=True)


def rank_groups(directory: Path) -> list[list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    groups = []
    for entry in manifest["shards"]:
        shard = torch.load(directory / entry["file"], map_location="cpu", weights_only=True)
        records = shard["records"]
        positions = torch.linspace(0, len(records)-1, min(POSITIONS_PER_LAYER,len(records))).round().long().unique()
        digests = shard["digests"]
        for position_index in positions.tolist():
            record = records[position_index]
            ids = record["block_ids"]
            if not torch.equal(ids, torch.arange(len(ids))):
                raise ValueError("No-fixed candidates must be contiguous from block zero")
            group = []
            for head in range(entry["heads"]):
                group.append({"query": record["query"][head],
                              "mean": digests["mean"][head, ids],
                              "basis": digests["covariance_basis"][head, ids],
                              "eigenvalues": digests["covariance_eigenvalues"][head, ids],
                              "counts": digests["counts"][ids],
                              "block_ids": ids, "block_size": BLOCK_SIZE,
                              "teacher_mass": record["teacher_mass"][head],
                              "fixed_mass": record["fixed_mass"][head],
                              "token_position": record["token_position"],
                              "layer_id": entry["layer_id"], "kv_head_id": head,
                              "document_index": entry["document_index"]})
            groups.append(group)
    return groups


@torch.no_grad()
def validation_coverage(ranker, groups: list[list[dict]], device) -> dict:
    ranker.eval()
    coverages, oracle_coverages = [], []
    for group in groups:
        packed = pack_scenes(group, device)
        scores = forward_packed(ranker, packed).view(len(group), -1)
        for head, scene in enumerate(group):
            k = min(SELECTED_K, len(scene["block_ids"]))
            selected = scores[head].argsort(descending=True, stable=True)[:k]
            oracle = scene["teacher_mass"].argsort(descending=True, stable=True)[:k]
            fixed = scene["fixed_mass"].to(device)
            mass = scene["teacher_mass"].to(device)
            coverages.append(float(fixed + mass[selected].sum()))
            oracle_coverages.append(float(fixed + mass[oracle].sum()))
    return {"coverage_at_32": statistics_mean(coverages),
            "oracle_coverage_at_32": statistics_mean(oracle_coverages),
            "scenes": len(coverages)}


def statistics_mean(values):
    return sum(values)/len(values) if values else 0.0


def train(args) -> None:
    check_inputs(args)
    train_dir, val_dir = args.output / "data/train", args.output / "data/validation"
    train_data, val_data = RankingDataset(train_dir), RankingDataset(val_dir)
    assert_disjoint(train_data, val_data)
    if (train_data.metadata["sink_blocks"], train_data.metadata["local_window_blocks"]) != (0,0):
        raise ValueError("Expected no-fixed train data")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for ranker training")
    out = args.output / "ranker"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new --output for training: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    ranker, initial = load_ranker(PILOT / "ranker/best.pt", device)
    ranker.requires_grad_(True)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=2e-4, weight_decay=.01)
    groups = rank_groups(train_dir)
    validation = rank_groups(val_dir)
    if len(groups) != 1536:
        raise ValueError(f"Expected 1536 original-style rank groups, got {len(groups)}")
    if not validation:
        raise ValueError("Empty held-out validation groups")
    pair_config = PairSamplingConfig(coverage=COVERAGE)
    metadata = train_data.metadata
    config = {"kind": "S_rank_only_no_fixed", "source_pilot_ranker_sha256": INITIAL_SHA256,
              "warm_start_has_old_fixed_training": True,
              "train_collection_sha256": sha(train_dir / "collection_protocol.json"),
              "validation_collection_sha256": sha(val_dir / "collection_protocol.json"),
              "train_docs": 12, "validation_docs": 6, "train_groups": len(groups),
              "validation_groups": len(validation), "rank_phases": 2,
              "expected_optimizer_steps": 384, "group_batch": GROUPS_PER_STEP,
              "coverage_target": COVERAGE, "fixed_history_blocks": 0,
              "current_incomplete_block_fixed": True, "deployment_history_budget_blocks": SELECTED_K,
              "rank_loss": "same mass-weighted RankLoss as original S; no boundary pairs",
              "budget_classifier_trained": False, "checkpoint_rule": "best validation coverage@32 among two rank phases"}
    write_json(out / "run_config.json", config)
    write_json(out / "initial_validation.json", validation_coverage(ranker, validation, device))
    history, best = [], -float("inf")
    for phase in range(PHASES):
        started = time.perf_counter()
        order = torch.randperm(len(groups), generator=torch.Generator().manual_seed(SEED+100+phase))
        pair_generator = torch.Generator().manual_seed(SEED+phase)
        ranker.train()
        pair_count = effective_batches = empty_batches = 0
        for indices in order.split(GROUPS_PER_STEP):
            scenes = [scene for index in indices.tolist() for scene in groups[index]]
            packed = pack_scenes(scenes, device)
            scores = forward_packed(ranker, packed)
            detached = scores.detach().float().cpu()
            pairs = sample_batch_pairs(detached, scenes, packed["offsets"], pair_config, pair_generator)
            loss, _ = loss_from_pairs(scores, scenes, packed["offsets"], pairs)
            pair_count += sum(len(pair) for pair in pairs)
            if loss is None:
                loss = scores.sum() * 0
                empty_batches += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            effective_batches += 1
        report = validation_coverage(ranker, validation, device)
        report.update({"phase": phase, "optimizer_steps": effective_batches,
                       "empty_batches": empty_batches, "pairs": pair_count,
                       "seconds": time.perf_counter()-started})
        history.append(report)
        save_ranker(out / f"phase_{phase}.pt", ranker, metadata, dict(config, **report))
        if report["coverage_at_32"] > best:
            best = report["coverage_at_32"]
            save_ranker(out / "best.pt", ranker, metadata, dict(config, **report))
        write_json(out / "history.json", history)
        print(json.dumps({"phase": phase, **report}), flush=True)
    total_steps = sum(row["optimizer_steps"] for row in history)
    if total_steps != 384:
        raise ValueError(f"Expected 384 effective optimizer steps, got {total_steps}")
    write_json(out / "TRAINING_COMPLETE.json", {"rank_steps": total_steps,
                                                   "checkpoint": "best.pt", "best_coverage_at_32": best})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "selftest", "collect", "train"))
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(check_inputs(args), ensure_ascii=False, indent=2))
    elif args.command == "selftest":
        selftest()
    elif args.command == "collect":
        if not args.split:
            parser.error("collect requires --split train or validation")
        collect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
