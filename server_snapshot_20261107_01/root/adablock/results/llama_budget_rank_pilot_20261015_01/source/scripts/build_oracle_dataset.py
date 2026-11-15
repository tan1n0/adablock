from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.block_oracle import (
    OracleConfig,
    aggregate_block_mass,
    arkvale_block_scores,
    bucket_index,
    cosine_block_scores,
    fixed_and_middle_blocks,
    make_block_ranges,
    reuse_label_from_previous_middle,
    quest_block_scores,
    score_summary_features,
    selector_oracle_budget_label,
    top_blocks,
)
from utils.qk_block_oracle import (
    QK_SCORE_SELECTORS,
    build_qk_block_digests,
    score_qk_block_digests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=32,
        help="Always-retained prefix tokens. Rounded up to complete blocks.",
    )
    parser.add_argument("--local-window-blocks", type=int, default=4)
    parser.add_argument(
        "--budget-buckets",
        nargs="+",
        type=int,
        default=[0, 4, 8, 16, 32, 64, 128],
        help="Dynamic middle-block budget buckets.",
    )
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--reuse-coverage", type=float, default=0.9)
    parser.add_argument(
        "--score-selector",
        choices=["mean", "cosine", "quest", "arkvale", "attention_topk"],
        default="cosine",
        help=(
            "Cheap block selector used for reuse labels, policy score_features, "
            "and selector-aware budget labels. 'mean' uses a mean centroid "
            "with cosine similarity; 'attention_topk' uses the full-attention "
            "block mass and is intended as an oracle upper-bound dataset."
        ),
    )
    parser.add_argument(
        "--retrieval-space",
        choices=["hidden", "post_rope_qk"],
        default="hidden",
        help=(
            "hidden preserves the legacy final-hidden selector. post_rope_qk "
            "builds selector-aware labels from the current token's per-layer, "
            "per-KV-head post-RoPE Q/K states."
        ),
    )
    parser.add_argument(
        "--qk-score-selectors",
        nargs="+",
        choices=list(QK_SCORE_SELECTORS),
        default=list(QK_SCORE_SELECTORS),
        help="Q/K selectors generated together in one model forward.",
    )
    parser.add_argument(
        "--primary-qk-selector",
        choices=list(QK_SCORE_SELECTORS),
        default="mean_pool",
        help="Q/K target mirrored into legacy top-level label fields.",
    )
    parser.add_argument("--cobs-rank", type=int, default=4)
    parser.add_argument("--phsa-lambda", type=float, default=0.5)
    parser.add_argument(
        "--budget-bucket-max",
        type=int,
        default=None,
        help="Use every integer budget bucket from 0 through this maximum.",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume a strictly aligned partial Oracle output.",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace an existing output file.",
    )
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument(
        "--transition-mode",
        choices=["causal_next_token"],
        default="causal_next_token",
        help="Use hidden[t-1] to predict the full-attention requirement of token t.",
    )
    parser.add_argument(
        "--anchor-offset-mode",
        choices=["document_hash", "fixed"],
        default="document_hash",
        help="Vary anchor phase per document or use --fixed-anchor-offset.",
    )
    parser.add_argument("--fixed-anchor-offset", type=int, default=1)
    parser.add_argument(
        "--oracle-layer-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional attention layer indices used to construct the oracle. Defaults to all layers.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=["eager_full", "sampled_recompute"],
        default="eager_full",
        help=(
            "eager_full returns complete attention matrices (legacy behavior). "
            "sampled_recompute runs the model with a memory-efficient backend and "
            "recomputes only sampled query rows for --oracle-layer-indices."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Base-model attention backend used by sampled_recompute.",
    )
    parser.add_argument(
        "--attention-query-chunk-size",
        type=int,
        default=8,
        help="Number of sampled query rows recomputed together per selected layer.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--device",
        default=None,
        help="Fallback device when device_map does not dispatch the model, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU oracle generation. Without this flag the script fails if CUDA is unavailable.",
    )
    parser.add_argument(
        "--skip-nonfinite-docs",
        action="store_true",
        help="Skip documents whose hidden states or attentions contain NaN/Inf.",
    )
    parser.add_argument("--log-every-doc", action="store_true")
    return parser.parse_args()


def load_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_device_map(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "false"}:
        return None
    return value


def reset_cuda_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def cuda_peak_memory_gib() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    return {
        f"cuda:{device_index}": round(
            torch.cuda.max_memory_allocated(device_index) / gib, 3
        )
        for device_index in range(torch.cuda.device_count())
    }


def iter_texts(path: Path, field: str, max_docs: int | None):
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_docs is not None and idx >= max_docs:
                break
            row = json.loads(line)
            text = row[field] if isinstance(row, dict) else str(row)
            if text.strip():
                source = row.get("source", "unknown") if isinstance(row, dict) else "unknown"
                source_doc_id = row.get("doc_id", idx) if isinstance(row, dict) else idx
                yield idx, text, source, source_doc_id


def oracle_row_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("source", "unknown")),
        str(row.get("source_doc_id", row.get("doc_id", "unknown"))),
        int(row["token_index"]),
    )


def anchor_offset(
    source: str,
    source_doc_id: str | int,
    stride: int,
    mode: str,
    fixed_offset: int,
) -> int:
    if mode == "fixed":
        return fixed_offset % stride
    identity = f"{source}\0{source_doc_id}".encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], byteorder="big") % stride


def causal_anchor_positions(
    seq_len: int,
    stride: int,
    block_size: int,
    offset: int,
) -> list[int]:
    first = offset if offset > 0 else stride
    return [
        target_t
        for target_t in range(first, seq_len, stride)
        if target_t >= max(2, block_size + 1)
    ]


def load_oracle_resume_progress(
    path: Path,
    expected_metadata: dict[str, object],
    score_selector: str = "cosine",
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], list[int]]]:
    document_order: list[tuple[str, str]] = []
    positions_by_document: dict[tuple[str, str], list[int]] = {}
    seen_keys: set[tuple[str, str, int]] = set()
    current_document: tuple[str, str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = oracle_row_key(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid partial Oracle row at {path}:{line_no}.") from exc
            if row.get("oracle_transition_version") != "causal_next_token_v1":
                raise ValueError(
                    f"Cannot resume non-causal Oracle row at {path}:{line_no}."
                )
            if int(row.get("oracle_transition_gap", -1)) != 1:
                raise ValueError(f"Cannot resume non-gap1 row at {path}:{line_no}.")
            expected_reuse_selector = f"causal_{score_selector}_topk"
            if row.get("reuse_previous_selector") != expected_reuse_selector:
                raise ValueError(
                    f"Cannot resume row with a different reuse selector at {path}:{line_no}."
                )
            observed_selector = row.get("oracle_score_selector", "cosine")
            if observed_selector != score_selector:
                raise ValueError(
                    f"Cannot resume row with score selector {observed_selector!r}; "
                    f"expected {score_selector!r} at {path}:{line_no}."
                )
            if not bool(row.get("oracle_causal_features", False)):
                raise ValueError(f"Cannot resume non-causal features at {path}:{line_no}.")
            if row.get("oracle_policy_input_mode") != "previous_hidden_only":
                raise ValueError(
                    f"Cannot resume a non-previous-hidden row at {path}:{line_no}."
                )
            for field, expected in expected_metadata.items():
                if field not in row:
                    raise ValueError(
                        f"Cannot resume row missing {field!r} at {path}:{line_no}."
                    )
                observed = row[field]
                if isinstance(expected, float):
                    try:
                        matches = abs(float(observed) - expected) <= 1e-9
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = observed == expected
                if not matches:
                    raise ValueError(
                        "Oracle resume configuration mismatch at "
                        f"{path}:{line_no}: {field}={observed!r}, "
                        f"expected {expected!r}."
                    )
            if key in seen_keys:
                raise ValueError(f"Duplicate Oracle key at {path}:{line_no}: {key}.")
            seen_keys.add(key)
            document = key[:2]
            if document != current_document:
                if document in positions_by_document:
                    raise ValueError(
                        f"Oracle document {document} is not contiguous in {path}."
                    )
                document_order.append(document)
                positions_by_document[document] = []
                current_document = document
            positions = positions_by_document[document]
            if positions and key[2] <= positions[-1]:
                raise ValueError(
                    f"Oracle token positions are not increasing at {path}:{line_no}."
                )
            positions.append(key[2])
    return document_order, positions_by_document


def select_oracle_attentions(attentions: torch.Tensor, layer_indices: list[int] | None) -> torch.Tensor:
    if not layer_indices:
        return attentions
    valid = [idx for idx in layer_indices if 0 <= idx < attentions.shape[0]]
    if not valid:
        raise ValueError(
            f"No valid oracle layers in {layer_indices}. Available layer range: 0..{attentions.shape[0] - 1}"
        )
    return attentions[valid]


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


PUNCTUATION_CHARACTERS = frozenset(".,!?;:，。！？；：、\n\r")


def punctuation_token_lookup(tokenizer) -> torch.Tensor:
    """Return a deterministic vocabulary lookup used by the PHSA digest."""

    vocabulary_size = max(
        int(getattr(tokenizer, "vocab_size", 0)),
        max(tokenizer.get_vocab().values(), default=-1) + 1,
    )
    lookup = torch.zeros(vocabulary_size, dtype=torch.bool)
    for token, token_id in tokenizer.get_vocab().items():
        if any(character in token for character in PUNCTUATION_CHARACTERS):
            lookup[int(token_id)] = True
    return lookup


class SampledAttentionCollector:
    """Recompute selected Qwen attention rows and aggregate them to blocks.

    The base model still performs its normal SDPA/Flash forward. Forward pre-hooks
    on selected self-attention modules use the normalized layer input and shared
    RoPE embeddings to recompute Q only at sampled token positions and K for the
    document. Token attention is immediately reduced over heads and blocks, so no
    full [heads, sequence, sequence] tensor is retained.
    """

    def __init__(
        self,
        layers,
        layer_indices: list[int],
        sample_positions: list[int],
        seq_len: int,
        block_size: int,
        query_chunk_size: int,
        qk_score_selectors: list[str] | None = None,
        qk_target_positions: list[int] | None = None,
        punctuation_mask: torch.Tensor | None = None,
        sink_blocks: int = 0,
        local_window_blocks: int = 0,
        budget_buckets: tuple[int, ...] = (),
        coverage_threshold: float = 0.95,
        reuse_coverage_threshold: float = 0.9,
        cobs_rank: int = 4,
        phsa_lambda: float = 0.5,
        ranking_writer=None,
    ) -> None:
        if not layer_indices:
            raise ValueError("sampled_recompute requires --oracle-layer-indices.")
        if query_chunk_size < 1:
            raise ValueError("--attention-query-chunk-size must be at least 1.")
        invalid = [idx for idx in layer_indices if not 0 <= idx < len(layers)]
        if invalid:
            raise ValueError(
                f"Invalid oracle layers {invalid}; available layer range is 0..{len(layers) - 1}."
            )
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("--oracle-layer-indices must not contain duplicates.")

        self.layer_indices = sorted(layer_indices)
        self.sample_positions = list(sample_positions)
        self.seq_len = int(seq_len)
        self.block_size = int(block_size)
        self.query_chunk_size = int(query_chunk_size)
        self.num_blocks = (self.seq_len + self.block_size - 1) // self.block_size
        self.position_to_index = {
            position: index for index, position in enumerate(self.sample_positions)
        }
        self.qk_score_selectors = list(qk_score_selectors or [])
        self.qk_target_positions = list(qk_target_positions or [])
        self.qk_target_position_set = set(self.qk_target_positions)
        self.qk_previous_position_set = {
            position - 1 for position in self.qk_target_positions
        }
        self.punctuation_mask = punctuation_mask
        self.sink_blocks = int(sink_blocks)
        self.local_window_blocks = int(local_window_blocks)
        self.budget_buckets = tuple(int(value) for value in budget_buckets)
        self.coverage_threshold = float(coverage_threshold)
        self.reuse_coverage_threshold = float(reuse_coverage_threshold)
        self.cobs_rank = int(cobs_rank)
        self.phsa_lambda = float(phsa_lambda)
        self.ranking_writer = ranking_writer
        self.max_dynamic_budget = max(self.budget_buckets, default=0)
        self.block_mass_sum = torch.zeros(
            len(self.sample_positions), self.num_blocks, dtype=torch.float32
        )
        self.self_mass_sum = torch.zeros(
            len(self.sample_positions), dtype=torch.float32
        )
        self.qk_coverage_sum = {
            selector: torch.zeros(
                len(self.sample_positions), self.num_blocks + 1, dtype=torch.float32
            )
            for selector in self.qk_score_selectors
        }
        self.qk_reuse_coverage_sum = {
            selector: torch.zeros(
                len(self.qk_target_positions), self.max_dynamic_budget + 1,
                dtype=torch.float32,
            )
            for selector in self.qk_score_selectors
        }
        self.qk_target_to_index = {
            position: index for index, position in enumerate(self.qk_target_positions)
        }
        self.qk_previous_rankings: dict[str, dict[int, torch.Tensor]] = {
            selector: {} for selector in self.qk_score_selectors
        }
        self.layer_calls: list[int] = []
        self.handles = []
        for layer_idx in self.layer_indices:
            module = layers[layer_idx].self_attn
            handle = module.register_forward_pre_hook(
                self._make_hook(layer_idx), with_kwargs=True
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, positional_args, keyword_args):
            hidden_states = keyword_args.get("hidden_states")
            if hidden_states is None and positional_args:
                hidden_states = positional_args[0]
            position_embeddings = keyword_args.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError(
                    "Could not capture Qwen hidden_states/position_embeddings for sampled attention."
                )
            if hidden_states.shape[0] != 1 or hidden_states.shape[1] != self.seq_len:
                raise RuntimeError(
                    "sampled_recompute currently expects one unpadded document per forward; "
                    f"got hidden shape={tuple(hidden_states.shape)}."
                )
            self._accumulate_layer(module, hidden_states, position_embeddings)
            self.layer_calls.append(layer_idx)

        return hook

    @torch.no_grad()
    def _accumulate_layer(self, module, hidden_states, position_embeddings) -> None:
        device = hidden_states.device
        positions = torch.tensor(self.sample_positions, dtype=torch.long, device=device)
        cos, sin = position_embeddings

        key_states = module.k_proj(hidden_states)
        key_states = key_states.view(
            1,
            self.seq_len,
            module.config.num_key_value_heads,
            module.head_dim,
        ).transpose(1, 2)
        key_cos = cos.unsqueeze(1)
        key_sin = sin.unsqueeze(1)
        key_states = (key_states * key_cos) + (rotate_half(key_states) * key_sin)
        qk_digests = None
        if self.qk_score_selectors or self.ranking_writer is not None:
            qk_digests = build_qk_block_digests(
                key_states[0],
                block_size=self.block_size,
                punctuation_mask=self.punctuation_mask,
                cobs_rank=self.cobs_rank,
            )
        if self.ranking_writer is not None:
            self.ranking_writer.start_layer(self.layer_indices[len(self.layer_calls)], qk_digests)

        for start in range(0, len(self.sample_positions), self.query_chunk_size):
            end = min(start + self.query_chunk_size, len(self.sample_positions))
            chunk_positions = positions[start:end]
            query_hidden = hidden_states.index_select(1, chunk_positions)
            query_states = module.q_proj(query_hidden)
            query_states = query_states.view(
                1,
                end - start,
                module.config.num_attention_heads,
                module.head_dim,
            ).transpose(1, 2)
            query_cos = cos.index_select(1, chunk_positions).unsqueeze(1)
            query_sin = sin.index_select(1, chunk_positions).unsqueeze(1)
            query_states = (query_states * query_cos) + (
                rotate_half(query_states) * query_sin
            )

            key_length = int(chunk_positions[-1].item()) + 1
            chunk_keys = key_states[:, :, :key_length, :]
            query_states = query_states.view(
                1,
                module.config.num_key_value_heads,
                module.num_key_value_groups,
                end - start,
                module.head_dim,
            )
            logits = torch.einsum(
                "bngqd,bnkd->bngqk", query_states, chunk_keys
            ) * module.scaling
            key_positions = torch.arange(key_length, device=device)
            causal = key_positions.unsqueeze(0) <= chunk_positions.unsqueeze(1)
            logits.masked_fill_(
                ~causal.view(1, 1, 1, end - start, key_length),
                torch.finfo(logits.dtype).min,
            )
            weights = F.softmax(logits, dim=-1, dtype=torch.float32)
            if self.ranking_writer is None:
                weights = weights.to(query_states.dtype)
            mean_attention = weights[0].float().mean(dim=(0, 1))
            local_query_indices = torch.arange(
                end - start, dtype=torch.long, device=device
            )
            self_mass = mean_attention[local_query_indices, chunk_positions]
            padded_length = self.num_blocks * self.block_size
            mean_attention = F.pad(
                mean_attention, (0, padded_length - key_length)
            )
            block_mass = mean_attention.view(
                end - start, self.num_blocks, self.block_size
            ).sum(dim=-1)
            self.block_mass_sum[start:end].add_(block_mass.cpu())
            self.self_mass_sum[start:end].add_(self_mass.cpu())

            if qk_digests is not None:
                qk_scores = score_qk_block_digests(
                    query_states[0] * float(module.scaling),
                    qk_digests,
                    selectors=self.qk_score_selectors,
                    phsa_lambda=self.phsa_lambda,
                )
                per_kv_head_attention = weights[0].float().mean(dim=1)
                padded_per_head = F.pad(
                    per_kv_head_attention,
                    (0, padded_length - key_length),
                )
                per_head_block_mass = padded_per_head.view(
                    module.config.num_key_value_heads,
                    end - start,
                    self.num_blocks,
                    self.block_size,
                ).sum(dim=-1)
                per_head_self_mass = per_kv_head_attention[
                    :, local_query_indices, chunk_positions
                ]
                for local_index, position_tensor in enumerate(chunk_positions):
                    position = int(position_tensor.item())
                    if self.ranking_writer is not None:
                        self.ranking_writer.add_query(
                            position, query_states[0, :, :, local_index].float() * float(module.scaling),
                            per_head_block_mass[:, local_index], per_head_self_mass[:, local_index],
                        )
                    self._accumulate_qk_position(
                        position=position,
                        position_index=start + local_index,
                        scores={
                            selector: values[local_index]
                            for selector, values in qk_scores.items()
                        },
                        block_mass=per_head_block_mass[:, local_index],
                        self_mass=per_head_self_mass[:, local_index],
                    )
            del (
                query_hidden,
                query_states,
                chunk_keys,
                logits,
                weights,
                mean_attention,
                self_mass,
                block_mass,
            )
            if qk_digests is not None:
                del qk_scores, per_kv_head_attention, padded_per_head
                del per_head_block_mass, per_head_self_mass

        if self.ranking_writer is not None:
            self.ranking_writer.end_layer()
        del key_states, qk_digests

    def _accumulate_qk_position(
        self,
        position: int,
        position_index: int,
        scores: dict[str, torch.Tensor],
        block_mass: torch.Tensor,
        self_mass: torch.Tensor,
    ) -> None:
        """Accumulate one layer's selector coverage curves for one query."""

        history_blocks = (position + self.block_size - 1) // self.block_size
        current_history_block = history_blocks - 1
        sink, local, middle = fixed_and_middle_blocks(
            num_blocks=history_blocks,
            sink_blocks=self.sink_blocks,
            local_window_blocks=self.local_window_blocks,
            current_block=current_history_block,
        )
        fixed = sink | local
        mass = block_mass[:, :history_blocks].clone()
        self_block = position // self.block_size
        if self_block < history_blocks:
            mass[:, self_block] -= self_mass
            mass.clamp_min_(0.0)
        if fixed:
            fixed_index = torch.tensor(sorted(fixed), dtype=torch.long, device=mass.device)
            base = self_mass + mass.index_select(1, fixed_index).sum(dim=1)
        else:
            base = self_mass.clone()
        middle_index = torch.tensor(
            sorted(middle), dtype=torch.long, device=mass.device
        )

        for selector, selector_scores in scores.items():
            if middle_index.numel() == 0:
                ordered = torch.empty(
                    mass.shape[0], 0, dtype=torch.long, device=mass.device
                )
                curve = base.mean().view(1)
            else:
                order_in_middle = torch.argsort(
                    selector_scores[:, middle_index], dim=-1, descending=True
                )
                ordered = middle_index[order_in_middle]
                selected_mass = torch.gather(mass, dim=1, index=ordered)
                curve = torch.cat(
                    [base.unsqueeze(1), base.unsqueeze(1) + selected_mass.cumsum(dim=1)],
                    dim=1,
                ).mean(dim=0)
            target_curve = self.qk_coverage_sum[selector][position_index]
            target_curve[: curve.numel()].add_(curve.detach().cpu())
            if curve.numel() < target_curve.numel():
                target_curve[curve.numel() :].add_(float(curve[-1].item()))

            if position in self.qk_previous_position_set:
                self.qk_previous_rankings[selector][position] = ordered[
                    :, : self.max_dynamic_budget
                ].detach().cpu()

            if position in self.qk_target_position_set:
                previous = self.qk_previous_rankings[selector].get(position - 1)
                if previous is None:
                    raise RuntimeError(
                        f"Missing Q/K ranking for previous position {position - 1}."
                    )
                previous = previous.to(mass.device)
                if previous.shape[1] == 0:
                    reuse_curve = base.mean().view(1)
                else:
                    valid = previous < history_blocks
                    if fixed:
                        for fixed_block in fixed:
                            valid &= previous != fixed_block
                    safe_previous = previous.clamp(max=max(history_blocks - 1, 0))
                    reused_mass = torch.gather(mass, dim=1, index=safe_previous)
                    reused_mass = reused_mass * valid.to(reused_mass.dtype)
                    reuse_curve = torch.cat(
                        [
                            base.unsqueeze(1),
                            base.unsqueeze(1) + reused_mass.cumsum(dim=1),
                        ],
                        dim=1,
                    ).mean(dim=0)
                target_index = self.qk_target_to_index[position]
                target_reuse = self.qk_reuse_coverage_sum[selector][target_index]
                take = min(reuse_curve.numel(), target_reuse.numel())
                target_reuse[:take].add_(reuse_curve[:take].detach().cpu())
                if take < target_reuse.numel():
                    target_reuse[take:].add_(float(reuse_curve[-1].item()))

    def finish(self) -> torch.Tensor:
        result, _ = self.finish_with_self_mass()
        return result

    def finish_with_self_mass(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.remove()
        if self.layer_calls != self.layer_indices:
            raise RuntimeError(
                "Selected attention hooks did not run exactly once in layer order: "
                f"expected={self.layer_indices}, observed={self.layer_calls}."
            )
        result = self.block_mass_sum / float(len(self.layer_indices))
        self_mass = self.self_mass_sum / float(len(self.layer_indices))
        if not torch.isfinite(result).all() or not torch.isfinite(self_mass).all():
            raise FloatingPointError("Sampled block attention contains NaN/Inf.")
        return result, self_mass

    def finish_qk_targets(self) -> dict[int, dict[str, dict[str, int | bool | float]]]:
        if not self.qk_score_selectors:
            return {}
        if self.layer_calls != self.layer_indices:
            raise RuntimeError("Q/K targets requested before all selected layers completed.")
        layer_count = float(len(self.layer_indices))
        targets: dict[int, dict[str, dict[str, int | bool | float]]] = {}
        coverage_floor = self.coverage_threshold - 1e-7
        for position in self.qk_target_positions:
            position_index = self.position_to_index[position]
            previous_position_index = self.position_to_index[position - 1]
            target_index = self.qk_target_to_index[position]
            method_targets: dict[str, dict[str, int | bool | float]] = {}
            for selector in self.qk_score_selectors:
                curve = self.qk_coverage_sum[selector][position_index] / layer_count
                previous_curve = (
                    self.qk_coverage_sum[selector][previous_position_index]
                    / layer_count
                )
                required_k = self._required_k(curve, coverage_floor)
                previous_required_k = self._required_k(previous_curve, coverage_floor)
                budget_label = bucket_index(required_k, self.budget_buckets)
                previous_budget_label = bucket_index(
                    previous_required_k, self.budget_buckets
                )
                budget_bucket_k = int(self.budget_buckets[budget_label])
                previous_budget_bucket_k = int(
                    self.budget_buckets[previous_budget_label]
                )
                history_blocks = (position + self.block_size - 1) // self.block_size
                _sink, _local, middle = fixed_and_middle_blocks(
                    history_blocks,
                    self.sink_blocks,
                    self.local_window_blocks,
                    history_blocks - 1,
                )
                previous_history_blocks = (
                    position - 1 + self.block_size - 1
                ) // self.block_size
                _ps, _pl, previous_middle = fixed_and_middle_blocks(
                    previous_history_blocks,
                    self.sink_blocks,
                    self.local_window_blocks,
                    previous_history_blocks - 1,
                )
                allocated_k = min(budget_bucket_k, len(middle))
                previous_allocated_k = min(
                    previous_budget_bucket_k, len(previous_middle)
                )
                reuse_curve = (
                    self.qk_reuse_coverage_sum[selector][target_index]
                    / layer_count
                )
                reuse_lookup = min(previous_allocated_k, reuse_curve.numel() - 1)
                method_targets[selector] = {
                    "score_selector": selector,
                    "budget_label_selector": (
                        f"post_rope_qk_{selector}_topk_attention_coverage"
                    ),
                    "reuse_previous_selector": f"causal_{selector}_topk",
                    "budget_label": budget_label,
                    "required_k": required_k,
                    "budget_bucket_k": budget_bucket_k,
                    "allocated_k": allocated_k,
                    "budget_saturated": required_k > allocated_k,
                    "reuse_label": int(
                        float(reuse_curve[reuse_lookup])
                        >= self.reuse_coverage_threshold - 1e-7
                    ),
                    "reuse_coverage": float(reuse_curve[reuse_lookup]),
                    "previous_budget_label": previous_budget_label,
                    "previous_required_k": previous_required_k,
                    "previous_budget_bucket_k": previous_budget_bucket_k,
                    "previous_allocated_k": previous_allocated_k,
                    "previous_budget_saturated": (
                        previous_required_k > previous_allocated_k
                    ),
                }
            targets[position] = method_targets
        return targets

    @staticmethod
    def _required_k(curve: torch.Tensor, threshold: float) -> int:
        reached = torch.nonzero(curve >= threshold, as_tuple=False)
        if reached.numel() == 0:
            return int(curve.numel() - 1)
        return int(reached[0, 0].item())

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main() -> None:
    args = parse_args()
    if args.retrieval_space == "post_rope_qk" and args.oracle_layer_indices is None:
        # The default model is Qwen2.5-7B (28 decoder layers).  Explicit layer
        # indices are still recorded and validated after loading the model.
        args.oracle_layer_indices = list(range(28))
    if args.retrieval_space == "post_rope_qk":
        if args.attention_mode != "sampled_recompute":
            raise ValueError("post_rope_qk retrieval requires --attention-mode sampled_recompute.")
        if args.primary_qk_selector not in args.qk_score_selectors:
            raise ValueError("--primary-qk-selector must be included in --qk-score-selectors.")
        if len(set(args.qk_score_selectors)) != len(args.qk_score_selectors):
            raise ValueError("--qk-score-selectors must not contain duplicates.")
        if args.cobs_rank < 0:
            raise ValueError("--cobs-rank must be non-negative.")
        if not 0.0 <= args.phsa_lambda <= 1.0:
            raise ValueError("--phsa-lambda must be in [0, 1].")
    if args.budget_bucket_max is not None:
        if args.budget_bucket_max < 0:
            raise ValueError("--budget-bucket-max must be non-negative.")
        args.budget_buckets = list(range(args.budget_bucket_max + 1))
    if args.block_size < 1:
        raise ValueError("--block-size must be at least 1.")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be at least 1.")
    if args.attention_query_chunk_size < 1:
        raise ValueError("--attention-query-chunk-size must be at least 1.")
    if not 0.0 < args.coverage <= 1.0:
        raise ValueError("--coverage must be in (0, 1].")
    if not 0.0 < args.reuse_coverage <= 1.0:
        raise ValueError("--reuse-coverage must be in (0, 1].")
    if args.sink_tokens < 0:
        raise ValueError("--sink-tokens must be non-negative.")
    if args.local_window_blocks < 0:
        raise ValueError("--local-window-blocks must be non-negative.")
    if not args.budget_buckets:
        raise ValueError("--budget-buckets must contain at least one value.")
    if any(bucket < 0 for bucket in args.budget_buckets):
        raise ValueError("--budget-buckets values must be non-negative.")
    if args.budget_buckets != sorted(set(args.budget_buckets)):
        raise ValueError(
            "--budget-buckets must be strictly increasing without duplicates."
        )
    if args.attention_mode == "sampled_recompute" and not args.oracle_layer_indices:
        raise ValueError(
            "--attention-mode sampled_recompute requires --oracle-layer-indices."
        )
    if args.oracle_layer_indices:
        if len(set(args.oracle_layer_indices)) != len(args.oracle_layer_indices):
            raise ValueError("--oracle-layer-indices must not contain duplicates.")
        args.oracle_layer_indices = sorted(args.oracle_layer_indices)
    sink_blocks = (args.sink_tokens + args.block_size - 1) // args.block_size
    config = OracleConfig(
        block_size=args.block_size,
        sink_blocks=sink_blocks,
        local_window_blocks=args.local_window_blocks,
        budget_buckets=tuple(args.budget_buckets),
        coverage_threshold=args.coverage,
        reuse_coverage_threshold=args.reuse_coverage,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_existed = output_path.exists()
    if output_existed and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --resume for an aligned partial "
            "causal Oracle or --overwrite intentionally."
        )
    existing_document_order: list[tuple[str, str]] = []
    existing_positions_by_document: dict[tuple[str, str], list[int]] = {}
    model_attention_backend = (
        "eager" if args.attention_mode == "eager_full" else args.attn_implementation
    )
    resume_metadata = {
        "oracle_block_size": args.block_size,
        "oracle_sink_tokens": args.sink_tokens,
        "oracle_local_window_blocks": args.local_window_blocks,
        "oracle_budget_buckets": list(config.budget_buckets),
        "oracle_coverage_threshold": config.coverage_threshold,
        "oracle_reuse_coverage_threshold": config.reuse_coverage_threshold,
        "oracle_attention_mode": args.attention_mode,
        "oracle_layer_indices": args.oracle_layer_indices,
        "oracle_attn_implementation": model_attention_backend,
        "oracle_sample_stride": args.sample_stride,
        "oracle_anchor_offset_mode": args.anchor_offset_mode,
    }
    if args.retrieval_space == "post_rope_qk":
        resume_metadata.update(
            {
                "oracle_retrieval_space": "post_rope_qk",
                "qk_score_selectors": list(args.qk_score_selectors),
                "qk_cobs_rank": args.cobs_rank,
                "qk_phsa_lambda": args.phsa_lambda,
            }
        )
    if args.resume and output_existed:
        resume_score_selector = (
            args.primary_qk_selector
            if args.retrieval_space == "post_rope_qk"
            else args.score_selector
        )
        existing_document_order, existing_positions_by_document = (
            load_oracle_resume_progress(
                output_path, resume_metadata, score_selector=resume_score_selector
            )
        )
        print(
            {
                "event": "oracle_resume_loaded",
                "documents": len(existing_document_order),
                "rows": sum(map(len, existing_positions_by_document.values())),
                "output_jsonl": str(output_path),
            },
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    punctuation_lookup = (
        punctuation_token_lookup(tokenizer)
        if args.retrieval_space == "post_rope_qk" and "phsa" in args.qk_score_selectors
        else None
    )
    cuda_available = torch.cuda.is_available()
    print(
        {
            "event": "torch_device_check",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count(),
        },
        flush=True,
    )
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this Python environment, so oracle generation would run on CPU. "
            "Install a CUDA-enabled PyTorch build or pass --allow-cpu intentionally."
        )

    device_map = normalize_device_map(args.device_map)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype(args.dtype),
        device_map=device_map,
        attn_implementation=model_attention_backend,
    )
    if getattr(model, "hf_device_map", None) is None:
        fallback_device = args.device or ("cuda:0" if cuda_available else "cpu")
        model.to(fallback_device)
    model.eval()
    transformer = getattr(model, "model", model)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise TypeError(
            "This Oracle builder currently requires a Qwen/Llama-style model with model.layers."
        )
    if args.oracle_layer_indices:
        invalid_layers = [
            layer_idx
            for layer_idx in args.oracle_layer_indices
            if not 0 <= layer_idx < len(layers)
        ]
        if invalid_layers:
            raise ValueError(
                f"Invalid oracle layers {invalid_layers}; model has {len(layers)} layers."
            )
    print(
        {
            "event": "model_loaded",
            "model": args.model_name,
            "device_map": getattr(model, "hf_device_map", None),
            "model_device": str(model.device),
            "attention_mode": args.attention_mode,
            "transition_mode": args.transition_mode,
            "attn_implementation": model_attention_backend,
            "attention_query_chunk_size": args.attention_query_chunk_size,
            "oracle_layer_indices": args.oracle_layer_indices,
            "sink_blocks": config.sink_blocks,
            "local_window_blocks": config.local_window_blocks,
            "budget_buckets": config.budget_buckets,
        },
        flush=True,
    )

    output_mode = "a" if args.resume and output_existed else "w"
    resumed_document_cursor = 0
    written_rows = 0
    skipped_rows = 0
    skipped_documents = 0
    with output_path.open(output_mode, encoding="utf-8") as out:
        text_iter = iter_texts(Path(args.input_jsonl), args.text_field, args.max_docs)
        for doc_id, text, source, source_doc_id in tqdm(
            text_iter, desc="building oracle", unit="doc"
        ):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            seq_len = int(input_ids.shape[-1])
            if seq_len < args.block_size * 2:
                continue
            block_ranges = make_block_ranges(seq_len, args.block_size)
            document_anchor_offset = anchor_offset(
                source=str(source),
                source_doc_id=source_doc_id,
                stride=args.sample_stride,
                mode=args.anchor_offset_mode,
                fixed_offset=args.fixed_anchor_offset,
            )
            sample_positions = causal_anchor_positions(
                seq_len=seq_len,
                stride=args.sample_stride,
                block_size=args.block_size,
                offset=document_anchor_offset,
            )
            if not sample_positions:
                continue
            document_key = (str(source), str(source_doc_id))
            existing_positions: list[int] = []
            if resumed_document_cursor < len(existing_document_order):
                expected_document = existing_document_order[resumed_document_cursor]
                if document_key != expected_document:
                    raise ValueError(
                        "Partial Oracle is not an input-document prefix: "
                        f"expected={expected_document}, observed={document_key}."
                    )
                existing_positions = existing_positions_by_document[document_key]
                resumed_document_cursor += 1
                if existing_positions != sample_positions[: len(existing_positions)]:
                    raise ValueError(
                        f"Partial Oracle positions are not an anchor prefix for {document_key}."
                    )
            if len(existing_positions) == len(sample_positions):
                skipped_documents += 1
                skipped_rows += len(existing_positions)
                if args.log_every_doc:
                    print(
                        {
                            "event": "doc_resume_skipped",
                            "doc_id": doc_id,
                            "seq_len": seq_len,
                            "rows": len(existing_positions),
                        },
                        flush=True,
                    )
                continue
            sample_positions = sample_positions[len(existing_positions) :]
            attention_positions = (
                sorted(
                    {
                        position
                        for anchor in sample_positions
                        for position in (anchor - 1, anchor)
                    }
                )
                if args.transition_mode == "causal_next_token"
                else sample_positions
            )
            attention_position_to_index = {
                position: index for index, position in enumerate(attention_positions)
            }
            if args.log_every_doc:
                print(
                    {
                        "event": "forward_start",
                        "doc_id": doc_id,
                        "seq_len": seq_len,
                        "anchor_queries": len(sample_positions),
                        "attention_queries": len(attention_positions),
                        "attention_mode": args.attention_mode,
                        "transition_mode": args.transition_mode,
                        "anchor_offset": document_anchor_offset,
                    },
                    flush=True,
                )
            reset_cuda_peak_memory()

            collector = None
            qk_targets: dict[int, dict[str, dict[str, int | bool | float]]] = {}
            try:
                if args.attention_mode == "sampled_recompute":
                    punctuation_mask = (
                        punctuation_lookup[input_ids[0].detach().cpu()]
                        if punctuation_lookup is not None
                        else None
                    )
                    collector = SampledAttentionCollector(
                        layers=layers,
                        layer_indices=args.oracle_layer_indices,
                        sample_positions=attention_positions,
                        seq_len=seq_len,
                        block_size=args.block_size,
                        query_chunk_size=args.attention_query_chunk_size,
                        qk_score_selectors=(
                            args.qk_score_selectors
                            if args.retrieval_space == "post_rope_qk"
                            else None
                        ),
                        qk_target_positions=(
                            sample_positions
                            if args.retrieval_space == "post_rope_qk"
                            else None
                        ),
                        punctuation_mask=punctuation_mask,
                        sink_blocks=config.sink_blocks,
                        local_window_blocks=config.local_window_blocks,
                        budget_buckets=config.budget_buckets,
                        coverage_threshold=config.coverage_threshold,
                        reuse_coverage_threshold=config.reuse_coverage_threshold,
                        cobs_rank=args.cobs_rank,
                        phsa_lambda=args.phsa_lambda,
                    )
                with torch.no_grad():
                    outputs = transformer(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=args.attention_mode == "eager_full",
                        output_hidden_states=False,
                        use_cache=False,
                    )
            except Exception:
                if collector is not None:
                    collector.remove()
                raise
            if args.log_every_doc:
                print(
                    {
                        "event": "forward_done",
                        "doc_id": doc_id,
                        "seq_len": seq_len,
                        "cuda_peak_allocated_gib": cuda_peak_memory_gib(),
                    },
                    flush=True,
                )

            hidden = outputs.last_hidden_state[0].detach().cpu().float()
            attentions = None
            oracle_attentions = None
            mean_attention = None
            sampled_block_mass = None
            sampled_self_mass = None
            if args.attention_mode == "eager_full":
                attentions = torch.stack(
                    [attn[0].detach().cpu().float() for attn in outputs.attentions]
                )
                oracle_attentions = select_oracle_attentions(
                    attentions, args.oracle_layer_indices
                )
                mean_attention = oracle_attentions.mean(dim=(0, 1))
            else:
                if args.retrieval_space == "post_rope_qk":
                    qk_targets = collector.finish_qk_targets()
                sampled_block_mass, sampled_self_mass = collector.finish_with_self_mass()
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attention_finite = (
                torch.isfinite(attentions).all()
                if attentions is not None
                else (
                    torch.isfinite(sampled_block_mass).all()
                    and torch.isfinite(sampled_self_mass).all()
                )
            )
            if not torch.isfinite(hidden).all() or not attention_finite:
                message = {
                    "event": "nonfinite_forward_output",
                    "doc_id": doc_id,
                    "seq_len": seq_len,
                    "hidden_finite": bool(torch.isfinite(hidden).all().item()),
                    "attentions_finite": bool(attention_finite.item()),
                }
                if args.skip_nonfinite_docs:
                    print(message | {"action": "skip"}, flush=True)
                    continue
                raise FloatingPointError(
                    f"Model output contains NaN/Inf: {message}. "
                    "Regenerate with CUDA, a smaller max_length, or dtype bfloat16/float32."
                )
            document_output: list[str] = []

            for t in sample_positions:
                controller_t = t - 1
                previous_target_t = t - 1
                previous_controller_t = t - 2
                history_ranges = make_block_ranges(t, args.block_size)
                previous_history_ranges = make_block_ranges(
                    previous_target_t, args.block_size
                )
                if sampled_block_mass is None:
                    current_self_mass = float(mean_attention[t, t].item())
                    block_mass = aggregate_block_mass(
                        mean_attention[t, :t], history_ranges
                    )
                    previous_self_mass = float(
                        mean_attention[previous_target_t, previous_target_t].item()
                    )
                    previous_block_mass = aggregate_block_mass(
                        mean_attention[previous_target_t, :previous_target_t],
                        previous_history_ranges,
                    )
                else:
                    current_attention_index = attention_position_to_index[t]
                    current_self_mass = float(
                        sampled_self_mass[current_attention_index].item()
                    )
                    block_mass = sampled_block_mass[
                        current_attention_index, : len(history_ranges)
                    ].clone()
                    current_self_block = t // args.block_size
                    if current_self_block < block_mass.numel():
                        block_mass[current_self_block] -= current_self_mass
                    block_mass.clamp_min_(0.0)

                    previous_attention_index = attention_position_to_index[
                        previous_target_t
                    ]
                    previous_self_mass = float(
                        sampled_self_mass[previous_attention_index].item()
                    )
                    previous_block_mass = sampled_block_mass[
                        previous_attention_index, : len(previous_history_ranges)
                    ].clone()
                    previous_self_block = previous_target_t // args.block_size
                    if previous_self_block < previous_block_mass.numel():
                        previous_block_mass[previous_self_block] -= previous_self_mass
                    previous_block_mass.clamp_min_(0.0)

                current_history_block = len(history_ranges) - 1
                sink, local, middle = fixed_and_middle_blocks(
                    num_blocks=block_mass.numel(),
                    sink_blocks=config.sink_blocks,
                    local_window_blocks=config.local_window_blocks,
                    current_block=current_history_block,
                )
                fixed = sink | local
                previous_history_block = len(previous_history_ranges) - 1
                previous_sink, previous_local, previous_middle = fixed_and_middle_blocks(
                    num_blocks=previous_block_mass.numel(),
                    sink_blocks=config.sink_blocks,
                    local_window_blocks=config.local_window_blocks,
                    current_block=previous_history_block,
                )
                previous_fixed = previous_sink | previous_local
                if args.retrieval_space == "post_rope_qk":
                    primary_target = qk_targets[t][args.primary_qk_selector]
                    budget_label = int(primary_target["budget_label"])
                    required_k = int(primary_target["required_k"])
                    budget_bucket_k = int(primary_target["budget_bucket_k"])
                    allocated_k = int(primary_target["allocated_k"])
                    budget_saturated = bool(primary_target["budget_saturated"])
                    previous_budget_label = int(
                        primary_target["previous_budget_label"]
                    )
                    previous_required_k = int(primary_target["previous_required_k"])
                    previous_budget_bucket_k = int(
                        primary_target["previous_budget_bucket_k"]
                    )
                    previous_allocated_k = int(primary_target["previous_allocated_k"])
                    reuse_label = int(primary_target["reuse_label"])
                    # The budget controller remains previous-hidden causal.  Q/K
                    # states are used only to construct its selector-aware target.
                    cheap_scores = cosine_block_scores(
                        hidden, history_ranges, controller_t
                    )
                else:
                    score_fn = (
                        quest_block_scores
                        if args.score_selector == "quest"
                        else arkvale_block_scores
                        if args.score_selector == "arkvale"
                        else cosine_block_scores
                    )
                    if args.score_selector == "attention_topk":
                        cheap_scores = block_mass
                    else:
                        cheap_scores = score_fn(hidden, history_ranges, controller_t)
                    budget_label, required_k = selector_oracle_budget_label(
                        block_mass=block_mass,
                        block_scores=cheap_scores,
                        fixed_blocks=fixed,
                        middle_blocks=middle,
                        coverage_threshold=config.coverage_threshold,
                        budget_buckets=config.budget_buckets,
                        always_retained_mass=current_self_mass,
                    )
                    budget_bucket_k = int(config.budget_buckets[budget_label])
                    allocated_k = min(budget_bucket_k, len(middle))
                    budget_saturated = required_k > allocated_k
                    if args.score_selector == "attention_topk":
                        previous_scores = previous_block_mass
                    else:
                        previous_scores = score_fn(
                            hidden, previous_history_ranges, previous_controller_t
                        )
                    previous_budget_label, previous_required_k = (
                        selector_oracle_budget_label(
                            block_mass=previous_block_mass,
                            block_scores=previous_scores,
                            fixed_blocks=previous_fixed,
                            middle_blocks=previous_middle,
                            coverage_threshold=config.coverage_threshold,
                            budget_buckets=config.budget_buckets,
                            always_retained_mass=previous_self_mass,
                        )
                    )
                    previous_budget_bucket_k = int(
                        config.budget_buckets[previous_budget_label]
                    )
                    previous_allocated_k = min(
                        previous_budget_bucket_k, len(previous_middle)
                    )
                    previous_middle_blocks_for_reuse = top_blocks(
                        previous_scores,
                        previous_allocated_k,
                        candidates=previous_middle,
                    )
                    reuse_label = reuse_label_from_previous_middle(
                        block_mass=block_mass,
                        fixed_blocks=fixed,
                        previous_middle_blocks=previous_middle_blocks_for_reuse,
                        reuse_coverage_threshold=config.reuse_coverage_threshold,
                        always_retained_mass=current_self_mass,
                    )

                score_features = score_summary_features(cheap_scores)
                query_drift = float(
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        hidden[controller_t].unsqueeze(0),
                        hidden[previous_controller_t].unsqueeze(0),
                        dim=-1,
                    ).item()
                )
                query_drift = min(2.0, max(0.0, query_drift))

                row = {
                    "doc_id": doc_id,
                    "source": source,
                    "source_doc_id": source_doc_id,
                    "token_index": t,
                    "attention_target_token_index": t,
                    "controller_token_index": controller_t,
                    "score_query_token_index": controller_t,
                    "selector_query_token_index": (
                        t if args.retrieval_space == "post_rope_qk" else controller_t
                    ),
                    "policy_feature_max_token_index": controller_t,
                    "oracle_policy_input_mode": "previous_hidden_only",
                    "oracle_causal_features": True,
                    "oracle_block_size": args.block_size,
                    "oracle_sink_tokens": args.sink_tokens,
                    "oracle_local_window_blocks": args.local_window_blocks,
                    "oracle_budget_buckets": list(config.budget_buckets),
                    "oracle_coverage_threshold": config.coverage_threshold,
                    "oracle_reuse_coverage_threshold": (
                        config.reuse_coverage_threshold
                    ),
                    "history_block_boundary": (
                        controller_t % args.block_size == 0
                    ),
                    "oracle_attention_mode": args.attention_mode,
                    "oracle_layer_indices": args.oracle_layer_indices,
                    "oracle_attn_implementation": model_attention_backend,
                    "oracle_sample_stride": args.sample_stride,
                    "oracle_anchor_stride": args.sample_stride,
                    "oracle_anchor_offset": document_anchor_offset,
                    "oracle_anchor_offset_mode": args.anchor_offset_mode,
                    "oracle_transition_version": "causal_next_token_v1",
                    "oracle_transition_gap": 1,
                    "oracle_retrieval_space": args.retrieval_space,
                    "oracle_selector_granularity": (
                        "per_layer_per_kv_head"
                        if args.retrieval_space == "post_rope_qk"
                        else "global_final_hidden"
                    ),
                    "oracle_coverage_aggregation": (
                        "mean_layer_query_head"
                        if args.retrieval_space == "post_rope_qk"
                        else "mean_layer_query_head_global_blocks"
                    ),
                    "reuse_previous_selector": (
                        f"causal_{args.primary_qk_selector}_topk"
                        if args.retrieval_space == "post_rope_qk"
                        else f"causal_{args.score_selector}_topk"
                    ),
                    "oracle_score_selector": (
                        args.primary_qk_selector
                        if args.retrieval_space == "post_rope_qk"
                        else args.score_selector
                    ),
                    "oracle_budget_label_selector": (
                        f"post_rope_qk_{args.primary_qk_selector}_topk_attention_coverage"
                        if args.retrieval_space == "post_rope_qk"
                        else "quest_topk_attention_coverage"
                        if args.score_selector == "quest"
                        else "arkvale_page_digest_topk_attention_coverage"
                        if args.score_selector == "arkvale"
                        else (
                            "attention_mass_topk"
                            if args.score_selector == "attention_topk"
                            else "mean_centroid_cosine_topk_attention_coverage"
                        )
                    ),
                    "hidden_state": hidden[controller_t].tolist(),
                    "query_drift": query_drift,
                    "score_features": score_features.tolist(),
                    "prev_feedback": [0.0] * 8,
                    "budget_label": budget_label,
                    "required_k": required_k,
                    "budget_bucket_k": budget_bucket_k,
                    "allocated_k": allocated_k,
                    "budget_saturated": budget_saturated,
                    "reuse_label": reuse_label,
                    "current_token_self_attention_mass": current_self_mass,
                    "past_attention_mass": float(block_mass.sum().item()),
                    "fixed_sink_blocks": len(sink),
                    "fixed_local_blocks": len(local - sink),
                    "eligible_middle_blocks": len(middle),
                    "previous_token_index": previous_target_t,
                    "previous_attention_target_token_index": previous_target_t,
                    "previous_controller_token_index": previous_controller_t,
                    "previous_budget_label": previous_budget_label,
                    "previous_required_k": previous_required_k,
                    "previous_budget_bucket_k": previous_budget_bucket_k,
                    "previous_allocated_k": previous_allocated_k,
                    "previous_budget_saturated": (
                        previous_required_k > previous_allocated_k
                    ),
                    "previous_token_self_attention_mass": previous_self_mass,
                }
                if args.retrieval_space == "post_rope_qk":
                    row.update(
                        {
                            "qk_score_selectors": list(args.qk_score_selectors),
                            "qk_cobs_rank": args.cobs_rank,
                            "qk_phsa_lambda": args.phsa_lambda,
                            "qk_arkvale_sphere_radius": "max_l2_from_key_mean",
                            "qk_mean_pool_similarity": "scaled_dot_product",
                            "budget_targets": qk_targets[t],
                        }
                    )
                document_output.append(json.dumps(row, allow_nan=False) + "\n")
            out.write("".join(document_output))
            out.flush()
            written_rows += len(document_output)
            if args.log_every_doc:
                print({"event": "doc_done", "doc_id": doc_id, "seq_len": seq_len}, flush=True)
            del (
                hidden,
                attentions,
                oracle_attentions,
                mean_attention,
                sampled_block_mass,
                sampled_self_mass,
                collector,
                qk_targets,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if resumed_document_cursor != len(existing_document_order):
        raise ValueError(
            "Partial Oracle contains documents beyond --max-docs or the packed input."
        )
    print(
        {
            "event": "oracle_build_complete",
            "written_rows": written_rows,
            "resume_skipped_documents": skipped_documents,
            "resume_skipped_rows": skipped_rows,
            "output_jsonl": str(output_path),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
