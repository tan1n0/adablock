from __future__ import annotations

import types
import copy
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from utils.block_oracle import fixed_and_middle_blocks, make_block_ranges
from utils.qk_block_oracle import (
    QK_SCORE_SELECTORS,
    QKBlockDigests,
    build_qk_block_digests,
    score_qk_block_digests,
)
from utils.ranking_data import check_metadata, model_signature


@dataclass
class LayerHeadStepStats:
    avg_selected_blocks: float
    avg_selected_tokens: float
    union_blocks: int
    union_tokens: int
    reused_heads: int
    total_heads: int


def _blocks_to_indices(
    block_ids: Iterable[int],
    block_ranges: list[tuple[int, int]],
) -> list[int]:
    indices: list[int] = []
    for block_id in sorted(block_ids):
        start, end = block_ranges[block_id]
        indices.extend(range(start, end))
    return indices


class LayerHeadQKSparseController:
    """Accuracy-first per-layer/per-KV-head sparse attention controller.

    The controller keeps the complete post-RoPE KV history on CPU.  At each
    decoder layer it uses that layer's current post-RoPE query to rank blocks
    independently for every KV head, using either cosine centroids (legacy mode)
    or the training-aligned Q/K digest selector, then gathers the chosen K/V.

    This implementation is intentionally a quality prototype.  It scores GPU
    centroid summaries but copies selected K/V from CPU to the layer device,
    so its latency is not representative of a fused sparse-attention kernel.
    """

    def __init__(
        self,
        model,
        full_past_key_values,
        block_size: int,
        sink_blocks: int,
        local_window_blocks: int,
        qk_score_selector: str | None = None,
        cobs_rank: int = 4,
        qk_layer_indices: Iterable[int] | None = None,
        ranker=None,
        ranker_metadata: dict | None = None,
        ranker_untrained_layers: str = "error",
        profile: bool = False,
    ) -> None:
        transformer = getattr(model, "model", model)
        self.layers = list(transformer.layers)
        self.block_size = int(block_size)
        self.sink_blocks = int(sink_blocks)
        self.local_window_blocks = int(local_window_blocks)
        self.profile = profile
        self.timings: dict[str, float] = {}
        self._profile_devices = {layer.self_attn.q_proj.weight.device for layer in self.layers}
        build_start = self._timer()
        self.ranker = ranker
        self.ranker_layers = set()
        self._rankers_by_device = {}
        if ranker is not None:
            if ranker_metadata is None or local_window_blocks < 1:
                raise ValueError("Ranker metadata and local_window_blocks >= 1 are required.")
            if ranker_untrained_layers not in {"error", "cobs"}:
                raise ValueError("ranker_untrained_layers must be error or cobs.")
            observed = dict(ranker_metadata, model_signature=model_signature(model.config),
                            block_size=block_size, sink_blocks=sink_blocks,
                            local_window_blocks=local_window_blocks, cobs_rank=cobs_rank,
                            query_scaling="attention_scale_applied_once", rope="post_rope",
                            gqa="mean_of_per_query_softmax_block_mass")
            check_metadata(ranker_metadata, observed)
            self.ranker_layers = set(ranker_metadata["layers"])
            if not self.ranker_layers <= set(range(len(self.layers))):
                raise ValueError("Ranker contains invalid trained layers.")
            if ranker_untrained_layers == "error" and self.ranker_layers != set(range(len(self.layers))):
                raise ValueError("Checkpoint does not cover every layer; explicitly select COBS fallback.")
            qk_score_selector, qk_layer_indices = "cobs", None
            ranker.eval()
        if qk_score_selector is not None and qk_score_selector not in QK_SCORE_SELECTORS:
            raise ValueError(
                f"Unknown qk_score_selector={qk_score_selector!r}; "
                f"choose from {QK_SCORE_SELECTORS}."
            )
        self.qk_score_selector = qk_score_selector
        self.cobs_rank = int(cobs_rank)
        self.qk_layer_indices = (
            set(range(len(self.layers)))
            if qk_layer_indices is None
            else {int(index) for index in qk_layer_indices}
        )
        self.full_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.block_key_sums: list[torch.Tensor] = []
        self.block_counts: list[torch.Tensor] = []
        self.qk_digests: list[QKBlockDigests | None] = []
        for layer_idx, (key, value) in enumerate(full_past_key_values):
            # Keep compact centroid state on the owning layer device for fast
            # scoring, while the much larger complete KV history lives on CPU.
            sums, counts = self._build_block_sums(key)
            self.block_key_sums.append(sums)
            self.block_counts.append(counts)
            digest = (
                build_qk_block_digests(
                    key[0],
                    block_size=self.block_size,
                    cobs_rank=self.cobs_rank,
                )
                if self.qk_score_selector is not None and layer_idx in self.qk_layer_indices
                else None
            )
            if digest is not None:
                digest_device = self.layers[len(self.qk_digests)].self_attn.q_proj.weight.device
                digest = QKBlockDigests(
                    **{
                        field: getattr(digest, field).to(digest_device)
                        for field in digest.__dataclass_fields__
                    }
                )
            self.qk_digests.append(digest)
            self.full_past.append(
                (key.detach().cpu(), value.detach().cpu())
            )

        self._record_time("summary_build_seconds", build_start)

        self._original_forwards: list[object] = []
        self._active = False
        self._history_len = 0
        self._max_blocks = 0
        self._requested_middle_k: int | None = None
        self._fill_to_budget = False
        self._layers_seen: set[int] = set()
        self._step_block_counts: list[int] = []
        self._step_token_counts: list[int] = []
        self._step_union_blocks: set[int] = set()
        self._step_union_tokens: set[int] = set()
        self._reuse_previous = False
        self._previous_selected_by_head: dict[tuple[int, int], set[int]] = {}
        self._step_reused_heads = 0

    def _timer(self, device=None):
        if not self.profile:
            return 0.0
        for dev in ([device] if device is not None else self._profile_devices):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        return time.perf_counter()

    def _record_time(self, name, start, device=None):
        if self.profile:
            elapsed = self._timer(device) - start
            self.timings[name] = self.timings.get(name, 0.0) + elapsed

    @torch.no_grad()
    def _learned_block_scores(self, layer_idx, grouped_query, digests, middle_tensor):
        device = grouped_query.device
        key = str(device)
        if key not in self._rankers_by_device:
            self._rankers_by_device[key] = copy.deepcopy(self.ranker).to(device).eval()
        model = self._rankers_by_device[key]
        result = grouped_query.new_full(digests.mean.shape[:2], -float("inf"))
        ids = middle_tensor
        if ids.numel() and not bool(((ids + 1) * self.block_size <= self._history_len).all()):
            raise ValueError("Learned retrieval requires complete historical candidate blocks.")
        for head in range(grouped_query.shape[0]):
            sample = {"query": grouped_query[head], "mean": digests.mean[head, ids],
                      "basis": digests.covariance_basis[head, ids],
                      "eigenvalues": digests.covariance_eigenvalues[head, ids],
                      "counts": digests.counts[ids], "block_ids": ids,
                      "block_size": self.block_size, "token_position": self._history_len,
                      "layer_id": layer_idx, "kv_head_id": head}
            result[head, ids] = model(sample)
        return result

    def _build_block_sums(self, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # key: [1, num_kv_heads, sequence, head_dim]
        sequence = int(key.shape[2])
        num_blocks = (sequence + self.block_size - 1) // self.block_size
        padded_length = num_blocks * self.block_size
        padded = F.pad(key[0], (0, 0, 0, padded_length - sequence))
        sums = padded.view(
            key.shape[1], num_blocks, self.block_size, key.shape[-1]
        ).float().sum(dim=2)
        counts = torch.tensor(
            [
                min(self.block_size, sequence - block_id * self.block_size)
                for block_id in range(num_blocks)
            ],
            dtype=torch.float32,
            device=key.device,
        )
        return sums, counts

    def install(self) -> None:
        if self._original_forwards:
            raise RuntimeError("LayerHeadQKSparseController is already installed.")
        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            self._original_forwards.append(attention.forward)

            def sparse_forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_value=None,
                cache_position=None,
                _layer_idx=layer_idx,
                **kwargs,
            ):
                return self._attention_forward(
                    module=module,
                    layer_idx=_layer_idx,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                )

            attention.forward = types.MethodType(sparse_forward, attention)

    def uninstall(self) -> None:
        if not self._original_forwards:
            return
        for layer, original in zip(self.layers, self._original_forwards):
            layer.self_attn.forward = original
        self._original_forwards.clear()

    def begin_step(
        self,
        history_len: int,
        max_blocks: int,
        requested_middle_k: int | None,
        fill_to_budget: bool,
        reuse_previous: bool = False,
    ) -> None:
        if self._active:
            raise RuntimeError("Previous layer/head sparse step was not finished.")
        cache_lengths = {int(key.shape[2]) for key, _value in self.full_past}
        if cache_lengths != {history_len}:
            raise RuntimeError(
                f"Full KV cache lengths {sorted(cache_lengths)} do not match history_len={history_len}."
            )
        self._active = True
        self._history_len = int(history_len)
        self._max_blocks = int(max_blocks)
        self._requested_middle_k = requested_middle_k
        self._fill_to_budget = bool(fill_to_budget)
        self._reuse_previous = bool(reuse_previous)
        self._step_reused_heads = 0
        self._layers_seen.clear()
        self._step_block_counts.clear()
        self._step_token_counts.clear()
        self._step_union_blocks.clear()
        self._step_union_tokens.clear()

    def finish_step(self) -> LayerHeadStepStats:
        if not self._active:
            raise RuntimeError("No active layer/head sparse step.")
        expected = set(range(len(self.layers)))
        if self._layers_seen != expected:
            raise RuntimeError(
                "Sparse attention did not run exactly once for every layer: "
                f"missing={sorted(expected - self._layers_seen)}, "
                f"extra={sorted(self._layers_seen - expected)}"
            )
        self._active = False
        count = max(len(self._step_block_counts), 1)
        return LayerHeadStepStats(
            avg_selected_blocks=sum(self._step_block_counts) / count,
            avg_selected_tokens=sum(self._step_token_counts) / count,
            union_blocks=len(self._step_union_blocks),
            union_tokens=len(self._step_union_tokens),
            reused_heads=self._step_reused_heads,
            total_heads=len(self._step_block_counts),
        )

    def _select_token_indices(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        # query_states: [1, num_query_heads, 1, head_dim]
        block_ranges = make_block_ranges(self._history_len, self.block_size)
        num_blocks = len(block_ranges)
        current_block = num_blocks - 1
        sink, local, middle = fixed_and_middle_blocks(
            num_blocks=num_blocks,
            sink_blocks=self.sink_blocks,
            local_window_blocks=self.local_window_blocks,
            current_block=current_block,
        )
        fixed = sink | local
        dynamic_capacity = max(0, min(self._max_blocks, num_blocks) - len(fixed))
        if self._fill_to_budget or self._requested_middle_k is None:
            middle_k = dynamic_capacity
        else:
            middle_k = min(max(self._requested_middle_k, 0), dynamic_capacity)

        query = query_states[0, :, 0, :].detach().float()
        use_qk = self.qk_score_selector is not None and layer_idx in self.qk_layer_indices
        if use_qk:
            digests = self.qk_digests[layer_idx]
            if digests is None:
                raise RuntimeError("Q/K digests are missing for Q/K selection.")
            num_kv_heads = int(digests.mean.shape[0])
        else:
            centroids = self.block_key_sums[layer_idx] / self.block_counts[layer_idx].view(
                1, -1, 1
            )
            num_kv_heads = int(centroids.shape[0])
        num_query_heads = int(query.shape[0])
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads={num_query_heads} is not divisible by num_kv_heads={num_kv_heads}."
            )
        group_size = num_query_heads // num_kv_heads
        grouped_query = query.view(num_kv_heads, group_size, query.shape[-1])
        score_start = self._timer(query.device)
        if self.ranker is not None and layer_idx in self.ranker_layers:
            middle_ids = torch.tensor(sorted(middle), dtype=torch.long, device=query.device)
            if middle_k > 0 and middle_ids.numel():
                qk_scores = self._learned_block_scores(
                    layer_idx, grouped_query * float(self.layers[layer_idx].self_attn.scaling),
                    digests, middle_ids,
                )
            else:
                qk_scores = query.new_zeros(num_kv_heads, len(block_ranges))
        elif self.qk_score_selector is not None and layer_idx in self.qk_layer_indices:
            grouped_query = grouped_query.unsqueeze(2)
            qk_scores = score_qk_block_digests(
                grouped_query * float(self.layers[layer_idx].self_attn.scaling),
                digests,
                selectors=[self.qk_score_selector],
            )[self.qk_score_selector][0]
        else:
            query_norm = F.normalize(grouped_query, dim=-1)
            centroid_norm = F.normalize(centroids, dim=-1)
            per_query_scores = torch.einsum("hgd,hbd->hgb", query_norm, centroid_norm)
            # Normalize each query head over blocks before combining the GQA group.
            qk_scores = torch.softmax(per_query_scores, dim=-1).mean(dim=1)

        self._record_time("scoring_seconds", score_start, query.device)
        selection_start = self._timer(query.device)

        middle_tensor = torch.tensor(
            sorted(middle), dtype=torch.long, device=qk_scores.device
        )
        selected_by_head: list[list[int]] = []
        for kv_head in range(num_kv_heads):
            selected = set(fixed)
            previous = self._previous_selected_by_head.get((layer_idx, kv_head), set())
            reusable = previous & middle
            reused = self._reuse_previous and bool(reusable)
            if reused:
                selected.update(reusable)
                self._step_reused_heads += 1
            elif middle_k > 0 and middle_tensor.numel() > 0:
                take = min(middle_k, int(middle_tensor.numel()))
                local_top = torch.topk(
                    qk_scores[kv_head, middle_tensor], k=take
                ).indices
                selected.update(middle_tensor[local_top].tolist())
            self._previous_selected_by_head[(layer_idx, kv_head)] = set(selected - fixed)
            token_indices = _blocks_to_indices(selected, block_ranges)
            selected_by_head.append(token_indices)
            self._step_block_counts.append(len(selected))
            self._step_token_counts.append(len(token_indices))
            self._step_union_blocks.update(selected)
            self._step_union_tokens.update(token_indices)

        lengths = {len(indices) for indices in selected_by_head}
        if len(lengths) != 1:
            raise RuntimeError(
                "Per-KV-head selections produced ragged token counts; "
                f"lengths={sorted(lengths)}."
            )
        indices = torch.tensor(selected_by_head, dtype=torch.long)
        self._record_time("selection_seconds", selection_start, query.device)
        return indices

    def _append_current_kv(
        self,
        layer_idx: int,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> None:
        key_cpu = current_key.detach().cpu()
        value_cpu = current_value.detach().cpu()
        full_key, full_value = self.full_past[layer_idx]
        if int(full_key.shape[2]) != self._history_len:
            raise RuntimeError(
                f"Layer {layer_idx} cache was appended more than once in the current step."
            )
        self.full_past[layer_idx] = (
            torch.cat([full_key, key_cpu], dim=2),
            torch.cat([full_value, value_cpu], dim=2),
        )

        block_id = self._history_len // self.block_size
        current = current_key[0, :, 0, :].detach().float()
        sums = self.block_key_sums[layer_idx]
        counts = self.block_counts[layer_idx]
        if block_id == int(sums.shape[1]):
            self.block_key_sums[layer_idx] = torch.cat(
                [sums, current.unsqueeze(1)], dim=1
            )
            self.block_counts[layer_idx] = torch.cat(
                [
                    counts,
                    torch.ones(1, dtype=counts.dtype, device=counts.device),
                ],
                dim=0,
            )
        else:
            sums[:, block_id, :].add_(current)
            counts[block_id].add_(1.0)

        if self.qk_score_selector is not None and layer_idx in self.qk_layer_indices:
            full_key_now = self.full_past[layer_idx][0]
            start = block_id * self.block_size
            end = int(full_key_now.shape[2])
            block_digest = build_qk_block_digests(
                full_key_now[0, :, start:end, :].to(current_key.device),
                block_size=self.block_size,
                cobs_rank=self.cobs_rank,
            )
            digests = self.qk_digests[layer_idx]
            assert digests is not None
            block_index = int(block_id)
            if block_index >= int(digests.mean.shape[1]):
                self.qk_digests[layer_idx] = QKBlockDigests(
                    counts=torch.cat([digests.counts, block_digest.counts], dim=0),
                    mean=torch.cat([digests.mean, block_digest.mean], dim=1),
                    minimum=torch.cat([digests.minimum, block_digest.minimum], dim=1),
                    maximum=torch.cat([digests.maximum, block_digest.maximum], dim=1),
                    sphere_radius=torch.cat([digests.sphere_radius, block_digest.sphere_radius], dim=1),
                    diagonal_variance=torch.cat([digests.diagonal_variance, block_digest.diagonal_variance], dim=1),
                    covariance_basis=torch.cat([digests.covariance_basis, block_digest.covariance_basis], dim=1),
                    covariance_eigenvalues=torch.cat([digests.covariance_eigenvalues, block_digest.covariance_eigenvalues], dim=1),
                    punctuation_mean=torch.cat([digests.punctuation_mean, block_digest.punctuation_mean], dim=1),
                )
            else:
                def replace_block(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
                    updated = old.clone()
                    updated[:, block_index] = new[:, 0]
                    return updated

                self.qk_digests[layer_idx] = QKBlockDigests(
                    counts=digests.counts.clone().index_copy_(0, torch.tensor([block_index], device=digests.counts.device), block_digest.counts),
                    mean=replace_block(digests.mean, block_digest.mean),
                    minimum=replace_block(digests.minimum, block_digest.minimum),
                    maximum=replace_block(digests.maximum, block_digest.maximum),
                    sphere_radius=replace_block(digests.sphere_radius, block_digest.sphere_radius),
                    diagonal_variance=replace_block(digests.diagonal_variance, block_digest.diagonal_variance),
                    covariance_basis=replace_block(digests.covariance_basis, block_digest.covariance_basis),
                    covariance_eigenvalues=replace_block(digests.covariance_eigenvalues, block_digest.covariance_eigenvalues),
                    punctuation_mean=replace_block(digests.punctuation_mean, block_digest.punctuation_mean),
                )

    def _attention_forward(
        self,
        module,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        if not self._active:
            raise RuntimeError("Layer/head sparse attention called outside an active step.")
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
            raise RuntimeError(
                "Layer/head sparse attention currently supports batch=1, query_length=1 only."
            )
        if layer_idx in self._layers_seen:
            raise RuntimeError(f"Layer {layer_idx} ran more than once in one sparse step.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        token_indices = self._select_token_indices(layer_idx, query_states)
        gather_start = self._timer(query_states.device)
        full_key, full_value = self.full_past[layer_idx]
        num_kv_heads, selected_tokens = token_indices.shape
        head_dim = int(full_key.shape[-1])
        gather_index = token_indices.unsqueeze(-1).expand(
            num_kv_heads, selected_tokens, head_dim
        )
        selected_key = torch.gather(full_key[0], dim=1, index=gather_index).to(
            device=query_states.device, dtype=query_states.dtype
        )
        selected_value = torch.gather(full_value[0], dim=1, index=gather_index).to(
            device=query_states.device, dtype=query_states.dtype
        )
        selected_key = torch.cat([selected_key.unsqueeze(0), key_states], dim=2)
        selected_value = torch.cat([selected_value.unsqueeze(0), value_states], dim=2)

        group_size = int(query_states.shape[1]) // num_kv_heads
        attention_key = selected_key.repeat_interleave(group_size, dim=1)
        attention_value = selected_value.repeat_interleave(group_size, dim=1)
        self._record_time("kv_gather_seconds", gather_start, query_states.device)
        attention_start = self._timer(query_states.device)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            attention_key,
            attention_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=module.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = module.o_proj(attn_output)

        self._record_time("attention_seconds", attention_start, query_states.device)
        update_start = self._timer(query_states.device)

        self._append_current_kv(layer_idx, key_states, value_states)
        self._record_time("cache_update_seconds", update_start, query_states.device)
        self._layers_seen.add(layer_idx)
        return attn_output, None
