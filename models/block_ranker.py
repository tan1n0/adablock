"""Query/block ranker; frozen post-RoPE Q/K inputs, trainable matching only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class BlockRankerConfig:
    head_dim: int
    num_layers: int
    num_kv_heads: int
    projection_size: int = 64
    hidden_size: int = 128
    use_position: bool = True
    identity_size: int = 8
    block_chunk_size: int = 256


def cobs_terms(query, mean, basis, eigenvalues):
    """query [G,D] already includes attention scaling; outputs [G,B]."""
    query, mean = query.float(), mean.float()
    first = torch.einsum("gd,bd->gb", query, mean)
    projected = torch.einsum("gd,brd->gbr", query, basis.float())
    second = (projected.square() * eigenvalues.float().unsqueeze(0)).sum(-1)
    return first, second


def position_features(block_ids, counts, token_position: int, block_size: int):
    # Keys are strictly before token_position; block end is inclusive here.
    end = block_ids.float() * block_size + counts.float() - 1
    distance = (float(token_position) - end).clamp_min(1)
    length = distance.new_tensor(max(token_position, 1))
    return torch.stack((distance.log1p() / length.log1p(), distance / length), -1)


class BlockRanker(nn.Module):
    def __init__(self, config: BlockRankerConfig):
        super().__init__()
        self.config = config
        if min(config.head_dim, config.num_layers, config.num_kv_heads,
               config.projection_size, config.hidden_size, config.block_chunk_size) < 1:
            raise ValueError("Ranker dimensions and block_chunk_size must be positive.")
        if config.identity_size < 0:
            raise ValueError("identity_size must be nonnegative.")
        p = config.projection_size
        self.query_proj = nn.Sequential(nn.LayerNorm(config.head_dim), nn.Linear(config.head_dim, p))
        self.mean_proj = nn.Sequential(nn.LayerNorm(config.head_dim), nn.Linear(config.head_dim, p))
        self.layer_embedding = nn.Embedding(config.num_layers, config.identity_size)
        self.head_embedding = nn.Embedding(config.num_kv_heads, config.identity_size)
        # Signed log1p(first), log1p(second), log(count) use a deterministic,
        # shared transform. No fitted validation statistics or changing scale.
        size = p * 3 + 3 + 2 * config.identity_size + (2 if config.use_position else 0)
        self.interaction = nn.Sequential(
            nn.Linear(size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
        )
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, sample: dict[str, object]) -> torch.Tensor:
        """One ragged scene, all candidates batched; no B x B materialization."""
        q = sample["query"].float()
        mean = sample["mean"].float()
        if mean.shape[0] == 0:
            return self.output.weight.sum().mul(0).expand(0)
        qp = self.query_proj(q)
        layer = self.layer_embedding.weight[int(sample["layer_id"])]
        head = self.head_embedding.weight[int(sample["kv_head_id"])]
        identity = torch.cat((layer, head))
        chunks = []
        for start in range(0, mean.shape[0], self.config.block_chunk_size):
            end = min(start + self.config.block_chunk_size, mean.shape[0])
            mp = self.mean_proj(mean[start:end])
            first, second = cobs_terms(q, mean[start:end], sample["basis"][start:end],
                                       sample["eigenvalues"][start:end])
            counts = sample["counts"][start:end].float()
            qa = qp[:, None, :].expand(-1, end - start, -1)
            ma = mp[None, :, :].expand(q.shape[0], -1, -1)
            scalars = torch.stack((first.sign() * first.abs().log1p(),
                                   second.clamp_min(0).log1p(),
                                   counts.clamp_min(1).log().expand_as(first)), -1)
            features = [qa, ma, qa * ma, scalars,
                        identity.view(1, 1, -1).expand(q.shape[0], end - start, -1)]
            if self.config.use_position:
                pos = position_features(sample["block_ids"][start:end], counts,
                                        int(sample["token_position"]), int(sample["block_size"]))
                features.append(pos.unsqueeze(0).expand(q.shape[0], -1, -1))
            # Nonlinearity BEFORE pooling the GQA group; never average raw Q.
            hidden = self.interaction(torch.cat(features, -1)).mean(dim=0)
            chunks.append(self.output(hidden).squeeze(-1))
        return torch.cat(chunks)


def to_device(sample, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()}


def save_ranker(path, model, metadata, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "adablock_ranker_v1", "config": asdict(model.config),
                "state_dict": model.state_dict(), "data_metadata": metadata,
                "training": training}, path)


def load_ranker(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "adablock_ranker_v1":
        raise ValueError("Not an AdaBlock ranker checkpoint.")
    model = BlockRanker(BlockRankerConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
