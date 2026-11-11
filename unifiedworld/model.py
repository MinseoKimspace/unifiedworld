from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .neighbors import radius_neighbors
from .schema import INPUT_DIM, KNOWN_SLICE, PARAM_SLICE


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    heads: int = 4
    blocks: int = 2
    ffn_dim: int = 128
    radius: float = 0.025
    max_neighbors: int = 64
    output_scale: float = 1 / 128
    checkpoint_forward: bool = True
    relation_mode: str = "concat"
    pair_rank: int | None = None
    pair_gate_norm: str = "none"
    global_grid_size: int = 4

    def __post_init__(self):
        counts = (self.width, self.heads, self.blocks, self.ffn_dim, self.max_neighbors)
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Model dimensions and neighbor limit must be positive integers")
        if self.width % self.heads or self.blocks < 2:
            raise ValueError("width must be divisible by heads and blocks must be at least two")
        if not all(isfinite(value) and value > 0 for value in (self.radius, self.output_scale)):
            raise ValueError("radius and output_scale must be finite and positive")
        if not isinstance(self.checkpoint_forward, bool):
            raise ValueError("checkpoint_forward must be boolean")
        if self.relation_mode not in {"concat", "score", "value", "score_value"}:
            raise ValueError("relation_mode must be concat, score, value, or score_value")
        if type(self.global_grid_size) is not int or self.global_grid_size < 1:
            raise ValueError("global_grid_size must be a positive integer")
        if self.pair_rank is not None and (type(self.pair_rank) is not int or self.pair_rank < 1):
            raise ValueError("pair_rank must be a positive integer or None")
        if self.pair_gate_norm not in {"none", "rms"}:
            raise ValueError("pair_gate_norm must be none or rms")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
    )


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


class LocalSparseAttention(nn.Module):
    """Compute state-dependent messages directly, without persistent edge features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.width
        rank = config.pair_rank or width
        self.heads = config.heads
        self.rank = rank
        self.gate_norm = config.pair_gate_norm
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.geometry_bias = _mlp(7, width, config.heads)
        self.value_basis = _mlp(2 * width + 7, rank, rank)
        self.value_out = nn.Linear(rank, width, bias=False)
        self.score_pair = None
        self.score_geometry = None
        self.score_out = None
        self.value_pair = None
        if config.relation_mode in {"score", "score_value"}:
            self.score_pair = _mlp(2 * width, rank, rank)
            self.score_geometry = _mlp(7, rank, rank)
            self.score_out = nn.Linear(rank, config.heads, bias=False)
        if config.relation_mode in {"value", "score_value"}:
            self.value_pair = _mlp(2 * width, rank, rank)

    def _pair_gate(self, encoder, receiver_material, sender_material):
        gate = encoder(torch.cat((receiver_material, sender_material), -1))
        if self.gate_norm == "rms":
            gate = gate * torch.rsqrt(gate.square().mean(-1, keepdim=True) + 1e-6)
        return gate

    def relation_scores(self, receiver_material, sender_material, relation):
        """Return geometry-dependent additive logits, with one channel per head."""
        scores = self.geometry_bias(relation)
        if self.score_pair is not None:
            gate = self._pair_gate(self.score_pair, receiver_material, sender_material)
            scores = scores + self.score_out(gate * self.score_geometry(relation)) / sqrt(self.rank)
        return scores

    def relation_values(self, receiver, sender, receiver_material, sender_material, relation):
        """Return each selected neighbor's message before attention aggregation."""
        basis = self.value_basis(torch.cat((receiver, sender, relation), -1))
        if self.value_pair is not None:
            basis = basis * self._pair_gate(self.value_pair, receiver_material, sender_material)
        return self.value_out(basis)

    def forward(self, hidden, material, relation, indices, valid):
        batch, particles, width = hidden.shape
        neighbors = indices.shape[-1]
        sender = _gather(hidden, indices)
        receiver = hidden.unsqueeze(-2).expand_as(sender)
        sender_material = _gather(material, indices)
        receiver_material = material.unsqueeze(-2).expand_as(sender_material)
        query = self.query(hidden).reshape(batch * particles, self.heads, 1, -1)
        key = _gather(self.key(hidden), indices)
        value = self.relation_values(
            receiver, sender, receiver_material, sender_material, relation
        )
        key, value = (
            item.reshape(batch * particles, neighbors, self.heads, -1).transpose(1, 2)
            for item in (key, value)
        )
        scores = self.relation_scores(receiver_material, sender_material, relation)
        # Isolated particles receive one dummy key whose final contribution is removed.
        has_neighbors = valid.any(-1)
        safe_mask = valid.clone()
        safe_mask[..., 0] |= ~has_neighbors
        scores = scores.masked_fill(~safe_mask.unsqueeze(-1), -torch.inf)
        scores = scores.permute(0, 1, 3, 2).reshape(batch * particles, self.heads, 1, neighbors)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=scores)
        return attended.reshape(batch, particles, width) * has_neighbors.unsqueeze(-1)


def _pool_grid(hidden, positions, mass, grid_size, radius):
    """Mass-average occupied spatial cells using only N-by-M assignment storage."""
    lower = positions.detach().amin(1, keepdim=True)
    span = (positions.detach().amax(1, keepdim=True) - lower).clamp_min(radius)
    normalized_positions = (positions - lower) / span
    cells = (normalized_positions.detach() * grid_size).long().clamp(0, grid_size - 1)
    cell_ids = (cells[..., 0] * grid_size + cells[..., 1]) * grid_size + cells[..., 2]
    assignment = F.one_hot(cell_ids, num_classes=grid_size**3).to(hidden.dtype)
    weights = assignment * mass.clamp_min(torch.finfo(hidden.dtype).tiny)
    total_mass = weights.sum(1)
    valid = total_mass > 0
    weights = weights / total_mass.clamp_min(torch.finfo(hidden.dtype).tiny).unsqueeze(1)
    weights = weights.transpose(1, 2)
    pooled = weights @ hidden
    pooled_positions = weights @ normalized_positions
    return pooled, pooled_positions, valid, normalized_positions


class GlobalAttention(nn.Module):
    """Exchange coarse spatial context and read it back at original particle resolution."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.grid_size = config.global_grid_size
        self.radius = config.radius
        self.position_encoder = _mlp(3, width, width)
        self.coarse_norm = nn.LayerNorm(width)
        self.coarse_qkv = nn.Linear(width, 3 * width)
        self.coarse_out = nn.Linear(width, width)
        self.coarse_ffn_norm = nn.LayerNorm(width)
        self.coarse_ffn = _mlp(width, config.ffn_dim, width)
        self.query = nn.Linear(width, width)
        self.context_norm = nn.LayerNorm(width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)

    def _heads(self, values):
        batch, tokens, width = values.shape
        return values.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, hidden, positions, mass):
        pooled, pooled_positions, valid, normalized_positions = _pool_grid(
            hidden, positions, mass, self.grid_size, self.radius
        )
        coarse = pooled + self.position_encoder(pooled_positions)
        query, key, value = self.coarse_qkv(self.coarse_norm(coarse)).chunk(3, -1)
        mask = valid[:, None, None, :]
        coarse_update = F.scaled_dot_product_attention(
            self._heads(query), self._heads(key), self._heads(value), attn_mask=mask
        )
        coarse_update = coarse_update.transpose(1, 2).reshape_as(coarse)
        coarse = coarse + self.coarse_out(coarse_update)
        coarse = coarse + self.coarse_ffn(self.coarse_ffn_norm(coarse))
        query = self.query(hidden + self.position_encoder(normalized_positions))
        key, value = self.key_value(self.context_norm(coarse)).chunk(2, -1)
        context = F.scaled_dot_product_attention(
            self._heads(query), self._heads(key), self._heads(value), attn_mask=mask
        )
        context = context.transpose(1, 2).reshape_as(hidden)
        return self.output(context)


class LocalGlobalBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.width
        self.node_norm = nn.LayerNorm(width)
        self.local_attention = LocalSparseAttention(config)
        self.global_attention = GlobalAttention(config)
        self.local_norm = nn.LayerNorm(width)
        self.global_norm = nn.LayerNorm(width)
        self.fusion = _mlp(4 * width + 2, width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = _mlp(width, config.ffn_dim, width)

    def forward(self, hidden, material, relation, indices, valid, scalars, positions, mass):
        normalized = self.node_norm(hidden)
        local = self.local_attention(normalized, material, relation, indices, valid)
        global_context = self.global_attention(normalized, positions, mass)
        hidden = hidden + self.fusion(torch.cat(
            (normalized, self.local_norm(local), self.global_norm(global_context), material, scalars),
            -1,
        ))
        return hidden + self.ffn(self.ffn_norm(hidden))


class ParticleTransformer(nn.Module):
    requires_raw_features = True

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        material_dim = KNOWN_SLICE.stop - PARAM_SLICE.start
        self.material_encoder = _mlp(material_dim, config.width, config.width)
        self.encoder = nn.Linear(INPUT_DIM - material_dim + config.width, config.width)
        self.blocks = nn.ModuleList(LocalGlobalBlock(config) for _ in range(config.blocks))
        self.aux_head = _mlp(config.width, config.width, 6)
        self.decoder = _mlp(config.width, config.width, 6)
        self.neighbor_counts = None

    def encode_particles(self, features):
        material = self.material_encoder(features[..., PARAM_SLICE.start:KNOWN_SLICE.stop])
        state = torch.cat(
            (features[..., :PARAM_SLICE.start], features[..., KNOWN_SLICE.stop:], material), -1
        )
        return self.encoder(state), material

    def relation_features(self, raw_features, indices):
        relative_x = (_gather(raw_features[..., :3], indices)
                      - raw_features[..., None, :3]) / self.config.radius
        relative_v = _gather(raw_features[..., 3:6], indices) - raw_features[..., None, 3:6]
        return torch.cat((relative_x, relative_v, relative_x.norm(dim=-1, keepdim=True)), -1)

    def _forward(self, features, raw_features, indices, valid, counts):
        hidden, material = self.encode_particles(features)
        relation = self.relation_features(raw_features, indices) * valid.unsqueeze(-1)
        mass = raw_features[..., 12:13]
        neighbor_mass = (_gather(mass, indices) * valid.unsqueeze(-1)).sum(-2)
        scalars = torch.cat((counts.to(hidden.dtype).log1p().unsqueeze(-1), neighbor_mass), -1)
        auxiliary = None
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, material, relation, indices, valid, scalars,
                           raw_features[..., :3], mass)
            if index == 0 and self.training:
                auxiliary = self.aux_head(hidden) * self.config.output_scale
        return self.decoder(hidden) * self.config.output_scale, auxiliary

    def forward(self, features: Tensor, raw_features: Tensor) -> tuple[Tensor, Tensor | None]:
        if (features.ndim != 3 or features.shape[-1] != INPUT_DIM
                or min(features.shape[:2]) < 1):
            raise ValueError(f"features must have shape [B,N,{INPUT_DIM}] with B and N positive")
        if (raw_features.shape != features.shape or raw_features.device != features.device
                or raw_features.dtype != features.dtype):
            raise ValueError("raw_features must match the features shape, device, and dtype")
        indices, valid, counts = radius_neighbors(
            raw_features[..., :3].detach(), self.config.radius, self.config.max_neighbors
        )
        self.neighbor_counts = counts.detach()
        args = (features, raw_features, indices.clamp_min(0), valid, counts)
        if self.training and torch.is_grad_enabled() and self.config.checkpoint_forward:
            return checkpoint(self._forward, *args, use_reentrant=False)
        return self._forward(*args)
