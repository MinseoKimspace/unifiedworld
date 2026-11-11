from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .data import FeatureNormalizer, make_features, predict_free_motion


@dataclass
class State:
    x: Tensor
    v: Tensor
    previous_v: Tensor
    older_v: Tensor
    observed: bool = True

    @classmethod
    def from_batch(cls, batch, start=2):
        if start < 2 or start >= batch["x"].shape[1]:
            raise ValueError("A starting state needs three reference frames")
        return cls(
            batch["x"][:, start],
            batch["v"][:, start],
            batch["v"][:, start - 1],
            batch["v"][:, start - 2],
        )

    def detach(self):
        return replace(
            self,
            **{key: getattr(self, key).detach() for key in ("x", "v", "previous_v", "older_v")},
        )


def apply_correction(x, v, correction):
    return x + correction[..., :3], v + correction[..., 3:]


def integrate(x, v, correction, acceleration, dt):
    return apply_correction(*predict_free_motion(x, v, acceleration, dt), correction)


class ParticleSystem(nn.Module):
    def __init__(self, model: nn.Module, normalizer: FeatureNormalizer):
        super().__init__()
        self.model = model
        self.normalizer = normalizer

    def step(self, batch, state: State, t: int):
        acceleration = batch["acceleration"][:, t]
        free_x, free_v = predict_free_motion(state.x, state.v, acceleration, batch["dt"])
        history = torch.cat((state.v - state.previous_v, state.previous_v - state.older_v), -1)
        features = make_features(
            batch,
            free_x,
            free_v,
            history,
            acceleration,
            state.observed,
        )
        normalized = self.normalizer(features)
        if getattr(self.model, "requires_raw_features", False):
            main, auxiliary = self.model(normalized, features)
        else:
            main, auxiliary = self.model(normalized)
        x, v = apply_correction(free_x, free_v, main)
        aux = (
            None
            if auxiliary is None
            else apply_correction(free_x, free_v, auxiliary)
        )
        return State(x, v, state.v, state.previous_v, False), aux


def free_rollout(system, batch, steps: int, start=2):
    if steps < 1 or start + steps >= batch["x"].shape[1]:
        raise ValueError("Reference trajectory is shorter than the requested rollout")
    state = State.from_batch(batch, start)
    positions, velocities = [], []
    for t in range(start, start + steps):
        state, _ = system.step(batch, state, t)
        positions.append(state.x)
        velocities.append(state.v)
    return torch.stack(positions, 1), torch.stack(velocities, 1)


def balanced_mse(predicted, target, material):
    error = (predicted - target).square().mean(-1)
    while material.ndim < error.ndim:
        material = material.unsqueeze(1)
    values = []
    for material_id in range(3):
        mask = (material == material_id).expand_as(error)
        count = mask.sum(-1)
        mean = (error * mask).sum(-1) / count.clamp_min(1)
        values.append((mean, count > 0))
    means, valid = (torch.stack(items, -1) for items in zip(*values))
    return ((means * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()


def state_loss(x, v, target_x, target_v, material):
    return (
        balanced_mse(x, target_x, material) / 0.028**2
        + balanced_mse(v, target_v, material) / 0.5**2
    )
