import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

from unifiedworld.data import canonicalize_scene
from unifiedworld.dynamics import ParticleSystem, State, balanced_mse, free_rollout, integrate
from unifiedworld.schema import BOUNDARY_SLICE, OBSERVATION_INDEX, schema_metadata


def batch():
    params = torch.tensor([[0, 0, 1e15, 0, 1], [100, 0, 1e10, 300, 1], [0, 0.5, 1e15, 0, 0]])
    masks = torch.ones_like(params)
    dt = 0.02
    acceleration = torch.tensor([0.2, -0.4, 0.1])
    time = torch.arange(8)[:, None, None] * dt
    v = torch.ones(1, 3, 3) * 0.3 + time * acceleration
    x = torch.ones(1, 3, 3) * 0.2 + 0.3 * time + time.square() * acceleration / 2
    return default_collate(
        [
            canonicalize_scene(
                {
                    "x": x,
                    "v": v,
                    "mass": torch.ones(3, 1),
                    "params": params,
                    "applicable": masks,
                    "known": masks,
                    "material": torch.arange(3),
                    "acceleration": acceleration,
                    "dt": dt,
                    "spacing": 0.028,
                    "schema": schema_metadata(),
                    "boundary": "box",
                    "radius": 0.014,
                    "density": 1000.0,
                    "bounds": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                }
            )
        ]
    )


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, features):
        self.inputs.append(features.detach().clone())
        shape = (*features.shape[:2], 6)
        return features.new_zeros(shape), features.new_full(
            shape, 1000.0
        ) if self.training else None


def test_ballistic_rollout_and_aux_do_not_change_main_state():
    reference = batch()
    model = RecordingModel()
    system = ParticleSystem(model, nn.Identity())
    x, v = free_rollout(system, reference, 5)
    torch.testing.assert_close(x, reference["x"][:, 3:])
    torch.testing.assert_close(v, reference["v"][:, 3:])
    assert len(model.inputs) == 5
    torch.testing.assert_close(model.inputs[0][..., :3], reference["x"][:, 3])
    torch.testing.assert_close(model.inputs[0][..., 3:6], reference["v"][:, 3])
    torch.testing.assert_close(
        model.inputs[0][..., BOUNDARY_SLICE],
        torch.cat((reference["x"][:, 3], 1 - reference["x"][:, 3]), -1),
    )
    assert model.inputs[0][..., OBSERVATION_INDEX].eq(1).all()
    assert all(features[..., OBSERVATION_INDEX].eq(0).all() for features in model.inputs[1:])
    torch.testing.assert_close(model.inputs[1][..., 6:9], v[:, 0] - reference["v"][:, 2])
    torch.testing.assert_close(
        model.inputs[1][..., 9:12], reference["v"][:, 2] - reference["v"][:, 1]
    )


def test_future_reference_state_cannot_leak_into_rollout():
    reference = batch()
    system = ParticleSystem(RecordingModel(), nn.Identity()).eval()
    expected = free_rollout(system, reference, 5)
    reference["x"][:, 3:] = torch.randn_like(reference["x"][:, 3:]) * 100
    reference["v"][:, 3:] = torch.randn_like(reference["v"][:, 3:]) * 100
    actual = free_rollout(system, reference, 5)
    for first, second in zip(expected, actual):
        torch.testing.assert_close(first, second)


def test_direct_position_and_velocity_residuals_follow_free_motion():
    x = torch.zeros(2, 1, 3)
    v = torch.ones_like(x)
    correction = torch.ones(2, 1, 6)
    dt = torch.tensor([0.1, 0.2])
    next_x, next_v = integrate(x, v, correction, torch.ones_like(x), dt)
    torch.testing.assert_close(next_v[:, 0, 0], torch.tensor([2.1, 2.2]))
    torch.testing.assert_close(next_x[:, 0, 0], torch.tensor([1.105, 1.22]))
    correction[..., :3] = 0
    next_x, _ = integrate(x, v, correction, torch.ones_like(x), dt)
    torch.testing.assert_close(next_x[:, 0, 0], torch.tensor([0.105, 0.22]))


def test_material_balancing_is_independent_of_particle_counts():
    target = torch.zeros(1, 3, 3)
    prediction = torch.tensor([1.0, 1.0, 3.0])[None, :, None].expand_as(target)
    assert balanced_mse(prediction, target, torch.tensor([[0, 0, 1]])).item() == 5


def test_detach_preserves_all_history_and_horizon_is_explicit():
    reference = batch()
    state = State.from_batch(reference)
    state.x.requires_grad_()
    detached = state.detach()
    for key in ("x", "v", "previous_v", "older_v"):
        torch.testing.assert_close(getattr(detached, key), getattr(state, key))
        assert not getattr(detached, key).requires_grad
    system = ParticleSystem(RecordingModel(), nn.Identity())
    with pytest.raises(ValueError):
        free_rollout(system, reference, 6)
