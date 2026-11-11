import json
from dataclasses import asdict, replace

import lightning as L
import pytest
import torch
from test_contact_runner import tiny_scene

from unifiedworld import experiment as compare_sparse
from unifiedworld.model import ModelConfig, ParticleTransformer
from unifiedworld.schema import INPUT_DIM
from unifiedworld.training import WorldModel

MODES = ("concat", "score", "value", "score_value")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    values = dict(width=8, heads=2, ffn_dim=16,
                  global_grid_size=2, max_neighbors=8, pair_rank=5)
    return ModelConfig(**(values | kwargs))


def features(batch=2, particles=7):
    result = torch.randn(batch, particles, INPUT_DIM)
    result[..., :3] = torch.rand(batch, particles, 3) * 0.02
    result[..., 12] = 0.002
    return result


@pytest.mark.parametrize("mode", MODES)
def test_local_global_preserves_particle_order_and_output_contract(mode):
    torch.manual_seed(29)
    model = ParticleTransformer(config(relation_mode=mode)).eval()
    inputs = features()
    permutation = torch.randperm(inputs.shape[1])
    expected, auxiliary = model(inputs, inputs)
    actual, _ = model(inputs[:, permutation], inputs[:, permutation])
    assert expected.shape == (2, 7, 6) and auxiliary is None
    torch.testing.assert_close(actual, expected[:, permutation], atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_local_global_handles_isolated_nodes_without_batch_mixing(mode):
    torch.manual_seed(17)
    model = ParticleTransformer(config(relation_mode=mode)).eval()
    inputs = features(particles=1)
    expected, _ = model(inputs, inputs)
    assert torch.isfinite(expected).all()
    assert model.neighbor_counts.count_nonzero() == 0
    changed = inputs.clone()
    changed[1] += 100
    actual, _ = model(changed, changed)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_local_global_main_and_auxiliary_keep_one_over_128_scale(mode):
    torch.manual_seed(17)
    settings = config(relation_mode=mode, checkpoint_forward=False)
    scaled = ParticleTransformer(settings)
    unscaled = ParticleTransformer(replace(settings, output_scale=1))
    unscaled.load_state_dict(scaled.state_dict())
    inputs = features()
    for actual, expected in zip(scaled(inputs, inputs), unscaled(inputs, inputs)):
        assert actual.shape == (2, 7, 6)
        torch.testing.assert_close(actual * 128, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_local_global_checkpoint_recomputation_preserves_gradients(mode):
    torch.manual_seed(43)
    eager = ParticleTransformer(config(relation_mode=mode, checkpoint_forward=False))
    checkpointed = ParticleTransformer(replace(eager.config, checkpoint_forward=True))
    checkpointed.load_state_dict(eager.state_dict())
    inputs = features().requires_grad_()
    copied = inputs.detach().clone().requires_grad_()
    expected = eager(inputs, inputs)
    actual = checkpointed(copied, copied)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    before = {name: value.detach().clone() for name, value in eager.state_dict().items()}
    sum(value.square().mean() for value in expected).backward()
    sum(value.square().mean() for value in actual).backward()
    torch.testing.assert_close(inputs.grad, copied.grad)
    for (name, left), right in zip(eager.named_parameters(), checkpointed.parameters()):
        assert left.grad is not None and torch.isfinite(left.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad)
    for name, value in eager.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_global_attention_carries_remote_particle_state_without_a_local_edge():
    torch.manual_seed(17)
    model = ParticleTransformer(config(checkpoint_forward=False)).eval()
    inputs = features(batch=1, particles=3)
    inputs[..., :3] = torch.tensor([[[0., 0., 0.], [0.4, 0.3, 0.2], [0.8, 0.7, 0.6]]])
    inputs.requires_grad_()
    raw = inputs.detach().clone()
    main, _ = model(inputs, raw)
    assert model.neighbor_counts.count_nonzero() == 0
    gradient, = torch.autograd.grad(main[0, 0].square().sum(), inputs)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 1, 3:6].abs().sum() > 0


def test_score_conditioning_changes_geometry_response_for_fixed_node_features():
    torch.manual_seed(29)
    attention = ParticleTransformer(config(relation_mode="score")).blocks[0].local_attention
    receiver = torch.randn(3, 8)
    sender = torch.randn(3, 8)
    relation = torch.randn(3, 7, requires_grad=True)

    def geometry_gradient(material):
        score = attention.relation_scores(receiver, material, relation)
        return torch.autograd.grad(score.square().sum(), relation, retain_graph=True)[0]

    first = geometry_gradient(sender)
    second = geometry_gradient(sender + 1)
    assert (first - second).abs().sum() > 0
    for parameter in attention.score_pair.parameters():
        torch.nn.init.zeros_(parameter)
    torch.testing.assert_close(geometry_gradient(sender), geometry_gradient(sender + 1))


def test_value_conditioning_changes_message_before_neighbor_aggregation():
    torch.manual_seed(29)
    attention = ParticleTransformer(config(relation_mode="value")).blocks[0].local_attention
    receiver_h, sender_h, receiver_z, sender_z = (torch.randn(3, 8) for _ in range(4))
    relation = torch.randn(3, 7)
    first = attention.relation_values(receiver_h, sender_h, receiver_z, sender_z, relation)
    second = attention.relation_values(receiver_h, sender_h, receiver_z, sender_z + 1, relation)
    assert (first - second).abs().sum() > 0
    for parameter in attention.value_pair.parameters():
        torch.nn.init.zeros_(parameter)
    actual = attention.relation_values(receiver_h, sender_h, receiver_z, sender_z, relation)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_worldmodel_reload_restores_local_global_backbone_and_predictions(tmp_path, mode):
    torch.manual_seed(17)
    scene = tiny_scene()
    module = WorldModel(asdict(config(relation_mode=mode)))
    module.validate_scenes([scene])
    module.system.normalizer.fit([scene])
    module.eval()
    path = tmp_path / "model.ckpt"
    torch.save({"pytorch-lightning_version": L.__version__,
                "state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams)}, path)
    restored = WorldModel.load_from_checkpoint(path, map_location="cpu", weights_only=True).eval()
    assert isinstance(restored.system.model, ParticleTransformer)
    inputs = features()
    torch.testing.assert_close(module.system.model(inputs, inputs)[0],
                               restored.system.model(inputs, inputs)[0], atol=0, rtol=0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_grid_size_requires_a_positive_integer(value):
    with pytest.raises(ValueError):
        config(global_grid_size=value)


@pytest.mark.parametrize("kwargs", [
    {"width": 63}, {"heads": 0}, {"blocks": 1}, {"ffn_dim": 0},
    {"radius": 0}, {"radius": float("nan")}, {"max_neighbors": 0},
    {"output_scale": float("inf")}, {"checkpoint_forward": 1},
    {"relation_mode": "unknown"}, {"pair_rank": 0}, {"pair_rank": True},
    {"pair_rank": 1.5}, {"pair_gate_norm": "unknown"},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_model_rejects_legacy_feature_schema_and_mismatched_raw_features():
    model = ParticleTransformer(config())
    with pytest.raises(ValueError, match="features must have shape"):
        model(torch.randn(1, 3, 35), torch.randn(1, 3, 35))
    with pytest.raises(ValueError, match="raw_features"):
        model(features(), features(particles=3))


def test_comparison_matches_every_tensor_shared_by_any_pair_of_variants():
    modules = compare_sparse.matched_modules(tiny_scene(), config(), compare_sparse.TrainConfig())
    common = {}
    assert tuple(modules) == MODES
    for mode, module in modules.items():
        assert module.system.model.config.relation_mode == mode
        for name, value in module.state_dict().items():
            if name in common:
                torch.testing.assert_close(value, common[name], atol=0, rtol=0)
            else:
                common[name] = value


def test_comparison_default_checks_forward_without_an_optimizer_or_results(tmp_path, monkeypatch):
    scene_path = tmp_path / "scene.pt"
    config_path = tmp_path / "config.json"
    output = tmp_path / "results"
    torch.save(tiny_scene(70), scene_path)
    config_path.write_text(json.dumps({"model": asdict(config()), "training": {}}),
                           encoding="utf-8")

    def unexpected_training(*args, **kwargs):
        pytest.fail("Default comparison must only check forward computation")

    monkeypatch.setattr(WorldModel, "configure_optimizers", unexpected_training)
    monkeypatch.setattr(compare_sparse, "run_variant", unexpected_training)
    compare_sparse.main(["--scene", str(scene_path), "--config", str(config_path),
                         "--output", str(output)])
    assert not output.exists()
