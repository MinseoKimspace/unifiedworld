from copy import deepcopy

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from unifiedworld.data import canonicalize_scene
from unifiedworld.experiment import (
    PROTOCOL,
    checkpoint_restore,
    checkpoint_save,
    contact_window,
    evaluate_contact,
    train_update,
)
from unifiedworld.schema import schema_metadata
from unifiedworld.training import WorldModel


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_scene(steps=7):
    material = torch.arange(3).repeat_interleave(2)
    velocity = torch.full((steps, 6, 3), 0.1)
    base = torch.linspace(0.2, 0.7, 18).reshape(6, 3)
    return canonicalize_scene({
        "x": base[None] + torch.arange(steps)[:, None, None] * 0.01 * velocity,
        "v": velocity,
        "mass": torch.ones(6, 1),
        "params": torch.tensor([[0, 0, 1e15, 0, 1], [100, 0, 1e10, 300, 1],
                                [0, 0.5, 1e15, 0, 0]])[material],
        "applicable": torch.ones(6, 5), "known": torch.ones(6, 5),
        "material": material, "acceleration": torch.zeros(3),
        "bounds": torch.tensor([[0, 0, 0], [1, 1, 1]]),
        "dt": 0.01, "spacing": 0.028, "radius": 0.014, "density": 1000.0,
        "schema": schema_metadata(), "boundary": "box",
    })


def fresh_module(scene):
    L.seed_everything(17)
    module = WorldModel(
        {"width": 8, "heads": 2, "ffn_dim": 16, "radius": 0.3, "max_neighbors": 4},
        {"batch_size": 4, "tbptt": 32},
    )
    module.validate_scenes([scene])
    module.system.normalizer.fit([scene])
    return module


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for actual, expected in zip(left, right):
            assert_same(actual, expected)
    else:
        assert left == right


def test_direct_update_matches_original_lightning_math(tmp_path):
    scene = tiny_scene()
    reference = fresh_module(scene)
    direct = fresh_module(scene)
    optimizer = direct.configure_optimizers()
    trainer = L.Trainer(
        default_root_dir=tmp_path, accelerator="cpu", devices=1,
        max_steps=1, max_epochs=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, deterministic=True,
    )
    trainer.fit(reference, train_dataloaders=DataLoader([scene] * 4, batch_size=4))
    actual_loss = train_update(direct, default_collate([scene] * 4), optimizer)

    assert actual_loss == trainer.callback_metrics["train_loss_step"].item()
    assert_same(direct.state_dict(), reference.state_dict())
    assert_same(optimizer.state_dict(), trainer.optimizers[0].state_dict())


def test_contact_checkpoint_preserves_warm_adam_and_next_update(tmp_path):
    scene = tiny_scene()
    original = fresh_module(scene)
    optimizer = original.configure_optimizers()
    batch = default_collate([scene] * 4)
    loss = train_update(original, batch, optimizer)
    metadata = {
        "completed_updates": 1, "source_step": 0,
        "training_history": [{"update": 1, "train_loss": loss}],
        "evaluations": [], "success_streak": 0,
        "provenance": {"purpose": "contact_resume_test"},
    }
    saved_model = deepcopy(original.state_dict())
    saved_optimizer = deepcopy(optimizer.state_dict())
    path = tmp_path / "progress.ckpt"
    checkpoint_save(path, original, optimizer, metadata)
    expected_random = torch.rand(5)
    expected_loss = train_update(original, batch, optimizer)

    resumed, resumed_optimizer, resumed_metadata = checkpoint_restore(path, device="cpu")
    assert_same(resumed_metadata, metadata)
    assert_same(resumed.state_dict(), saved_model)
    assert_same(resumed_optimizer.state_dict(), saved_optimizer)
    torch.testing.assert_close(torch.rand(5), expected_random, atol=0, rtol=0)
    assert not path.with_name(f".{path.name}.tmp").exists()

    resumed_loss = train_update(resumed, batch, resumed_optimizer)
    assert resumed_loss == expected_loss
    assert_same(resumed.state_dict(), original.state_dict())
    assert_same(resumed_optimizer.state_dict(), optimizer.state_dict())


def test_contact_window_indices_and_rollout_do_not_read_future_targets():
    scene = tiny_scene(70)
    window = contact_window(scene)
    for key in ("x", "v", "acceleration", "phase"):
        torch.testing.assert_close(window[key], scene[key][32:67], atol=0, rtol=0)
    assert len(window["x"]) == 35
    module = fresh_module(window)
    report, original = evaluate_contact(module, window, "cpu")
    altered = deepcopy(window)
    altered["x"][3:] += 0.15
    altered["v"][3:] += 1.0
    changed_report, changed = evaluate_contact(module, altered, "cpu")

    torch.testing.assert_close(changed["x"], original["x"], atol=0, rtol=0)
    torch.testing.assert_close(changed["v"], original["v"], atol=0, rtol=0)
    assert changed_report["overall"] != report["overall"]


@pytest.mark.parametrize("component", ["position", "velocity"])
def test_success_requires_final_frame_for_every_material(monkeypatch, component):
    scene = tiny_scene(35)
    module = fresh_module(scene)
    tolerance_key = f"{component}_tolerance_" + ("m" if component == "position" else "m_s")
    threshold = PROTOCOL[tolerance_key]
    metric_key = f"{component}_rms_" + ("m" if component == "position" else "m_s")

    def final_frame_failure(system, batch, steps):
        x, v = batch["x"][:, 3:].clone(), batch["v"][:, 3:].clone()
        output = x if component == "position" else v
        output[:, -1, batch["material"][0] == 0, 0] += 2 * threshold
        return x, v

    monkeypatch.setattr("unifiedworld.experiment.free_rollout", final_frame_failure)
    report, _ = evaluate_contact(module, scene, "cpu")
    water = report["per_material"]["water"]
    assert water["whole_window"][metric_key] <= threshold
    assert water["final_frame"][metric_key] > threshold
    assert not report["passed"]
