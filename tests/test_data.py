import pytest
import torch
from torch.utils.data import default_collate

from unifiedworld.data import (
    FeatureNormalizer,
    SceneDataset,
    canonicalize_scene,
    make_features,
    predict_free_motion,
    scene_physics,
    validate_physics,
)
from unifiedworld.schema import (
    APPLICABLE_SLICE,
    FIXED_MATERIAL_PARAMETERS,
    KNOWN_SLICE,
    OBSERVATION_INDEX,
    PARAM_NAMES,
    schema_metadata,
)


def scene():
    masks = torch.ones(3, 5)
    return {
        "schema": schema_metadata(),
        "boundary": "box",
        "radius": 0.014,
        "density": 1000.0,
        "x": torch.arange(36).reshape(4, 3, 3).float() / 100,
        "v": torch.arange(4).float()[:, None, None].square().expand(4, 3, 3),
        "mass": torch.ones(3, 1),
        "params": torch.tensor([[2, 0, 1e15, 0, 1], [100, 0, 1e10, 300, 1], [0, 0.5, 1e15, 0, 0]]),
        "applicable": masks.clone(),
        "known": masks.clone(),
        "material": torch.arange(3),
        "acceleration": torch.tensor([0, -9.81, 0]),
        "bounds": torch.tensor([[0, 0, 0], [1, 2, 3]]),
        "dt": 0.01,
        "spacing": 0.028,
    }


def features(sample):
    batch = default_collate([canonicalize_scene(sample)])
    x, v = predict_free_motion(
        batch["x"][:, 2], batch["v"][:, 2], batch["acceleration"][:, 2], batch["dt"]
    )
    history = torch.cat(
        (batch["v"][:, 2] - batch["v"][:, 1], batch["v"][:, 1] - batch["v"][:, 0]),
        dim=-1,
    )
    return make_features(
        batch,
        x,
        v,
        history,
        batch["acceleration"][:, 2],
        True,
    )


def test_feature_order_history_and_box_distances():
    sample = scene()
    raw = features(sample)
    expected_v = sample["v"][2] + sample["dt"] * sample["acceleration"]
    expected_x = sample["x"][2] + sample["dt"] * (sample["v"][2] + expected_v) / 2
    assert raw.shape == (1, 3, 38)
    assert PARAM_NAMES == (
        "viscosity",
        "friction",
        "yield_pressure",
        "yield_stress",
        "tensile_yield_ratio",
    )
    torch.testing.assert_close(raw[0, :, :3], expected_x)
    torch.testing.assert_close(raw[0, :, 3:6], expected_v)
    torch.testing.assert_close(raw[..., 6:9], torch.full((1, 3, 3), 3.0))
    torch.testing.assert_close(raw[..., 9:12], torch.ones(1, 3, 3))
    torch.testing.assert_close(raw[0, :, 13:18], sample["params"].float())
    torch.testing.assert_close(raw[0, :, 18:23], sample["applicable"].float())
    torch.testing.assert_close(raw[0, :, 23:28], sample["known"].float())
    torch.testing.assert_close(raw[0, :, 28:31], sample["acceleration"].expand(3, 3))
    torch.testing.assert_close(raw[0, :, 31:34], expected_x)
    torch.testing.assert_close(raw[0, :, 34:37], sample["bounds"][1] - expected_x)
    assert (raw[..., 37] == 1).all()


def test_unknown_parameters_never_leak_and_masks_are_preserved():
    sample = scene()
    sample["known"][:2, 1:3] = 0
    sample["params"] = sample["params"].float()
    sample["params"][1, 1:3] = float("nan")
    sample["params"][0, 1:3] = float("inf")
    raw = features(sample)
    normalizer = FeatureNormalizer().fit([sample])
    normalized = normalizer(raw)
    assert torch.isfinite(normalized).all()
    assert normalized[0, :2, 14:16].eq(0).all()
    torch.testing.assert_close(normalized[..., APPLICABLE_SLICE], raw[..., APPLICABLE_SLICE])
    torch.testing.assert_close(normalized[..., KNOWN_SLICE], raw[..., KNOWN_SLICE])
    torch.testing.assert_close(normalized[..., OBSERVATION_INDEX], raw[..., OBSERVATION_INDEX])


def test_normalizer_source_statistics_transform_and_state_dict():
    source = scene()
    source["params"][0, 0] = 3
    source["params"][1, 0] = 7
    source["params"][1, 3] = 9
    source["params"][2, 1] = 0.7
    normalizer = FeatureNormalizer("log1p_viscosity").fit([source])
    assert normalizer.mean[13] == pytest.approx(torch.tensor([3.0, 7.0, 0]).log1p().mean().item())
    assert normalizer.mean[14] == pytest.approx(0.7 / 3)
    assert normalizer.mean[16] == 3
    expected_x = (
        source["x"][2:]
        + source["dt"] * source["v"][2:]
        + source["dt"] ** 2 * source["acceleration"] / 2
    )
    torch.testing.assert_close(normalizer.mean[:3], expected_x.mean((0, 1)))
    raw = features(source)
    saved = {key: value.clone() for key, value in normalizer.state_dict().items()}
    heldout = scene()
    heldout["x"] += 10
    normalizer(features(heldout))
    restored = FeatureNormalizer("log1p_viscosity")
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored(raw), normalizer(raw))
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, saved[key])


def test_dataset_canonicalizes_and_sorts_without_pickle_objects(tmp_path):
    torch.save(scene(), tmp_path / "b.pt")
    torch.save(scene(), tmp_path / "a.pt")
    dataset = SceneDataset(tmp_path)
    assert len(dataset) == 2
    assert dataset[0]["scene_id"] == "a"
    assert dataset[0]["acceleration"].shape == (4, 3, 3)
    assert (dataset[0]["phase"] == -1).all()
    assert default_collate([dataset[0], dataset[1]])["dt"].shape == (2,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt", 0),
        ("radius", 0),
        ("density", float("nan")),
        ("mass", torch.zeros(3, 1)),
        ("material", torch.tensor([0, 1, 3])),
        ("phase", torch.tensor([0, 1, 2, 4])),
        ("v", torch.zeros(3, 3, 3)),
    ],
)
def test_invalid_scene_rejected(field, value):
    sample = scene()
    sample[field] = value
    with pytest.raises(ValueError):
        canonicalize_scene(sample)


def test_normalizer_must_fit_source_and_rejects_empty_source():
    with pytest.raises(RuntimeError):
        FeatureNormalizer()(features(scene()))
    with pytest.raises(ValueError):
        FeatureNormalizer().fit([])


@pytest.mark.parametrize("field", ["params", "applicable", "known"])
def test_legacy_four_parameter_shape_is_rejected(field):
    sample = scene()
    sample[field] = torch.zeros(3, 4)
    with pytest.raises(ValueError, match=rf"{field} must have shape \(3, 5\).*yield_pressure"):
        canonicalize_scene(sample)


@pytest.mark.parametrize(
    "change", ["missing", "id", "units", "parameter_names", "fixed_value", "fixed_unit"]
)
def test_semantic_schema_mismatch_is_rejected(change):
    sample = scene()
    if change == "missing":
        del sample["schema"]
    elif change == "id":
        sample["schema"]["id"] = "legacy_friction_angle_degrees"
    elif change == "units":
        sample["schema"]["parameter_units"][-1] = "degree"
    elif change == "parameter_names":
        sample["schema"]["parameter_names"][-1] = "friction_angle"
    elif change == "fixed_value":
        sample["schema"]["fixed_material_parameters"]["damping"] = 75
    else:
        sample["schema"]["fixed_material_parameter_units"]["damping"] = "1/s"
    with pytest.raises(ValueError, match="schema"):
        canonicalize_scene(sample)


def test_schema_units_and_physics_roundtrip_through_safe_storage(tmp_path):
    path = tmp_path / "scene.pt"
    torch.save(canonicalize_scene(scene()), path)
    restored = canonicalize_scene(torch.load(path, weights_only=True))
    assert restored["schema"] == schema_metadata()
    assert restored["schema"]["fixed_material_parameters"] == FIXED_MATERIAL_PARAMETERS
    assert "young_modulus" not in restored["schema"]["parameter_names"]
    assert restored["boundary"] == "box"
    validate_physics(restored, scene_physics(scene()))


@pytest.mark.parametrize("boundary", [None, "floor", "mesh"])
def test_non_box_boundary_rejected(boundary):
    sample = scene()
    sample["boundary"] = boundary
    with pytest.raises(ValueError, match="boundary='box'"):
        canonicalize_scene(sample)


@pytest.mark.parametrize(
    "channel,particle,value,message",
    [
        (0, 0, -1, "nonnegative"),
        (1, 2, -1, "nonnegative"),
        (2, 1, 0, "positive"),
        (3, 1, -1, "nonnegative"),
        (4, 1, 1.1, "in \\[0,1\\]"),
    ],
)
def test_visible_parameter_domains_are_checked(channel, particle, value, message):
    sample = scene()
    sample["params"][particle, channel] = value
    with pytest.raises(ValueError, match=message):
        canonicalize_scene(sample)


def test_material_applicability_must_match_material_ids():
    sample = scene()
    sample["applicable"][0, 0] = 0
    with pytest.raises(ValueError, match="applicability"):
        canonicalize_scene(sample)


@pytest.mark.parametrize("particle,channel,value", [(0, 3, 1), (1, 1, 0.5), (2, 0, 1)])
def test_material_family_fixed_coefficients_are_checked(particle, channel, value):
    sample = scene()
    sample["params"][particle, channel] = value
    with pytest.raises(ValueError, match="has fixed"):
        canonicalize_scene(sample)


def test_known_zero_is_visible_and_differs_from_unknown():
    sample = scene()
    sample["params"][0, 0] = 0
    raw = features(sample)
    assert raw[0, 0, 13] == 0
    assert raw[0, 0, KNOWN_SLICE.start] == 1
    sample["known"][0, 0] = 0
    sample["params"][0, 0] = float("nan")
    hidden = features(sample)
    assert hidden[0, 0, 13] == 0
    assert hidden[0, 0, KNOWN_SLICE.start] == 0


@pytest.mark.parametrize("field", ["radius", "density", "dt"])
def test_physical_constants_must_match_while_box_bounds_can_vary(field):
    reference = scene_physics(scene())
    sample = scene()
    sample["bounds"][1] *= 2
    validate_physics(canonicalize_scene(sample), reference)
    sample[field] *= 2
    with pytest.raises(ValueError, match="fixed radius, density, and dt"):
        validate_physics(canonicalize_scene(sample), reference)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), [0.01, 0.02]])
def test_fixed_physics_metadata_rejects_invalid_scalar_values(value):
    physics = scene_physics(scene())
    physics["radius"] = value
    with pytest.raises(ValueError, match="finite positive scalar"):
        scene_physics(physics)
