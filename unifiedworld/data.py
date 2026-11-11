from collections.abc import Iterable, Mapping
from math import isclose
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .schema import (
    APPLICABLE_SLICE,
    INPUT_DIM,
    KNOWN_SLICE,
    MATERIAL_APPLICABILITY,
    MATERIAL_FIXED_PARAMETERS,
    MATERIAL_NAMES,
    OBSERVATION_INDEX,
    PARAM_DIM,
    PARAM_NAMES,
    PARAM_SLICE,
    schema_metadata,
    validate_schema,
)


def validate_material_parameters(params: Tensor, material: Tensor, visible: Tensor | None = None):
    params, material = torch.as_tensor(params), torch.as_tensor(material)
    if params.ndim != 2 or params.shape[1] != PARAM_DIM or material.shape != (len(params),):
        raise ValueError(f"Material parameters must have shape [N,{PARAM_DIM}] with [N] IDs")
    if not (
        (material >= 0) & (material < len(MATERIAL_NAMES)) & (material == material.long())
    ).all():
        raise ValueError("Material IDs must be integers for water, mud, or sand")
    if visible is None:
        visible = torch.ones_like(params, dtype=torch.bool)
    elif visible.shape != params.shape or not ((visible == 0) | (visible == 1)).all():
        raise ValueError("Material visibility must be a matching binary mask")
    visible = visible.bool()
    values = params[visible]
    if not values.isfinite().all() or (values < 0).any():
        raise ValueError("Visible material parameters must be finite and nonnegative")
    if (visible[:, 2] & (params[:, 2] <= 0)).any():
        raise ValueError("Visible yield_pressure must be positive")
    if (visible[:, 4] & (params[:, 4] > 1)).any():
        raise ValueError("Visible tensile_yield_ratio must be in [0,1]")
    for kind, fixed in enumerate(MATERIAL_FIXED_PARAMETERS):
        for name, value in fixed.items():
            channel = PARAM_NAMES.index(name)
            mask = (material == kind) & visible[:, channel]
            expected = params.new_tensor(value)
            if not torch.isclose(params[mask, channel], expected, rtol=1e-6, atol=0).all():
                raise ValueError(f"{MATERIAL_NAMES[kind]} has fixed {name}={value}")


def canonicalize_scene(scene: Mapping) -> dict:
    validate_schema(scene.get("schema"))
    if scene.get("boundary") != "box":
        raise ValueError("Scenes must declare boundary='box' with physical walls at bounds")
    required = {
        "x",
        "v",
        "mass",
        "params",
        "applicable",
        "known",
        "material",
        "acceleration",
        "bounds",
        "dt",
        "spacing",
        "radius",
        "density",
    }
    if missing := required - scene.keys():
        raise ValueError(f"Missing scene fields: {sorted(missing)}")
    result = {
        key: torch.as_tensor(scene[key], dtype=torch.float32).detach().cpu()
        for key in required - {"material"}
    }
    x, v = result["x"], result["v"]
    if x.ndim != 3 or x.shape[-1] != 3 or len(x) < 3 or v.shape != x.shape:
        raise ValueError("x and v must have matching [T>=3,N,3] shapes")
    steps, particles, _ = x.shape
    if particles == 0:
        raise ValueError("A scene must contain particles")
    shapes = {
        "mass": (particles, 1),
        "params": (particles, PARAM_DIM),
        "applicable": (particles, PARAM_DIM),
        "known": (particles, PARAM_DIM),
        "bounds": (2, 3),
    }
    for key, shape in shapes.items():
        if result[key].shape != shape:
            channels = f" for {PARAM_NAMES}" if key in {"params", "applicable", "known"} else ""
            raise ValueError(f"{key} must have shape {shape}{channels}")
    for key in ("applicable", "known"):
        if not ((result[key] == 0) | (result[key] == 1)).all():
            raise ValueError(f"{key} must contain binary masks")
    visible = result["applicable"].bool() & result["known"].bool()
    result["params"] = torch.where(visible, result["params"], 0)
    acceleration = result["acceleration"]
    if acceleration.shape not in {(3,), (particles, 3), (steps, particles, 3)}:
        raise ValueError("acceleration must have shape [3], [N,3], or [T,N,3]")
    result["acceleration"] = acceleration.expand(steps, particles, 3)
    for key in ("dt", "spacing", "radius", "density"):
        if result[key].numel() != 1 or result[key].item() <= 0:
            raise ValueError(f"{key} must be a positive scalar")
        result[key] = result[key].reshape(())
    if (result["mass"] <= 0).any() or (result["bounds"][1] <= result["bounds"][0]).any():
        raise ValueError("Masses and box side lengths must be positive")
    if any(not value.isfinite().all() for value in result.values()):
        raise ValueError("Visible scene values must be finite")
    material = torch.as_tensor(scene["material"]).detach().cpu()
    if (
        material.shape != (particles,)
        or not ((material >= 0) & (material <= 2) & (material == material.long())).all()
    ):
        raise ValueError("material must have shape [N] with integer IDs 0, 1, or 2")
    result["material"] = material.long()
    expected = result["applicable"].new_tensor(MATERIAL_APPLICABILITY)[result["material"]]
    if not torch.equal(result["applicable"], expected):
        raise ValueError("Material applicability masks do not match material IDs")
    validate_material_parameters(result["params"], result["material"], visible)
    phase = torch.as_tensor(scene.get("phase", torch.full((steps,), -1))).detach().cpu()
    if (
        phase.shape != (steps,)
        or not ((phase >= -1) & (phase <= 3) & (phase == phase.long())).all()
    ):
        raise ValueError("phase must have shape [T] with integer IDs 0..3 (-1 when unavailable)")
    result["phase"] = phase.long()
    for key in ("scene_id", "recipe", "group", "restart_kind", "reference"):
        result[key] = str(scene.get(key, ""))
    result["schema"] = schema_metadata()
    result["boundary"] = "box"
    return result


def scene_physics(scene: Mapping) -> dict:
    keys = {"radius", "density", "dt"}
    if missing := keys - scene.keys():
        raise ValueError(f"Missing fixed physics fields: {sorted(missing)}")
    physics = {}
    for key in ("radius", "density", "dt"):
        value = torch.as_tensor(scene[key], dtype=torch.float64)
        if value.numel() != 1 or not value.isfinite().all() or value.item() <= 0:
            raise ValueError(f"Fixed {key} must be a finite positive scalar")
        physics[key] = value.item()
    return physics


def validate_physics(scene: Mapping, expected: Mapping) -> None:
    physics, reference = scene_physics(scene), scene_physics(expected)
    if any(not isclose(physics[key], reference[key], rel_tol=1e-6, abs_tol=0.0) for key in physics):
        raise ValueError("All scenes must use the checkpoint's fixed radius, density, and dt")


class SceneDataset(Dataset):
    def __init__(self, directory: str | Path):
        self.files = sorted(Path(directory).glob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt scenes in {directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        scene = canonicalize_scene(torch.load(path, map_location="cpu", weights_only=True))
        scene["scene_id"] = scene["scene_id"] or path.stem
        return scene


def predict_free_motion(x: Tensor, v: Tensor, acceleration: Tensor, dt: Tensor):
    dt = dt.reshape(-1, 1, 1)
    free_v = v + dt * acceleration
    free_x = x + dt * (v + free_v) / 2
    return free_x, free_v


def make_features(
    batch: Mapping[str, Tensor],
    x: Tensor,
    v: Tensor,
    history: Tensor,
    acceleration: Tensor,
    observed: bool,
) -> Tensor:
    low, high = batch["bounds"][:, 0, None], batch["bounds"][:, 1, None]
    visible = batch["applicable"].bool() & batch["known"].bool()
    features = torch.cat(
        (
            x,
            v,
            history,
            batch["mass"],
            torch.where(visible, batch["params"], 0),
            batch["applicable"],
            batch["known"],
            acceleration,
            x - low,
            high - x,
            x.new_full((*x.shape[:-1], 1), float(observed)),
        ),
        dim=-1,
    )
    return features


class FeatureNormalizer(nn.Module):
    def __init__(self, params_transform: str = "identity"):
        super().__init__()
        if params_transform not in {"identity", "log1p_viscosity"}:
            raise ValueError("params_transform must be identity or log1p_viscosity")
        self.params_transform = params_transform
        self.register_buffer("mean", torch.zeros(INPUT_DIM))
        self.register_buffer("std", torch.ones(INPUT_DIM))
        self.register_buffer("fitted", torch.tensor(False))

    def _transform(self, features: Tensor) -> Tensor:
        if self.params_transform == "identity":
            return features
        viscosity = PARAM_SLICE.start
        if (features[..., viscosity] < 0).any():
            raise ValueError("log1p_viscosity requires nonnegative viscosity")
        transformed = features.clone()
        transformed[..., viscosity] = features[..., viscosity].log1p()
        return transformed

    @staticmethod
    def _mask(features: Tensor) -> Tensor:
        mask = torch.ones_like(features, dtype=torch.bool)
        mask[..., PARAM_SLICE] = (
            features[..., APPLICABLE_SLICE].bool() & features[..., KNOWN_SLICE].bool()
        )
        mask[..., APPLICABLE_SLICE] = False
        mask[..., KNOWN_SLICE] = False
        mask[..., OBSERVATION_INDEX] = False
        return mask

    @torch.no_grad()
    def fit(self, source: Iterable[Mapping]) -> "FeatureNormalizer":
        count, total, square = (torch.zeros(INPUT_DIM, dtype=torch.float64) for _ in range(3))
        for source_scene in source:
            scene = canonicalize_scene(source_scene)
            length = len(scene["x"]) - 2
            static = {
                key: scene[key].unsqueeze(0).expand(length, *scene[key].shape)
                for key in ("mass", "params", "applicable", "known", "bounds")
            }
            free_x, free_v = predict_free_motion(
                scene["x"][2:],
                scene["v"][2:],
                scene["acceleration"][2:],
                scene["dt"].expand(length),
            )
            history = torch.cat(
                (scene["v"][2:] - scene["v"][1:-1], scene["v"][1:-1] - scene["v"][:-2]),
                dim=-1,
            )
            raw = make_features(
                static,
                free_x,
                free_v,
                history,
                scene["acceleration"][2:],
                True,
            )
            transformed = self._transform(raw).double().reshape(-1, INPUT_DIM)
            mask = self._mask(raw).reshape(-1, INPUT_DIM)
            count += mask.sum(0)
            total += torch.where(mask, transformed, 0).sum(0)
            square += torch.where(mask, transformed.square(), 0).sum(0)
        if count[0] == 0:
            raise ValueError("Cannot fit normalization on an empty source")
        mean = total / count.clamp_min(1)
        variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
        std = torch.where(variance > 1e-12, variance.sqrt(), 1)
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.fitted.fill_(True)
        return self

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != INPUT_DIM:
            raise ValueError(f"features must have {INPUT_DIM} channels")
        if not self.fitted:
            raise RuntimeError("Fit normalization on source training scenes before use")
        transformed = self._transform(features)
        normalized = torch.where(
            self._mask(features), (transformed - self.mean) / self.std, features
        )
        visible = features[..., APPLICABLE_SLICE].bool() & features[..., KNOWN_SLICE].bool()
        normalized = normalized.clone()
        normalized[..., PARAM_SLICE] = torch.where(visible, normalized[..., PARAM_SLICE], 0)
        return normalized
