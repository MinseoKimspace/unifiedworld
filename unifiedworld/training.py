from dataclasses import asdict, dataclass
from math import isfinite

import lightning as L
import torch

from .data import FeatureNormalizer, scene_physics, validate_physics
from .dynamics import ParticleSystem, State, state_loss
from .model import ModelConfig, ParticleTransformer
from .schema import schema_metadata, validate_schema


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    tbptt: int = 32

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if any(type(value) is not int or value < 1 for value in (self.batch_size, self.tbptt)):
            raise ValueError("Batch and TBPTT counts must be positive integers")
        if not all(isfinite(value) for value in (self.lr, self.weight_decay)):
            raise ValueError("Training settings must be finite")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid AdamW settings")


class WorldModel(L.LightningModule):
    def __init__(
        self,
        model_config=None,
        train_config=None,
        params_transform="identity",
        provenance=None,
        schema=None,
        physics=None,
    ):
        super().__init__()
        schema = schema_metadata() if schema is None else schema
        validate_schema(schema)
        model_config = ModelConfig(**(model_config or {}))
        self.config = TrainConfig(**(train_config or {}))
        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "train_config": asdict(self.config),
                "params_transform": params_transform,
                "provenance": provenance or {},
                "schema": schema,
                "physics": None if physics is None else scene_physics(physics),
            }
        )
        backbone = ParticleTransformer(model_config)
        self.system = ParticleSystem(backbone, FeatureNormalizer(params_transform))
        self.automatic_optimization = False

    def validate_scenes(self, scenes):
        for scene in scenes:
            validate_schema(scene.get("schema"))
            if self.hparams.physics is None:
                self.hparams["physics"] = scene_physics(scene)
            validate_physics(scene, self.hparams.physics)

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.physics is None:
            raise ValueError("Checkpoints require fixed radius, density, and dt")
        validate_schema(self.hparams.schema)
        scene_physics(self.hparams.physics)

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("hyper_parameters", {})
        validate_schema(saved.get("schema"))
        if saved.get("physics") is None:
            raise ValueError("Checkpoint is missing fixed reference physics")
        validate_physics(self.hparams.physics, scene_physics(saved["physics"]))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

    def _loss(self, state, auxiliary, batch, t):
        material = batch["material"]
        target_x, target_v = batch["x"][:, t], batch["v"][:, t]
        loss = state_loss(state.x, state.v, target_x, target_v, material)
        if auxiliary is not None:
            loss = loss + 0.1 * state_loss(
                auxiliary[0], auxiliary[1], target_x, target_v, material
            )
        return loss

    def training_step(self, batch, batch_idx):
        horizon = batch["x"].shape[1] - 3
        optimizer = self.optimizers()
        optimizer.zero_grad()
        state = State.from_batch(batch)
        total = state.x.new_zeros(())
        chunk = state.x.new_zeros(())
        for offset in range(horizon):
            t = 2 + offset
            state, auxiliary = self.system.step(batch, state, t)
            loss = self._loss(state, auxiliary, batch, t + 1)
            chunk = chunk + loss
            if (offset + 1) % self.config.tbptt == 0 or offset + 1 == horizon:
                if not torch.isfinite(chunk):
                    raise RuntimeError(
                        "Nonfinite rollout loss; inspect data, units, and normalization"
                    )
                self.manual_backward(chunk / horizon)
                total = total + chunk.detach()
                chunk = state.x.new_zeros(())
                state = state.detach()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in self.parameters()
        ):
            raise RuntimeError("Nonfinite gradients; optimizer state was not updated")
        optimizer.step()
        self.log(
            "train_loss", total / horizon, on_step=True, on_epoch=True, batch_size=state.x.shape[0]
        )
        counts = self.system.model.neighbor_counts
        self.log(
            "neighbor_cap_fraction", (counts > self.system.model.config.max_neighbors).float().mean(),
            on_step=True, on_epoch=True, batch_size=state.x.shape[0],
        )
        self.log("neighbor_max", counts.max().float(), on_step=True, on_epoch=False)

