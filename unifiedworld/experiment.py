"""Compare four local/global attention variants; training requires --train."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".warp-cache"))

import lightning as L
import torch
from torch.utils.data import default_collate

from .data import canonicalize_scene, predict_free_motion
from .dynamics import State, free_rollout
from .model import ModelConfig
from .render import render_video
from .schema import MATERIAL_NAMES
from .training import TrainConfig, WorldModel

PROTOCOL = {
    "start_frame": 34, "horizon": 32, "max_updates": 500, "evaluate_every": 25,
    "position_tolerance_m": 0.00125, "velocity_tolerance_m_s": 0.01,
    "consecutive_passes": 3,
}


VARIANTS = ("concat", "score", "value", "score_value")


LABELS = {"concat": "Concat", "score": "Score", "value": "Value",
          "score_value": "Score + Value"}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def device_batch(scene, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in default_collate([scene]).items()}


def errors(px, pv, tx, tv):
    return {"position_rms_m": (px.double() - tx.double()).square().sum(-1).mean().sqrt().item(),
            "velocity_rms_m_s": (pv.double() - tv.double()).square().sum(-1).mean().sqrt().item()}


def contact_window(scene, start=34, horizon=32):
    if start < 2 or start + horizon >= len(scene["x"]):
        raise ValueError("Contact window requires three observations and all target frames")
    return {key: value[start - 2:start + horizon + 1]
            if key in {"x", "v", "acceleration", "phase"} else value
            for key, value in scene.items()}


def train_update(module, batch, optimizer):
    """Same one-chunk BPTT, auxiliary loss, and AdamW update as WorldModel.training_step."""
    horizon = batch["x"].shape[1] - 3
    if not 0 < horizon <= module.config.tbptt:
        raise ValueError("This diagnostic uses a single full BPTT chunk")
    module.train()
    optimizer.zero_grad()
    state = State.from_batch(batch)
    loss = state.x.new_zeros(())
    for t in range(2, horizon + 2):
        state, auxiliary = module.system.step(batch, state, t)
        loss = loss + module._loss(state, auxiliary, batch, t + 1)
    loss = loss / horizon
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite loss; optimizer was not updated")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
        raise RuntimeError("Nonfinite gradients; optimizer was not updated")
    optimizer.step()
    return loss.detach().item()


def checkpoint_save(path, module, optimizer, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "format": "fixed_contact_overfit.v1",
        "pytorch-lightning_version": L.__version__,
        "state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams),
        "optimizer_states": [optimizer.state_dict()],
        "global_step": metadata["source_step"] + metadata["completed_updates"],
        "contact_overfit": metadata, "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def checkpoint_restore(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "fixed_contact_overfit.v1":
        raise ValueError("Use a checkpoint from this fixed-contact diagnostic")
    metadata = saved["contact_overfit"]
    if saved["global_step"] != metadata["source_step"] + metadata["completed_updates"]:
        raise ValueError("Diagnostic and cumulative update counters are inconsistent")
    module = WorldModel.load_from_checkpoint(path, map_location="cpu", weights_only=True).to(device)
    optimizer = module.configure_optimizers()
    optimizer.load_state_dict(saved["optimizer_states"][0])
    torch.set_rng_state(saved["rng_state"])
    if saved["cuda_rng_state"] and str(device).startswith("cuda"):
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    return module, optimizer, metadata


@torch.inference_mode()
def evaluate_contact(module, window, device):
    module.eval()
    batch = device_batch(window, device)
    x, v = free_rollout(module.system, batch, PROTOCOL["horizon"])
    finite = bool(torch.isfinite(x).all() and torch.isfinite(v).all())
    if not finite:
        raise RuntimeError("Nonfinite diagnostic rollout")
    tx, tv = batch["x"][:, 3:], batch["v"][:, 3:]
    report = {"overall": errors(x, v, tx, tv), "per_material": {}, "finite": finite}
    for kind, name in enumerate(MATERIAL_NAMES):
        mask = batch["material"][0] == kind
        report["per_material"][name] = {
            "whole_window": errors(x[:, :, mask], v[:, :, mask], tx[:, :, mask], tv[:, :, mask]),
            "final_frame": errors(x[:, -1, mask], v[:, -1, mask], tx[:, -1, mask], tv[:, -1, mask]),
        }
    bounds = batch["bounds"][:, None]
    report["max_outside_m"] = torch.maximum(
        bounds[:, :, 0, None] - x, x - bounds[:, :, 1, None]
    ).clamp_min(0).max().item()
    report["passed"] = all(
        point["position_rms_m"] <= PROTOCOL["position_tolerance_m"]
        and point["velocity_rms_m_s"] <= PROTOCOL["velocity_tolerance_m_s"]
        for material in report["per_material"].values() for point in material.values()
    )
    prediction = {"x": torch.cat((window["x"][:3], x[0].cpu())),
                  "v": torch.cat((window["v"][:3], v[0].cpu())),
                  **{key: window[key] for key in ("material", "bounds", "dt", "schema")},
                  "absolute_frames": [32, 66]}
    return report, prediction


@torch.inference_mode()
def teacher_forced_metrics(module, window, device):
    """Reference states, using the same observation-flag schedule as the rollout."""
    module.eval()
    batch = device_batch(window, device)
    xs, vs, free_xs, free_vs = [], [], [], []
    for t in range(2, len(window["x"]) - 1):
        state = State.from_batch(batch, start=t)
        state.observed = t == 2
        fx, fv = predict_free_motion(state.x, state.v, batch["acceleration"][:, t], batch["dt"])
        predicted, _ = module.system.step(batch, state, t)
        xs.append(predicted.x)
        vs.append(predicted.v)
        free_xs.append(fx)
        free_vs.append(fv)
    x, v, fx, fv = (torch.stack(values, 1) for values in (xs, vs, free_xs, free_vs))
    tx, tv = batch["x"][:, 3:], batch["v"][:, 3:]
    if not (torch.isfinite(x).all() and torch.isfinite(v).all()):
        raise RuntimeError("Nonfinite teacher-forced predictions")
    return {"prediction": errors(x, v, tx, tv), "free_motion": errors(fx, fv, tx, tv)}


def matched_modules(scene, model_config, train_config):
    """Identical common tensors and fresh optimizers for every condition."""
    modules = {}
    common = {}
    for variant in VARIANTS:
        L.seed_everything(train_config.seed, workers=True)
        config = replace(model_config, relation_mode=variant)
        module = WorldModel(asdict(config), asdict(train_config))
        module.system.normalizer.fit([scene])
        module.validate_scenes([scene])
        state = module.state_dict()
        for name, value in state.items():
            key = (name, tuple(value.shape))
            if key not in common:
                common[key] = value.detach().clone()
            state[name] = common[key]
        module.load_state_dict(state, strict=True)
        modules[variant] = module
    return modules


def experiment_identity(scene_path, config_path, modules, variants):
    sources = [ROOT / "unifiedworld" / name for name in (
        "experiment.py", "model.py", "data.py", "schema.py", "dynamics.py",
        "neighbors.py", "training.py", "render.py",
    )]
    identity = {
        "purpose": "matched_fixed_contact_ablation_not_generalization",
        "initialization": "fresh_seeded_identical_common_tensors_fresh_adamw",
        "scene": str(scene_path.resolve()), "scene_sha256": digest(scene_path),
        "config": str(config_path.resolve()), "config_sha256": digest(config_path),
        "protocol": dict(PROTOCOL), "variants": list(variants),
        "training": asdict(modules["concat"].config),
        "models": {name: asdict(modules[name].system.model.config) for name in variants},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
                          for path in sources},
    }
    return json.loads(json.dumps(identity))


@torch.inference_mode()
def check_forward(modules, scene, variants):
    """Exercise the normal particle-system entry point without optimizer creation."""
    batch = device_batch(contact_window(scene), "cpu")
    for variant in variants:
        module = modules[variant].eval()
        predicted, _ = module.system.step(batch, State.from_batch(batch), 2)
        if not (torch.isfinite(predicted.x).all() and torch.isfinite(predicted.v).all()):
            raise RuntimeError(f"Nonfinite preflight output for {variant}")


def postprocess(module, scene, output, device, variant):
    window = contact_window(scene)
    module.to(device).eval()
    report, prediction = evaluate_contact(module, window, device)
    report["teacher_forced"] = teacher_forced_metrics(module, window, device)
    torch.save(prediction, output / "prediction-after.pt")
    save_json(output / "metrics-after.json", report)
    video = output / "comparison.mp4"
    if not video.exists():
        render_video({"Reference": window["x"], LABELS[variant]: prediction["x"]},
                     window["material"], window["bounds"], video,
                     fps=1 / float(window["dt"]))
    report["video"] = str(video.resolve())
    module.cpu()
    return report


def run_variant(module, scene, output, identity, variant, device, resume, postprocess_only):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "checkpoints/last.ckpt"
    provenance = identity | {"variant": variant}
    if path.exists():
        if not (resume or postprocess_only):
            raise FileExistsError("Use an empty output directory or --resume")
        module, optimizer, metadata = checkpoint_restore(path, device)
        if metadata["provenance"] != provenance:
            raise ValueError("Resume requires the same scene, configuration, code and protocol")
    else:
        if postprocess_only:
            raise FileNotFoundError(path)
        if any(output.iterdir()):
            raise FileExistsError(f"Incomplete output without a checkpoint: {output}")
        module.hparams["provenance"] = provenance
        module.to(device)
        optimizer = module.configure_optimizers()
        metadata = {"source_step": 0, "completed_updates": 0, "training_history": [],
                    "evaluations": [], "success_streak": 0, "provenance": provenance}
        before, prediction = evaluate_contact(module, contact_window(scene), device)
        save_json(output / "metrics-before.json", before)
        torch.save(prediction, output / "prediction-before.pt")
        checkpoint_save(path, module, optimizer, metadata)
    save_json(output / "settings.json", provenance)
    if not postprocess_only:
        window = contact_window(scene)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in default_collate([window] * module.config.batch_size).items()}
        save_json(output / "status.json", {"status": "training", "pid": os.getpid(),
                                           "completed_updates": metadata["completed_updates"]})
        try:
            for update in range(metadata["completed_updates"] + 1, PROTOCOL["max_updates"] + 1):
                if metadata["success_streak"] >= PROTOCOL["consecutive_passes"]:
                    break
                started = time.monotonic()
                loss = train_update(module, batch, optimizer)
                metadata["completed_updates"] = update
                record = {"completed_updates": update, "train_loss": loss,
                          "seconds": time.monotonic() - started}
                metadata["training_history"].append(record)
                if update % PROTOCOL["evaluate_every"] == 0:
                    report, _ = evaluate_contact(module, window, device)
                    passed = metadata["success_streak"] + 1 if report["passed"] else 0
                    metadata["success_streak"] = passed
                    metadata["evaluations"].append(report | {"completed_updates": update})
                checkpoint_save(path, module, optimizer, metadata)
                save_json(output / "progress.json", metadata)
                print(json.dumps({"variant": variant, **record}), flush=True)
        except BaseException as error:
            save_json(output / "status.json", {"status": "interrupted_or_failed",
                                               "error": str(error),
                                               "resume_from": str(path.resolve())})
            raise
    report = postprocess(module, scene, output, device, variant)
    report.update(completed_updates=metadata["completed_updates"],
                  success_streak=metadata["success_streak"],
                  success=metadata["success_streak"] >= PROTOCOL["consecutive_passes"],
                  parameters=sum(p.numel() for p in module.parameters()))
    save_json(output / "status.json", {"status": "completed", **report})
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "data/scene.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/comparison")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--postprocess", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must be unique")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    model, train = ModelConfig(**settings["model"]), TrainConfig(**settings["training"])
    if train.tbptt != 32 or train.batch_size != 4:
        raise ValueError("Contact comparison uses the approved batch=4, BPTT=32 protocol")
    scene = canonicalize_scene(torch.load(args.scene, map_location="cpu", weights_only=True))
    contact_window(scene)
    modules = matched_modules(scene, model, train)
    identity = experiment_identity(args.scene, args.config, modules, args.variants)
    if not (args.train or args.postprocess):
        check_forward(modules, scene, args.variants)
    print(json.dumps({"train": args.train, "postprocess": args.postprocess,
                      "forward_checked": not (args.train or args.postprocess),
                      "initialization": identity["initialization"], "protocol": PROTOCOL,
                      "variants": {name: {"parameters": sum(p.numel() for p in modules[name].parameters()),
                                          "model": identity["models"][name]}
                                   for name in args.variants}}, indent=2), flush=True)
    if not (args.train or args.postprocess):
        return
    identity_path = args.output / "settings.json"
    if args.output.exists() and any(args.output.iterdir()):
        if not (args.resume or args.postprocess):
            raise FileExistsError("Use an empty output directory or --resume")
        if not identity_path.exists() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Comparison identity changed")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(identity_path, identity)
    reports = {}
    for variant in args.variants:
        reports[variant] = run_variant(modules[variant], scene, args.output / variant, identity,
                                      variant, args.device, args.resume, args.postprocess)
        save_json(args.output / "summary.json", reports)
    predictions = {"Reference": contact_window(scene)["x"]}
    for variant in args.variants:
        predictions[LABELS[variant]] = torch.load(
            args.output / variant / "prediction-after.pt", map_location="cpu", weights_only=True)["x"]
    video = args.output / "comparison.mp4"
    if not video.exists():
        render_video(predictions, scene["material"], scene["bounds"], video,
                     fps=1 / float(scene["dt"]))
    print(json.dumps({"status": "completed",
                      "summary": str((args.output / "summary.json").resolve())}), flush=True)


if __name__ == "__main__":
    main()
