"""Fine-tune FAH-S or GEM-X EgoHand on the same native NOKOV/MANO splits."""
import argparse
import math
import random
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.hand_finetune.assets import DEFAULT_ROOT, WEIGHTS
from tools.hand_finetune.data import HandCrops
from tools.hand_finetune.fah_continuation import EpochSampler, continue_run, save_training_state
from tools.hand_finetune.fah_experiments import (
    ABLATIONS, VERSIONS, check_version_arguments, component_settings, native_validation, protocol, snapshot, split_protocol,
    stage_release, verify_training_inputs, write_json,
)


ROOT = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("model", choices=["fah", "gem-x"])
    parser.add_argument("--dataset", type=Path, help="Prepared split directory (required for --version)")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--upstream", type=Path, help="Optional developer checkout override")
    parser.add_argument("--init", type=Path, help="Native checkpoint. V0–V5 require the pinned FAH-S release bytes")
    parser.add_argument("--soma-assets", type=Path, help="GEM-X SOMA assets directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, help="Cap optimizer steps across all epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-samples", type=int, help="Fixed random subset, default entire validation split")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate --init without optimization")
    group = parser.add_argument_group("FAH-S validation and ablations")
    group.add_argument("--version", type=str.upper, choices=VERSIONS, help="V0 evaluates the initialization")
    group.add_argument("--detector-cache", type=Path, help="Fixed native-frame proposals for all validation frames and V2 training")
    group.add_argument("--combine", type=str.upper, nargs="+", choices=ABLATIONS, help="Components selected for V5")
    group.add_argument("--detector-probability", type=float, help="Fraction of V2 crops drawn from detector boxes/offsets (0.5)")
    group.add_argument("--offset-samples", type=Path, help="Use measured offsets JSON instead of exact detector boxes for V2 training")
    group.add_argument("--wrist-loss-weight", type=float, help="V3 absolute wrist loss multiplier (1.0)")
    group.add_argument("--adapter-lr", type=float, help="Default one quarter of --lr")
    group.add_argument("--eval-every", type=int, help="Validate every N optimizer steps and at the end (default each epoch)")
    group.add_argument("--save-training-state", action="store_true", help="Save AdamW and RNG state at each validation for continuation")
    group.add_argument("--resume-run", type=Path, help="Continue a state-saving FAH run into a new output directory; step limits are cumulative")
    group.add_argument("--release-weights", type=Path, help="Released FAH weights directory for the deployment evaluator")
    group.add_argument("--mano-dir", type=Path, help="Licensed MANO model directory")
    group.add_argument("--sam3d", type=Path, help="SAM-3D assets used by native FAH inference")
    args = parser.parse_args(argv)
    if args.resume_run or args.save_training_state:
        if not args.version or args.version == "V0":
            parser.error("Training state is supported for V1 through V5")
        args.save_training_state = True
    if not math.isfinite(args.lr) or min(args.batch_size, args.epochs, args.lr) <= 0 or args.workers < 0 or any(
            n is not None and n <= 0 for n in (args.max_steps, args.val_samples)):
        parser.error("Batch size, epochs, LR, step and validation limits must be positive")
    if not args.version and any(value is not None for value in (
            args.combine, args.detector_cache, args.offset_samples, args.eval_every, args.release_weights,
            args.mano_dir, args.sam3d, args.detector_probability, args.wrist_loss_weight, args.adapter_lr)):
        parser.error("Use --version with the FAH-S ablation and native validation options")
    if args.detector_probability is None:
        args.detector_probability = 0.5
    if args.wrist_loss_weight is None:
        args.wrist_loss_weight = 1.0
    if args.adapter_lr is None:
        args.adapter_lr = args.lr / 4
    if args.version:
        try:
            check_version_arguments(args)
        except ValueError as error:
            parser.error(str(error))
    if args.dataset is None:
        args.dataset = ROOT / "output/group_001_20261116_193922/dataset"
    return args


def on_gpu(batch):
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader):
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    count = 0
    for batch in loader:
        batch = on_gpu(batch)
        loss, xyz, uv_error = model.objective(batch)
        gt = batch["points"][:, :21]
        diff = (xyz - xyz[:, :1]) - (gt - gt[:, :1])
        mpjpe = 1000 * torch.linalg.vector_norm(diff, dim=-1).mean()
        px = torch.linalg.vector_norm(uv_error, dim=-1).mean()
        count += len(gt)
        totals += torch.stack((loss, mpjpe, px)).double() * len(gt)
    metrics = dict(zip(("loss", "wrist_relative_mpjpe_mm", "direct_2d_crop_px"), (totals / count).tolist()))
    if not all(map(math.isfinite, metrics.values())):
        raise FloatingPointError(f"Invalid validation metrics {metrics}")
    return {"samples": count, **metrics}


def _run(args, release_snapshot=None):
    enabled = check_version_arguments(args) if args.version else ()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for FAH/GEM-X fine-tuning")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("high")
    package = "fah" if args.model == "fah" else "gem"
    asset_root = DEFAULT_ROOT
    if args.upstream:
        # Explicit legacy checkout mode also retains its original asset layout.
        checkout = args.upstream.resolve()
        if not (checkout / package / "__init__.py").is_file():
            raise FileNotFoundError(f"Missing {package} package in {checkout}")
        sys.path.insert(0, str(checkout))
        asset_root = checkout
    spec = find_spec(package)
    if spec is None:
        raise ModuleNotFoundError(f"Install the backend with python -m pip install -r requirements/hand-{args.model}.txt")
    weights = args.release_weights or asset_root / "weights"
    checkpoint = args.init or weights / WEIGHTS[args.model][2]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}. Supply --init or download weights as described in docs/training/FAH_GEM_X_zh.md")
    if args.version:
        init_sha256 = stage_release(checkpoint, release_snapshot)
        splits = split_protocol(args.dataset, fingerprint_inputs=True)
    if args.model == "fah":
        from tools.hand_finetune.fah import FAHFineTune
        model = FAHFineTune(release_snapshot or checkpoint, wrist_loss_weight=args.wrist_loss_weight if "V3" in enabled else 0.,
                            train_adapter="V4" in enabled)
    else:
        from tools.hand_finetune.gem_x import GEMFineTune
        soma_assets = args.soma_assets or asset_root / "assets/soma"
        if not (soma_assets / "SOMAHand.npz").is_file():
            raise FileNotFoundError(f"Missing hand geometry in {soma_assets}. Supply --soma-assets or download the SOMA v0.3.3 hand assets.")
        model = GEMFineTune(checkpoint, soma_assets)
    model.cuda().eval()  # Keep the frozen feature extractor deterministic during head updates.
    augmentation = None
    if "V2" in enabled:
        from tools.hand_finetune.boxes import BoxAugmentation
        augmentation = BoxAugmentation(cache=None if args.offset_samples else args.detector_cache,
                                       offset_samples=args.offset_samples,
                                       detector_probability=args.detector_probability, seed=args.seed)
    train = HandCrops(args.dataset, "train", model.size, model.canonical_side, box_augmentation=augmentation)
    val = HandCrops(args.dataset, "val", model.size, model.canonical_side)
    if train.groups & val.groups:
        raise ValueError("A source_group and sync_index appears in both splits")
    if args.val_samples and args.val_samples < len(val):
        val = Subset(val, random.Random(args.seed).sample(range(len(val)), args.val_samples))
    loaders = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                   persistent_workers=args.workers > 0)
    val_rng = torch.Generator().manual_seed(args.seed) if args.save_training_state else None
    val_loader = DataLoader(val, generator=val_rng, **loaders)
    params = [p for p in model.parameters() if p.requires_grad]
    out = args.output
    report = {"model": args.model, "base_checkpoint": str(checkpoint.resolve()), "backend_module": spec.origin,
              "arguments": vars(args), "train_samples": len(train), "trainable_parameters": sum(p.numel() for p in params),
              "gpu": torch.cuda.get_device_name(), "epochs": []}
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    planned_steps = args.epochs * batches_per_epoch
    if args.max_steps is not None:
        planned_steps = min(planned_steps, args.max_steps)
    if args.version:
        report.update(version=args.version, components=list(enabled), component_settings=component_settings(args, enabled), status="running",
                      protocol=protocol(args, init_sha256, planned_steps, splits))
        if args.save_training_state:
            report["protocol"].update(sampling="epoch_randperm_v1", validation_schedule=[
                dict(after_step=0, interval=args.eval_every or batches_per_epoch)])
    if not args.version:
        out.mkdir(parents=True, exist_ok=False)
    if not args.resume_run:
        report["baseline"] = evaluate(model, val_loader)
        write_json(out / "metrics.json", report)
        print("initial validation", report["baseline"], flush=True)
    if args.version and not args.resume_run:
        # Every run can retain its initialization when subsequent updates regress.
        snapshot(release_snapshot, out / model.filename)
        native_validation(args, out / model.filename, 0, report)
        if args.version == "V0":
            report["status"] = "complete"
        write_json(out / "metrics.json", report)
    if args.evaluate or args.version == "V0":
        return

    sampler = EpochSampler(len(train), args.seed) if args.save_training_state else None
    train_loader = DataLoader(train, sampler=sampler, shuffle=sampler is None,
                              generator=torch.Generator().manual_seed(args.seed), **loaders)
    groups = model.optimizer_groups(args.lr, args.adapter_lr) if args.version else params
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    if args.version:
        report["optimizer_groups"] = [{"name": g["name"], "lr": g["lr"],
             "parameters": sum(p.numel() for p in g["params"])} for g in groups]
    step = continue_run(args.resume_run, out, model, optimizer, report) if args.resume_run else 0
    if args.resume_run:
        write_json(out / "metrics.json", report)
        print(f"Restored {args.version} and AdamW at step {step}, target {planned_steps}", flush=True)
    start_epoch = step // batches_per_epoch + 1
    examples, loss_sum = 0, 0.
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.epoch = epoch
            sampler.offset = (step % batches_per_epoch) * args.batch_size
        if augmentation is not None:
            train.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model.objective(on_gpu(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step+1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if step == 0 and grad_norm.item() == 0:
                raise RuntimeError("No gradient from the supervised heads")
            optimizer.step()
            step += 1
            examples += len(batch["rgb"])
            loss_sum += loss.item() * len(batch["rgb"])
            if step == 1 or step % 50 == 0:
                print(f"step {step}  loss {loss.item():.5f}  gradient norm {grad_norm.item():.5f}", flush=True)
            at_interval = step % (args.eval_every or batches_per_epoch) == 0
            if at_interval or step == planned_steps:
                result = {"epoch": epoch, "step": step, "trained_examples": examples, "train_loss": loss_sum / examples,
                          "validation": evaluate(model, val_loader)}
                tmp = out / (model.filename + ".tmp")
                model.export(tmp)
                tmp.replace(out / model.filename)
                if args.version:
                    if step == planned_steps:
                        verify_training_inputs(args.dataset, report["protocol"])
                    result["native_validation"] = native_validation(args, out / model.filename, step, report)
                    if step == planned_steps:
                        report["status"] = "complete"
                report["epochs"].append(result)
                if args.save_training_state:
                    save_training_state(out, model, optimizer, report)
                write_json(out / "metrics.json", report)
                print("saved", out / model.filename, result, flush=True)
                examples, loss_sum = 0, 0.
            if step == planned_steps:
                break
        if step == planned_steps:
            break


def main():
    args = arguments()
    if args.version is None:
        return _run(args)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(dir=args.output) as private:
            _run(args, Path(private) / "release.pt")
    except (Exception, KeyboardInterrupt) as error:
        write_json(args.output / "failure.json", {"version": args.version, "status": "failed",
                   "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
