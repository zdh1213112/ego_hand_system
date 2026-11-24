"""Versioned FAH experiments share a split, initialization and evaluation protocol."""
import hashlib
from importlib.util import find_spec
import json
import math
import os
import shutil
import time
from pathlib import Path

from .assets import WEIGHTS

VERSIONS = ("V0", "V1", "V2", "V3", "V4", "V5")
ABLATIONS = ("V2", "V3", "V4")
# LFS object ID at the pinned WEIGHTS revision, also published in FAH weights/SHA256SUMS.
FAH_S_RELEASE_SHA256 = "f469c80356391c2993c58b4421ef3ee81c36549af932cffed1824a038147484c"
RELEASE = dict(zip(("repository", "revision", "file"), WEIGHTS["fah"]))
PROTOCOL_SCHEMA = 4
SELECTION_METRIC = "wrist_relative_mpjpe_mm"
# Each independent training/evaluation process reuses immutable input digests across validation intervals.
_FILE_DIGESTS = {}


def features(version, combine):
    if version != "V5":
        if combine:
            raise ValueError("--combine applies to V5 only")
        return (version,) if version in ABLATIONS else ()
    if not combine or len(set(combine)) != len(combine) or not set(combine) <= set(ABLATIONS):
        raise ValueError("V5 needs an explicit --combine of distinct V2, V3 and/or V4")
    return tuple(sorted(combine))


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def digest(path, *, copy_to=None):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
            if copy_to is not None:
                copy_to.write(block)
    return sha.hexdigest()


def content_digest(path):
    def identity():
        st = path.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    key = identity()
    if key in _FILE_DIGESTS:
        return _FILE_DIGESTS[key]
    measured = digest(path)
    if identity() != key:
        raise ValueError(f"Input changed while computing its SHA256: {path}")
    # Same-sized writes in one filesystem clock tick can share mtime and ctime.
    # Only retain digests for files whose last change predates that window.
    if max(key[-2:]) < time.time_ns() - 2_000_000_000:
        _FILE_DIGESTS[key] = measured
    return measured


def stage_release(checkpoint, destination):
    """Hash the exact bytes the trainer will load in a single copy to its private snapshot."""
    with destination.open("xb") as staged:
        actual = digest(checkpoint, copy_to=staged)
    if actual != FAH_S_RELEASE_SHA256:
        raise ValueError(f"V0–V5 require the pinned FAH-S release {FAH_S_RELEASE_SHA256}; "
                         f"{checkpoint} has SHA256 {actual}. Use unversioned --init or fah_evaluation for fine-tunes.")
    return actual


def split_protocol(root, *, fingerprint_inputs=False):
    """Check recording identity and synchronization before any optimization or scoring."""
    from .boxes import FrameKey
    from .splits import SplitAssetGuard, recording_timeline

    root = Path(root)
    hash_splits = {"train", "val"} if fingerprint_inputs is True else set(fingerprint_inputs or ())
    if not hash_splits <= {"train", "val", "test"}:
        raise ValueError(f"Unknown split to fingerprint: {hash_splits}")
    fingerprints, input_hashes, groups, recordings = {}, {}, {}, {}
    single_recording, timings, scanned = None, [], {}
    guard = SplitAssetGuard(root)
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        if split == "test" and not path.exists():
            continue
        groups[split] = set()
        sha, inputs = hashlib.sha256(), hashlib.sha256()
        with path.open("rb") as source:
            for number, line in enumerate(source, 1):
                sha.update(line)
                row = json.loads(line)
                try:
                    frame = FrameKey.from_row(row)
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                guard.add(split, row)
                key = (frame.recording_id, frame.sync_index)
                if single_recording is None:
                    single_recording = frame.recording_id
                if frame.recording_id != single_recording:
                    timings = None  # Whole-recording splits need no timeline for all subsequent rows.
                if timings is not None:
                    timings.append({k: row[k] for k in ("camera", "sync_index", "source_frame_index", "reference_timestamp_ns")
                                    if k in row})
                groups[split].add(key)
                if split in hash_splits:
                    # Commit to each row's ordered image and hand label. Read shared image bytes only once.
                    for field in ("image", "label"):
                        asset = root / row[field]
                        if asset not in scanned:
                            scanned[asset] = bytes.fromhex(content_digest(asset))
                        inputs.update(scanned[asset])
        if not groups[split]:
            raise ValueError(f"Empty split: {path}")
        recordings[split] = {r for r, _ in groups[split]}
        fingerprints[split] = sha.hexdigest()
        if split in hash_splits:
            input_hashes[split] = inputs.hexdigest()
    seen = set()
    for split, keys in groups.items():
        if seen & keys:
            raise ValueError(f"{split} shares a (recording_id, sync_index) with an earlier split")
        seen.update(keys)
    if len(set.union(*recordings.values())) == 1:
        strategy = "contiguous_time"
        time_field, times = recording_timeline(timings)
        previous = None
        for split, keys in groups.items():
            timeline = [times[sync] for _, sync in keys]
            low, high = min(timeline), max(timeline)
            if previous is not None and low <= previous:
                raise ValueError(f"Single-recording {split} must follow the preceding contiguous-time split")
            previous = high
    else:
        strategy = "cross_recording"
        time_field = None
        if sum(map(len, recordings.values())) != len(set.union(*recordings.values())):
            raise ValueError("Cross-recording holdout needs complete recordings assigned to one split each")
    summary = root / "splits.json"
    if summary.is_file():
        declared = json.loads(summary.read_text())
        if declared.get("strategy") != strategy or declared.get("time_field") != time_field:
            raise ValueError(f"Holdout strategy in {summary} disagrees with the manifests")
    result = {"holdout": strategy, "time_field": time_field, "manifests_sha256": fingerprints}
    if hash_splits:
        result["input_contents_sha256"] = input_hashes
    return result


def check_version_arguments(args):
    if args.model != "fah" or args.dataset is None or args.detector_cache is None:
        raise ValueError("--version requires fah, an explicit --dataset and --detector-cache")
    if args.val_samples is not None:
        raise ValueError("Versioned runs evaluate the entire val split; omit --val-samples")
    if args.evaluate and args.version != "V0":
        raise ValueError("Use --version V0 for an evaluation-only run")
    if args.eval_every is not None and args.eval_every <= 0:
        raise ValueError("--eval-every must be a positive number of optimizer steps")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be an integer from 0 through 2**32 - 1")
    if not all(math.isfinite(value) for value in (args.wrist_loss_weight, args.adapter_lr, args.detector_probability)):
        raise ValueError("Ablation weights, learning rates, and probabilities must be finite")
    enabled = features(args.version, args.combine)
    if "V3" in enabled and not (0 < args.wrist_loss_weight < math.inf):
        raise ValueError("V3 needs a finite positive --wrist-loss-weight")
    if "V4" in enabled and not (0 < args.adapter_lr < args.lr):
        raise ValueError("V4 needs 0 < --adapter-lr < --lr")
    if "V2" in enabled and not (0 < args.detector_probability < 1):
        raise ValueError("V2 needs 0 < --detector-probability < 1 for a GT/detector mixture")
    return enabled


def verify_training_inputs(dataset, run_protocol):
    """A completed run must still identify the training images and labels it started with."""
    current = split_protocol(dataset, fingerprint_inputs=("train",))
    if (current["manifests_sha256"] != run_protocol["manifests_sha256"] or
            current["input_contents_sha256"]["train"] != run_protocol["input_contents_sha256"]["train"]):
        raise ValueError("Training images, labels, or manifests changed during the run")


def component_settings(args, enabled):
    settings = {}
    if "V2" in enabled:
        settings["V2"] = {"detector_probability": args.detector_probability, "min_iou": 0.5,
                          "offset_samples_sha256": digest(args.offset_samples) if args.offset_samples else None}
    if "V3" in enabled:
        settings["V3"] = {"wrist_loss_weight": args.wrist_loss_weight}
    if "V4" in enabled:
        settings["V4"] = {"adapter_lr": args.adapter_lr}
    return settings


def protocol(args, init_sha256, planned_steps, splits):
    """Identities for inputs and controls that must agree in a side-by-side result."""
    from importlib.metadata import version
    import cv2

    root = args.dataset.resolve()
    package = Path(__file__).resolve().parent
    modules = ("assets.py", "boxes.py", "crop_geometry.py", "data.py", "fah.py", "fah_views.py", "splits.py",
               "fah_evaluation.py", "fah_experiments.py")
    sources = [package.parent.parent / "train_hand.py", *(package / name for name in modules)]
    if args.save_training_state:
        sources.append(package / "fah_continuation.py")

    return {
        "schema": PROTOCOL_SCHEMA, "dataset": str(root), **splits,
        "code_sha256": {path.name: digest(path) for path in sources},
        "runtime": {"opencv": cv2.__version__, **{name: version(name) for name in
                    ("numpy", "torch", "torchvision", "timm", "scipy", "smplx")}},
        "init_sha256": init_sha256, "release": RELEASE, "cache_sha256": digest(args.detector_cache),
        "eval_split": "val", "seed": args.seed,
        "selection_metric": SELECTION_METRIC, "batch_size": args.batch_size,
        "workers": args.workers, "head_lr": args.lr, "eval_every": args.eval_every,
        "steps": planned_steps,
    }


def deployment_fingerprints(result):
    """Identify the resolved geometry, fitter, and FAH engine once, at release validation."""
    engine = Path(result["backend"]).parent
    sam, mano = Path(result["sam3d"]), Path(result["mano_dir"])
    config = Path(result["release_weights"]) / "fah-s/config.json"
    paths = {"config": config, "mesh_fitter": config.parent / json.loads(config.read_text())["mesh_fitter"],
             "MANO_LEFT": mano / "MANO_LEFT.pkl", "MANO_RIGHT": mano / "MANO_RIGHT.pkl",
             "sam_config": sam / "model_config.yaml", "mhr_model": sam / "assets/mhr_model.pt"}
    for name in ("bench.methods.fah", "bench.common", "train.mesh_fitter"):
        spec = find_spec(name)
        if spec is None or not spec.origin:
            raise ModuleNotFoundError(f"Missing FAH deployment module {name}")
        paths[name] = Path(spec.origin)
    for p in [*engine.rglob("*.py"), *(engine / "assets").glob("*.npz")]:
        paths["fah/" + p.relative_to(engine).as_posix()] = p
    return {name: content_digest(path) for name, path in sorted(paths.items())}


def snapshot(source, destination):
    """Both run-owned names can share an inode because latest is replaced atomically."""
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copyfile(source, tmp)
    tmp.replace(destination)


def mesh_cohorts(result):
    """Count and identify the labeled hands contributing to each arm's mesh mean."""
    cohorts = {}
    for arm in ("gt", "detected"):
        block = result[arm]
        labeled, scored = (block["counts"][key] for key in ("labeled", "scored"))
        identity = block.get("mesh_cohort_sha256")
        if (type(labeled) is not int or type(scored) is not int or not 0 <= scored <= labeled
                or not isinstance(identity, str) or len(identity) != 64
                or any(c not in "0123456789abcdef" for c in identity)):
            raise ValueError(f"Invalid {arm} mesh cohort counts or SHA-256 in native validation")
        cohorts[arm] = {"labeled": labeled, "scored": scored, "mesh_cohort_sha256": identity}
    return cohorts


def native_validation(args, checkpoint, step, report):
    from .fah_evaluation import evaluate_checkpoint

    result = evaluate_checkpoint(checkpoint=checkpoint, dataset=args.dataset, split="val",
                                 detector_cache=args.detector_cache, upstream=args.upstream,
                                 release_weights=args.release_weights, mano_dir=args.mano_dir,
                                 sam3d=args.sam3d, batch_size=args.batch_size, device="cuda",
                                 validated_protocol=report.get("protocol"))
    if step == 0 and "protocol" in report:
        report["protocol"]["deployment_sha256"] = result["deployment_sha256"]
    path = args.output / f"val_step_{step:08d}.json"
    cohorts = mesh_cohorts(result)
    if step == 0:
        report["release_validation_report"] = path.name
        report["release_mesh_cohorts"] = cohorts
    changed = [arm for arm in cohorts if cohorts[arm] != report["release_mesh_cohorts"][arm]]
    score = result["selection_metric_mm"]
    if score is not None and not math.isfinite(score):
        raise FloatingPointError(f"Nonfinite {SELECTION_METRIC} at step {step}")
    reason = None
    if changed:
        reason = f"Mesh cohort differs from step zero in {', '.join(changed)}"
    elif score is None or not cohorts["detected"]["scored"]:
        reason = "No wrist-relative MPJPE on scored detector hands"
    entry = {"step": step, "selection_metric_mm": score, "validation_report": path.name,
             "checkpoint_sha256": result["checkpoint_sha256"], "selection_eligible": reason is None,
             "selection_exclusion": reason}
    write_json(path, {**result, "selection_eligible": reason is None, "selection_exclusion": reason})
    report["latest"] = entry
    if reason:
        if step == 0:
            report["best"] = None
            if args.version != "V0":
                raise ValueError(f"Release validation has no finite native selection metric. Inspect {path}")
        return entry
    if report.get("best") is None or score < report["best"]["selection_metric_mm"]:
        snapshot(checkpoint, args.output / "best_model.pt")
        report["best"] = entry
    return entry
