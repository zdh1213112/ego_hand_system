"""Adapt the public recording CLI to the signed per-recording label pipeline."""

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
PIPE = REPO / "label_export/pipeline"


def run(args, profile, resolve_sources):
    from export_ego_nokov_labels import raw_links

    sys.path.insert(0, str(PIPE))
    from export_glove_labels import save as write_json

    cfg = profile["label_pipeline"]
    if (
        set(profile) != {"schema_version", "label_pipeline"}
        or profile["schema_version"] != 1
        or not isinstance(cfg, dict)
        or (set(cfg) - {"overrides"})
        != {"dual_hand", "single_hand_config", "marker_placement", "observation_weights", "seed_alignment"}
        or cfg["dual_hand"] != "adaptive_wrist_v4"
    ):
        raise ValueError("Invalid signed label export profile")
    if any(v is not None for v in (args.seed_alignment,args.sample_stride,args.workers,args.source_fps)):
        raise ValueError("Set fitting parameters and seed_alignment in the optimization JSON")
    if args.joint or args.observations_only or args.marker_offset_mm or args.device != "cuda":
        raise ValueError("Signed export uses the CUDA fitting pipeline and requires a full label export")
    if (args.resume or args.publish) and not args.output:
        raise ValueError("--resume and --publish require --output")
    ego, nokov, name = resolve_sources(args.recording, args.ego, args.nokov)
    profile_path = args.optimization_config.expanduser().resolve()
    seed = (
        (args.seed_alignment or profile_path.parent / cfg["seed_alignment"]).expanduser().resolve(strict=True)
    )
    descriptor = json.loads((nokov / "asset_descriptions.json").read_text())
    hands = [
        side
        for side in ("left", "right")
        if any(key.startswith(side + "_hand") for key in descriptor["marker_sets"])
    ]
    if not hands:
        raise ValueError(f"No NOKOV hand marker set in {nokov}")
    if not (nokov / "nokov_markers.csv").is_file() or not list(nokov.glob("*skeleton*.csv")):
        raise ValueError(f"Missing NOKOV markers or skeleton in {nokov}")
    calibrations = list(ego.glob("*_calibration_camera.yaml"))
    if len(calibrations) != 1 or not (nokov / "nokov_rigid_bodies.csv").is_file():
        raise ValueError("Expected one camera calibration and a matching NOKOV rigid-body CSV")
    archives = list((ego / "part0001").glob("*.mcap"))
    if len(archives) > 1 or (ego / "part0002").exists():
        raise ValueError("Expected a single recording part")
    if not archives:
        raw_links(ego)
    out = (
        (args.output or REPO.parent / "work" / f"{name}_labels_{datetime.now():%Y%m%d_%H%M%S}")
        .expanduser()
        .resolve()
    )
    if any(out == p or out in p.parents or p in out.parents for p in (ego.resolve(), nokov.resolve(), REPO)):
        raise ValueError(f"Output overlaps input or source tree: {out}")
    placement = (profile_path.parent / cfg["marker_placement"]).resolve(strict=True)
    single_config = (profile_path.parent / cfg["single_hand_config"]).resolve(strict=True)
    mode = "adaptive_wrist_v4" if len(hands) == 2 else "single_hand_temporal"
    obs = args.wilor_observations.expanduser().resolve(strict=True) if args.wilor_observations else None
    if obs is not None and len(hands) == 1:
        raise ValueError("--wilor-observations applies to the two-hand v4 workflow")
    weights = cfg["observation_weights"]
    if set(weights) != {"checkpoint", "adapter", "detector", "model_config"}:
        raise ValueError("Expected detector, checkpoint, adapter and model_config")
    from label_export_settings import provenance, workspace_path
    weights = {k: workspace_path(REPO, v).resolve() for k, v in weights.items()}
    if mode == "adaptive_wrist_v4":
        (REPO / "models/mano/MANO_RIGHT.pkl").resolve(strict=True)
        if not obs:
            for path in weights.values():
                path.resolve(strict=True)
    settings = dict(
        mode=mode,
        hands=hands,
        ego=str(ego.resolve()),
        nokov=str(nokov.resolve()),
        seed_alignment=str(seed),
        optimization_config=str(profile_path),
        optimization_config_content=profile,
        observation_source=str(obs) if obs else None,
        single_hand_config=json.loads(single_config.read_text()),
    )
    actual = provenance(REPO,cfg,name,hands,weights,obs,seed,placement,single_config)
    settings.update(actual)
    runtime = actual["runtime_parameters"]
    for key, diff in actual["signed_parameter_differences"].items():
        print(f"[config] {key}: {diff['signed']} -> {diff['effective']}",flush=True)
    if args.resume or args.publish:
        if (
            not (out / "settings.json").is_file()
            or json.loads((out / "settings.json").read_text()) != settings
        ):
            raise ValueError("--resume requires matching recording, method, observations and settings")
    elif out.exists():
        raise FileExistsError(f"Use a new --output, or --resume for an existing run: {out}")
    print(f"[label-export] {', '.join(hands)} | {mode} | {out}", flush=True)
    fit_root = out / "fitting"
    legacy = fit_root / "legacy" / name
    result = out / "optimized"
    if mode == "adaptive_wrist_v4":
        print(
            "[label-export] same-exposure WiLoR → signed steps 01–12 → v4 training labels and geometric validation",
            flush=True,
        )
    else:
        print(
            "[label-export] single-hand NOKOV + SDK temporal fit; output uses the native temporal label format",
            flush=True,
        )
    if args.dry_run:
        return
    if args.publish:
        if mode != "adaptive_wrist_v4":
            raise ValueError("Publication requires the two-hand v4 review")
        subprocess.run(
            [sys.executable, "-Bu", str(PIPE / "v4/export.py"), "--settings", str(out/"settings.json"), str(legacy), str(result), "--publish"],
            check=True,
        )
        print("[published]", out / "published", flush=True)
        return
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / "settings.json", settings)
        pair = out / "raw_input"
        pair.mkdir(exist_ok=True)
        for dest, source in ((pair / ego.name, ego), (pair / "session_nokov" / "nokov", nokov)):
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                dest.symlink_to(source.resolve(), target_is_directory=True)
            if not dest.samefile(source):
                raise ValueError(f"Recording link changed: {dest}")
        if mode == "single_hand_temporal":
            if archives:
                converted = out / "converted"
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(PIPE / "stages/convert.py"),
                        "--group",
                        str(pair),
                        "--output",
                        str(converted),
                    ],
                    check=True,
                )
                pair = converted
            if cfg.get("overrides",{}).get("single_hand"):
                effective_single = dict(actual["single_hand_parameters"])
                effective_single["placement"] = str((single_config.parent/effective_single["placement"]).resolve(strict=True))
                single_config = out/"single_hand_effective.json"
                write_json(single_config,dict(schema_version=1,joint_refit=effective_single))
            inner = out / "native"
            cmd = [
                sys.executable,
                "-B",
                str(REPO / "label_export/export_ego_nokov_labels.py"),
                "--recording",
                str(pair),
                "--output",
                str(inner),
                "--seed-alignment",
                str(seed),
                "--optimization-config",
                str(single_config),
                "--workers",
                str(runtime["single_workers"]),
                "--source-fps",
                str(runtime["single_fps"]),
                "--sample-stride",
                str(runtime["single_baseline_stride"]),
            ]
            if (inner / "settings.json").is_file():
                cmd.append("--resume")
            subprocess.run(cmd, check=True)
            if not result.exists():
                result.symlink_to("native/optimized", target_is_directory=True)
            print("[training labels]", result / "dense/fit/final/export/merged", flush=True)
            return
        if obs is None:
            from recordings.inference import archive

            inference = out / "wilor"
            cmd = [
                "bash",
                str(REPO / "inference/run.sh"),
                "--input",
                str(ego / "part0001"),
                "--calibration",
                str(calibrations[0]),
                "--output",
                str(inference),
                "--batch-size",
                str(runtime["observation_batch_size"]),
                "--frame-batch-size",
                str(runtime["observation_frame_batch_size"]),
                "--preprocess-workers",
                str(runtime["observation_preprocess_workers"]),
            ]
            for k, v in weights.items():
                cmd.extend(["--" + k.replace("_", "-"), str(v)])
            subprocess.run(cmd, check=True, env={**os.environ, "EGO_HAND_PYTHON": sys.executable})
            obs = out / "native_wilor.npz"
            if not obs.exists():
                archive(inference, obs)
        config = dict(
            mode="from_raw",
            fit_method="legacy",
            wrist_policy="adaptive_measured",
            id=name,
            raw_group=str(pair),
            output_root=str(fit_root),
            initial_alignment=str(seed),
            camera_calibration=str(calibrations[0].resolve()),
            marker_placement=str(placement),
            mano_directory=str(REPO / "models/mano"),
            native_wilor=str(obs),
            verify_exposure_clock=True,
        )
        if cfg.get("overrides",{}).get("parameters"):
            config["parameters"] = {k:actual["effective_parameters"][k] for k in cfg["overrides"]["parameters"] if k!="v4"}
        resolved = out / "recording.yaml"
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
        cmd = [sys.executable, "-Bu", str(PIPE / "export_glove_labels.py"), "--config", str(resolved)]
        if fit_root.exists():
            cmd.append("--resume")
        subprocess.run(cmd, check=True)
        subprocess.run(
            [sys.executable, "-Bu", str(PIPE / "v4/export.py"), "--settings", str(out/"settings.json"), str(legacy), str(result)], check=True
        )
        print("[training labels]", result, flush=True)
        print("[final video]", result / "review/camera0_full.mp4", flush=True)
