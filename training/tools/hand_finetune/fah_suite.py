"""Queue independent FAH ablations on local GPUs and compare their validation results."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from .fah_experiments import (
    ABLATIONS, FAH_S_RELEASE_SHA256, PROTOCOL_SCHEMA, RELEASE, SELECTION_METRIC, VERSIONS,
    features, mesh_cohorts, write_json,
)


TRAIN = Path(__file__).resolve().parents[2] / "train_hand.py"


def selection_metric(protocol):
    schema, name = protocol.get("schema"), protocol.get("selection_metric")
    if schema == 3 and name == "selection_metric_mm":
        return "fitted_mesh_mpvpe_mm"
    if schema == PROTOCOL_SCHEMA and name == SELECTION_METRIC:
        return SELECTION_METRIC
    raise ValueError(f"Unknown selection metric {name!r} for FAH protocol schema {schema}")


def compare(runs):
    reference, active_settings, rows = None, {}, []
    for run in runs:
        report = json.loads((run / "metrics.json").read_text())
        if report.get("status") != "complete" or report.get("version") not in VERSIONS:
            raise ValueError(f"Incomplete or unversioned run: {run}")
        protocol = report["protocol"].copy()
        metric = selection_metric(protocol)
        if set(protocol.get("input_contents_sha256", {})) != {"train", "val"} or not protocol.get("deployment_sha256"):
            raise ValueError(f"{run} needs input and deployment identities")
        if protocol.get("init_sha256") != FAH_S_RELEASE_SHA256 or protocol.get("release") != RELEASE:
            raise ValueError(f"{run} has no verified initialization from the pinned FAH-S release")
        # Dataset location is used by in-process validation. Relocated identical inputs compare by content.
        protocol.pop("dataset", None)
        best = report["best"] or report["latest"]  # V0 may measure zero detector coverage.
        if report["version"] == "V0" and (best["step"] != 0 or report["latest"]["step"] != 0):
            raise ValueError(f"{run} claims V0 after optimizer updates")
        native = json.loads((run / best["validation_report"]).read_text())
        initial = json.loads((run / report["release_validation_report"]).read_text())
        release_cohorts = mesh_cohorts(initial)
        if report.get("release_mesh_cohorts") != release_cohorts:
            raise ValueError(f"{run} recorded release mesh cohort counts or membership disagree with step zero")
        selected_cohorts = mesh_cohorts(native)
        for arm in ("gt", "detected"):
            if release_cohorts[arm] != selected_cohorts[arm]:
                raise ValueError(f"{run} selected {arm} mesh cohort differs from step zero")
            protocol[f"{arm}_mesh_cohort"] = release_cohorts[arm]
        score = native["selection_metric_mm"]
        diagnostic = report["version"] == "V0" and report["best"] is None and score is None
        if not diagnostic and (report["best"] is None or not selected_cohorts["detected"]["scored"]
                               or score is None or not math.isfinite(score)
                               or best.get("selection_eligible") is not True or native.get("selection_eligible") is not True):
            raise ValueError(f"{run} has no eligible selected checkpoint")
        if (initial["checkpoint_sha256"] != FAH_S_RELEASE_SHA256
                or (best["step"] == 0 and native["checkpoint_sha256"] != FAH_S_RELEASE_SHA256)):
            raise ValueError(f"{run} step zero did not evaluate the pinned FAH-S release")
        if native["checkpoint_sha256"] != best["checkpoint_sha256"] or native["selection_metric_mm"] != best["selection_metric_mm"]:
            raise ValueError(f"{run} selected checkpoint or metric disagrees with its validation report")
        for scored in (native, initial):
            if ((protocol["schema"] == PROTOCOL_SCHEMA and scored.get("selection_metric") != metric)
                    or scored["selection_metric_mm"] != scored["detected"]["metrics"][metric]):
                raise ValueError(f"{run} selection score disagrees with detected {metric}")
            if (scored["split"] != "val" or scored["cache_sha256"] != protocol["cache_sha256"]
                    or scored["split_manifest_sha256"] != protocol["manifests_sha256"]["val"]
                    or scored["scored_inputs_sha256"] != protocol["input_contents_sha256"]["val"]
                    or scored["deployment_sha256"] != protocol["deployment_sha256"]):
                raise ValueError(f"{run} native validation inputs disagree with its protocol")
        settings = report["component_settings"]
        if set(settings) != set(report["components"]):
            raise ValueError(f"{run} must record settings for every active component")
        for name, values in settings.items():
            if name in active_settings and active_settings[name] != values:
                raise ValueError(f"Incompatible active {name} settings in {run}: {active_settings[name]} vs {values}")
            active_settings[name] = values
        if reference is None:
            reference = protocol
        differences = []
        for key in reference.keys() | protocol.keys():
            if key in reference and key in protocol and reference[key] == protocol[key]:
                continue
            old, new = reference.get(key), protocol.get(key)
            if isinstance(old, dict) and isinstance(new, dict):
                differences.extend(f"{key}.{k}" for k in old.keys() | new.keys()
                                   if k not in old or k not in new or old[k] != new[k])
            else:
                differences.append(key)
        if differences:
            raise ValueError(f"Incompatible runs {runs[0]} and {run}: {', '.join(sorted(differences))}")
        step = best["step"]
        crop = report["baseline"] if step == 0 else next(row["validation"] for row in report["epochs"] if row["step"] == step)
        rows.append({
            "version": report["version"], "components": report["components"], "component_settings": settings,
            "run": str(run.resolve()), "checkpoint_sha256": best["checkpoint_sha256"],
            "best_step": step if report["best"] else None, "latest_step": report["latest"]["step"],
            "latest_selection_eligible": report["latest"]["selection_eligible"],
            "latest_selection_exclusion": report["latest"].get("selection_exclusion"),
            "selection_metric_mm": best["selection_metric_mm"],
            "release_mesh_mm": {arm: initial[arm]["metrics"]["fitted_mesh_mpvpe_mm"] for arm in ("gt", "detected")},
            "gt_crop": crop,
            **{arm: {key: native[arm][key] for key in ("metrics", "counts", "coverage", "coverage_2d",
                   "metric_denominators", "joint_2d_count_by_index", "mesh_cohort_sha256")} for arm in ("gt", "detected")},
        })
    initial_spread = {}
    for arm in ("gt", "detected"):
        scores = [r["release_mesh_mm"][arm] for r in rows if r["release_mesh_mm"][arm] is not None]
        initial_spread[arm] = max(scores) - min(scores) if scores else None
    return {"protocol": reference, "release_mesh_spread_mm": initial_spread, "runs": rows}


def print_comparison(table):
    metric = selection_metric(table["protocol"])
    label = "wrist-relative MPJPE" if metric == SELECTION_METRIC else "mesh MPVPE"
    print(f"version   best/last   det mesh coverage   det full-2D coverage   GT {label} (mm)   detected {label} (mm)   run")
    for row in table["runs"]:
        step = row["best_step"] if row["best_step"] is not None else "n/a"
        score = row["selection_metric_mm"]
        detected_score = f"{score:.3f}" if score is not None else "n/a"
        gt = row["gt"]["metrics"][metric]
        gt_score = f"{gt:.3f}" if gt is not None else "n/a"
        print(f"{row['version']:<7}   {step:>4}/{row['latest_step']:<4}   "
              f"{row['detected']['coverage']:>15.1%}   {row['detected']['coverage_2d']:>20.1%}   {gt_score:>12}   "
              f"{detected_score:>18}   {row['run']}")


def gpu_slots(option, visible):
    mask = visible.split(",") if visible is not None else None
    gpus = option.split(",") if option is not None else (mask[:1] if mask is not None else ["0"])
    if (not gpus or any(not g or g == "-1" for g in gpus) or len(set(gpus)) != len(gpus)
            or (mask is not None and not set(gpus) <= set(mask))):
        raise ValueError("Choose distinct local GPU identifiers within the inherited CUDA_VISIBLE_DEVICES mask")
    return gpus


def rank_screen(table):
    rows = table["runs"]
    if (len(rows) != 5 or {r["version"] for r in rows} != set(VERSIONS[1:])
            or selection_metric(table["protocol"]) != SELECTION_METRIC):
        raise ValueError("Promotion needs five comparable V1–V5 runs selected by detected wrist-relative MPJPE")
    for row in rows:
        if row["latest_step"] != table["protocol"]["steps"] or not math.isfinite(row["selection_metric_mm"]):
            raise ValueError(f"Incomplete screening budget or missing score for {row['version']}")
    return sorted(rows, key=lambda r: (r["selection_metric_mm"], r["version"]))


def promote(out, table, jobs, target, interval):
    rows = rank_screen(table)
    winner = rows[0]
    job = jobs[winner["version"]]
    source = out / winner["version"]
    prior = json.loads((source / "metrics.json").read_text())
    start = winner["latest_step"]
    if target <= start or interval < 1:
        raise ValueError(f"Promotion needs a total budget above {start} and a positive evaluation interval")
    per_epoch = math.ceil(prior["train_samples"] / prior["arguments"]["batch_size"])
    # The screening arguments are normalized by run_suite (one output/version per job).
    command = job["command"].copy()
    command[command.index("--output") + 1] = str(out / "final")
    command.extend(["--resume-run", str(source), "--epochs", str(math.ceil(target / per_epoch)),
                    "--max-steps", str(target), "--eval-every", str(interval)])
    result = dict(status="running", selection_metric=SELECTION_METRIC, version=winner["version"],
                  ranking=[{k: row[k] for k in ("version", "selection_metric_mm", "best_step")} for row in rows],
                  start_step=start, target_step=target, gpu=job["gpu"], command=command)
    write_json(out / "promotion.json", result)
    print(f"Promoting {winner['version']} from step {start} to {target}", flush=True)
    try:
        with (out / "final.log").open("x") as log:
            subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": job["gpu"]},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        final = json.loads((out / "final/metrics.json").read_text())
        if (final["status"] != "complete" or final["latest"]["step"] != target
                or final["continued_from"]["step"] != start or final["version"] != winner["version"]):
            raise ValueError("Final run did not complete the selected version's cumulative budget")
        compare([out / "final"])
        result.update(status="complete", best=final["best"], latest=final["latest"])
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(out / "promotion.json", result)


def run_suite(args):
    versions = args.versions
    if len(set(versions)) != len(versions):
        raise ValueError("--versions contains a duplicate")
    if "V5" in versions:
        features("V5", args.combine)
    elif args.combine:
        raise ValueError("--combine needs V5 in --versions")
    common = args.train_args
    if not common or common[0] != "--":
        raise ValueError("Pass common train_hand.py options after --, including --dataset and --detector-cache")
    common = common[1:]
    if args.promote_to_steps is not None:
        if (set(versions) != set(VERSIONS[1:]) or args.promote_to_steps < 1 or args.promote_eval_every < 1
                or any(a.split("=")[0] == "--resume-run" for a in common)):
            raise ValueError("Promotion requires fresh V1–V5 runs, a positive budget and evaluation interval")
        common = [*common, "--save-training-state"]
    if any(arg.split("=")[0] in {"--version", "--output", "--combine"} for arg in common):
        raise ValueError("Put --versions, --output and --combine before the separator")
    from train_hand import arguments

    slots = gpu_slots(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    out = args.output
    jobs = {}
    for version in versions:
        options = ["fah", "--version", version, "--output", str(out / version), *common]
        if version == "V5":
            options += ["--combine", *args.combine]
        arguments(options)  # Validate every job before starting the queue.
        jobs[version] = {"command": [sys.executable, str(TRAIN), *options], "status": "queued"}
    out.mkdir(exist_ok=False, parents=True)
    write_json(out / "jobs.json", jobs)
    pending, active = deque(versions), {}
    try:
        while pending or active:
            for gpu in slots:
                if gpu in active:
                    version, child = active[gpu]
                    if child.poll() is None:
                        continue
                    jobs[version].update(status="failed" if child.returncode else "complete", returncode=child.returncode)
                    print(f"{version} {jobs[version]['status']} on GPU {gpu} (log: {out / (version + '.log')})", flush=True)
                    del active[gpu]
                    write_json(out / "jobs.json", jobs)
                if pending:
                    version = pending.popleft()
                    log_path = out / (version + ".log")
                    try:
                        with log_path.open("wb") as log:
                            child = subprocess.Popen(jobs[version]["command"], stdout=log, stderr=subprocess.STDOUT,
                                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                                     start_new_session=True)
                    except OSError as error:
                        jobs[version].update(status="failed", error=str(error))
                        write_json(out / "jobs.json", jobs)
                        continue
                    active[gpu] = version, child
                    jobs[version].update(status="running", gpu=gpu, pid=child.pid)
                    write_json(out / "jobs.json", jobs)
            if active:
                time.sleep(0.25)
    finally:
        for version, child in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            jobs[version].update(status="interrupted", returncode=child.returncode)
        for version in pending:
            jobs[version]["status"] = "skipped"
        write_json(out / "jobs.json", jobs)
    completed = [out / v for v in versions if jobs[v]["status"] == "complete"]
    if completed:
        try:
            table = compare(completed)
        except ValueError as error:
            write_json(out / "comparison_error.json", {"error": str(error), "runs": completed})
            print(error, file=sys.stderr)
            return 1
        write_json(out / "comparison.json", table)
        print_comparison(table)
        if len(completed) == len(versions) and args.promote_to_steps is not None:
            promote(out, table, jobs, args.promote_to_steps, args.promote_eval_every)
    return int(len(completed) != len(versions))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", allow_abbrev=False)
    run.add_argument("--versions", choices=VERSIONS, nargs="+", default=list(VERSIONS[:5]))
    run.add_argument("--combine", choices=ABLATIONS, nargs="+")
    run.add_argument("--output", type=Path, required=True, help="New directory for the queue and run outputs")
    run.add_argument("--gpus", help="Comma-separated CUDA GPU identifiers, default first visible GPU")
    run.add_argument("--promote-to-steps", type=int, help="Continue the best of V1–V5 to this total optimizer step")
    run.add_argument("--promote-eval-every", type=int, default=1400, help="Validation interval in the continuation phase")
    run.add_argument("train_args", nargs=argparse.REMAINDER)
    show = commands.add_parser("compare", allow_abbrev=False)
    show.add_argument("runs", type=Path, nargs="+", help="Completed versioned run directories")
    show.add_argument("--output", type=Path, help="New comparison JSON file")
    args = parser.parse_args()
    if args.action == "run":
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        return run_suite(args)
    try:
        table = compare(args.runs)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")
    if args.output:
        with args.output.open("x") as destination:
            json.dump(table, destination, indent=2, allow_nan=False, ensure_ascii=False)
            destination.write("\n")
    print_comparison(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
