#!/usr/bin/env python3
"""Report mean per-frame processing and mean WiLoR forward/module times.

Extra arguments are forwarded to run_multiview_wilor_experiment.sh. Defaults
use the 20261009 EGO4 recording, detector_1009.pt and the local wilor conda env.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from wilor_timing_report import build_averages, write_simple_report


ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "recordings/20261009/EGO_Pro_AZLR86000F7_20261009_171149"


def record_stage(directory, name, command):
    started = time.perf_counter()
    status = 130
    try:
        status = subprocess.call(command)  # Inherit stdin, including run_config heredoc.
        return status
    finally:
        row = {"stage": "run_config" if name == "-" else Path(name).stem,
               "wall_seconds": time.perf_counter() - started, "exit_code": status}
        with (Path(directory) / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


def conda_proxy(arguments):
    """Intercept only the experiment's conda command, without editing it."""
    real_conda = os.environ["EGO_TIMING_REAL_CONDA"]
    command = [real_conda, *arguments]
    if arguments[:1] == ["run"]:
        for index, value in enumerate(arguments[:-1]):
            if value == "python":
                target = arguments[index + 1]
                if Path(target).name == "wilor_multiview_inference.py":
                    # Insert our bootstrap before the original target argument.
                    command.insert(index + 2, str(ROOT / "scripts/wilor_timing_bootstrap.py"))
                return record_stage(os.environ["EGO_WILOR_TIMING_DIR"], target, command)
    return subprocess.call(command)


def write_report(directory, elapsed, returncode, command):
    stages_path = directory / "stages.jsonl"
    stages = [json.loads(line) for line in stages_path.read_text().splitlines()] if stages_path.exists() else []
    component_path = directory / "components.json"
    components = json.loads(component_path.read_text()) if component_path.exists() else None
    averages = build_averages(directory, stages, components, command)
    report = {
        "schema_version": 2, "command": command, "exit_code": returncode,
        "total_wall_seconds": elapsed, "stages": stages, **averages,
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_simple_report(directory, averages, elapsed, returncode)
    return report


def main():
    # Internal entry point used by the existing shell runner.
    if len(sys.argv) > 1 and sys.argv[1] == "--record-stage":
        directory, name = sys.argv[2:4]
        if sys.argv[4] != "--":
            raise SystemExit("expected -- before stage command")
        return record_stage(directory, name, sys.argv[5:])
    if len(sys.argv) > 1 and sys.argv[1] == "--conda-proxy":
        return conda_proxy(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--timing-dir", type=Path, help="new directory for reports (default OUTPUT/timing)")
    parser.add_argument("--warmup-batches", type=int, default=1,
                        help="warmup count for raw diagnostics; main averages include all successful calls")
    parser.add_argument("--summarize", type=Path,
                        help="rebuild simple reports from an existing timing directory; do not rerun inference")
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be non-negative")

    if args.summarize is not None:
        directory = args.summarize.resolve()
        previous = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        write_report(directory, previous["total_wall_seconds"], previous["exit_code"], previous["command"])
        print(f"[timing] simplified report: {directory / 'report.md'}")
        return 0

    def option_value(name):
        if name in forwarded:
            index = forwarded.index(name)
            if index + 1 < len(forwarded):
                return forwarded[index + 1]
        return None

    defaults = {
        "--output": str(ROOT / "output" / ("wilor_timing_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))),
        "--detector": str(ROOT / "models/wilor/detector_1009.pt"),
        "--conda-env": "wilor", "--compile-backbone": "0",
    }
    if not any(flag in forwarded for flag in ("--session", "--mcap", "--experimental-session-3d")):
        defaults["--session"] = str(SESSION)
    for flag, value in defaults.items():
        if flag not in forwarded:
            forwarded.extend([flag, value])
    output = Path(option_value("--output")).resolve()
    if (output / "wilor_multiview/summary.json").exists():
        parser.error("WiLoR predictions already cached; use a new --output directory to measure inference")
    directory = (args.timing_dir or output / "timing").resolve()
    real_conda = shutil.which("conda")
    if real_conda is None:
        parser.error("conda executable is not on PATH")
    directory.mkdir(parents=True, exist_ok=False)
    shim_directory = directory / "launcher"
    shim_directory.mkdir()
    shim = shim_directory / "conda"
    shim.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
        + shlex.quote(str(Path(__file__).resolve())) + ' --conda-proxy "$@"\n')
    shim.chmod(0o755)
    env = dict(os.environ, EGO_WILOR_TIMING_DIR=str(directory),
               EGO_WILOR_TIMING_WARMUP=str(args.warmup_batches),
               EGO_TIMING_REAL_CONDA=real_conda,
               PATH=str(shim_directory) + os.pathsep + os.environ.get("PATH", ""))
    command = ["bash", str(ROOT / "scripts/run_multiview_wilor_experiment.sh"), *forwarded]
    (directory / "invocation.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"[timing] reports: {directory}", flush=True)
    started = time.perf_counter()
    returncode = 130
    try:
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
    finally:
        write_report(directory, time.perf_counter() - started, returncode, command)
        print(f"\n[timing] report: {directory / 'report.md'}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
