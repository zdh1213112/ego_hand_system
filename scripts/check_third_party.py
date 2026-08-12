#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ego_hand_system external dependencies.")
    parser.add_argument("--require-mano", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--require-basalt", action="store_true")
    parser.add_argument("--require-wilor", action="store_true")
    return parser.parse_args()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project / "third_party" / "manifest.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    print(f"External dependency audit\nProject: {project}\n")
    for name, entry in manifest["public_submodules"].items():
        path = project / entry["path"]
        revision = git_revision(path) if path.is_dir() else None
        ready = revision == entry["revision"] and (project / entry["license_file"]).is_file()
        print(f"[{'OK' if ready else 'MISSING'}] source {name}: {entry['path']}")
        if revision and revision != entry["revision"]:
            print(f"  expected {entry['revision']}, found {revision}")
        if not ready:
            print("  run: git submodule update --init --recursive")
            failures.append(f"source:{name}")

    required = set()
    required.add("mediapipe_hand_landmarker")
    if args.require_mano:
        required.add("mano_models")
    if args.require_live:
        required.add("orbbec_sdk_linux_x86_64")
    if args.require_basalt:
        required.add("basalt_runtime_linux_x86_64")
    if args.require_wilor:
        required.add("wilor_models")

    print()
    for name, entry in manifest["local_assets"].items():
        missing = [p for p in entry["required_files"] if not (project / p).is_file()]
        print(f"[{'OK' if not missing else 'MISSING'}] local asset {name}")
        if missing:
            for relative in missing:
                print(f"  - {relative}")
            print(f"  {entry['install']}")
            if name in required:
                failures.append(f"asset:{name}")
        elif "sha256" in entry and "checksum_file" in entry:
            import hashlib
            path = project / entry["checksum_file"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != entry["sha256"]:
                print(f"  checksum mismatch: expected {entry['sha256']}, found {actual}")
                failures.append(f"checksum:{name}")
        elif "tested_sha256" in entry:
            import hashlib
            for filename, expected in entry["tested_sha256"].items():
                path = project / "models" / "mano" / filename
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    print(f"  {filename} differs from the project-tested MANO v1.2 asset")
                    print(f"  expected {expected}, found {actual}")
                    if name in required:
                        failures.append(f"checksum:{name}:{filename}")
        elif name == "basalt_runtime_linux_x86_64":
            receipt = (project / "third_party" / "basalt_runtime" / "VERSION").read_text()
            expected = entry["sha256"]
            if f"sha256={expected}" not in receipt:
                print("  VERSION receipt does not contain the verified archive SHA-256")
                if name in required:
                    failures.append(f"receipt:{name}")

    if failures:
        print("\nRequired dependencies are incomplete: " + ", ".join(failures))
        return 1
    print("\nDependency audit completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
