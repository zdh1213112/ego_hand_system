#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

EXPECTED = {
    "MANO_LEFT.pkl": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "MANO_RIGHT.pkl": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install licensed MANO_LEFT.pkl and MANO_RIGHT.pkl from a user-provided directory or archive."
    )
    parser.add_argument("--source", required=True, type=Path,
                        help="directory, ZIP, or tar archive downloaded by the user")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {filename}, found {len(matches)} under {root}")
    return matches[0]


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    project = Path(__file__).resolve().parents[1]
    destination = project / "models" / "mano"
    temporary = None
    try:
        if source.is_dir():
            search_root = source
        else:
            temporary = tempfile.TemporaryDirectory(prefix="mano_models_")
            search_root = Path(temporary.name)
            if zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(search_root)
            elif tarfile.is_tarfile(source):
                with tarfile.open(source) as archive:
                    archive.extractall(search_root, filter="data")
            else:
                raise RuntimeError("source is not a directory, ZIP, or supported tar archive")

        found = {name: locate(search_root, name) for name in EXPECTED}
        destination.mkdir(parents=True, exist_ok=True)
        for name, path in found.items():
            target = destination / name
            if target.exists() and not args.replace:
                if sha256(target) == sha256(path):
                    print(f"already installed and identical: {target}")
                    continue
                raise FileExistsError(f"target already exists: {target}; use --replace deliberately")
            if target.exists() and target.samefile(path):
                print(f"already installed: {target}")
                continue
            shutil.copy2(path, target)
            print(f"installed {target} ({target.stat().st_size} bytes, sha256={sha256(target)})")

        for name, expected in EXPECTED.items():
            actual = sha256(destination / name)
            if actual != expected:
                print(f"warning: {name} differs from the project-tested MANO v1.2 asset")
                print(f"  tested: {expected}")
                print(f"  actual: {actual}")
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
