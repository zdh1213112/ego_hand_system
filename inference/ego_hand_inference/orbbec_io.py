"""File, timestamp and provenance helpers for Orbbec recordings."""

import csv
import hashlib
from pathlib import Path


def unique_file(session: Path, suffix: str) -> Path:
    matches = sorted(session.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one '*{suffix}' in {session}, found {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_timestamps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"timestamp CSV is empty: {path}")
    result = []
    for row in rows:
        if row.get("timestamp_ns") not in (None, ""):
            value = int(row["timestamp_ns"])
        elif row.get("timestamp_us") not in (None, ""):
            value = int(row["timestamp_us"]) * 1000
        elif row.get("timestamp") not in (None, ""):
            # Orbbec exports timestamp in microseconds when no unit is named.
            value = int(float(row["timestamp"]) * 1000.0)
        else:
            raise RuntimeError(
                f"timestamp CSV has no timestamp_ns/timestamp_us column: {path}"
            )
        result.append(value)
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise RuntimeError(f"timestamps are not strictly increasing: {path}")
    return result


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def recording_metadata_root(source: Path) -> Path:
    """A selected part keeps calibration/metadata at its recording parent."""
    import re

    folder = source if source.is_dir() else source.parent
    return folder.parent if re.fullmatch(r"part\d+", folder.name) else folder


def discover_calibration(source: Path) -> Path:
    """Resolve the recorder's declared camera calibration, or an unambiguous sidecar."""
    import json

    folder = recording_metadata_root(source)
    metadata = sorted(folder.glob("meta_*.json"))
    if len(metadata) == 1:
        declared = (
            json.loads(metadata[0].read_text())
            .get("files", {})
            .get("calibration_camera")
        )
        if declared:
            path = (folder / declared).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError(
                    f"metadata camera calibration is missing or outside recording: {declared}"
                )
            return path
    candidates = sorted(
        set(folder.glob("*_calibration_camera.yaml"))
        | set(folder.glob("*_params_cam_*.yaml"))
    )
    if len(candidates) != 1:
        raise ValueError(
            "shared 3D requires matching four-camera calibration; supply --calibration YAML_OR_JSON_DIR"
        )
    return candidates[0].resolve()


def resolve_recording_input(
    source: Path, forced: str | None = None
) -> tuple[Path, str]:
    """Accept a selected MP4 directory, MCAP file, or unambiguous recording root."""
    source = source.expanduser().resolve()
    if forced == "mp4":
        if not source.is_dir():
            raise ValueError(f"four-MP4 recording directory does not exist: {source}")
        return source, "mp4"
    if source.is_file():
        if source.suffix.lower() != ".mcap":
            raise ValueError(
                "file input must be an Orbbec .mcap; use a directory for four MP4 streams"
            )
        return source, "mcap"
    if not source.is_dir():
        raise FileNotFoundError(source)
    mcaps = sorted(set(source.glob("*.mcap")) | set(source.glob("part*/*.mcap")))
    mp4s = list(source.glob("*_camera_*.mp4")) + list(
        source.glob("part*/*_camera_*.mp4")
    )
    if forced != "mcap" and mp4s:
        if mcaps:
            raise ValueError(
                "directory contains MP4 and MCAP; select --session or --mcap explicitly"
            )
        return source, "mp4"
    if len(mcaps) != 1:
        raise ValueError(
            f"expected one MCAP in recording/part directory, found {len(mcaps)}; select a specific part or file"
        )
    return mcaps[0].resolve(), "mcap"
