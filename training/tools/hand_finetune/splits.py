"""Prepare FAH manifests by whole recordings, or contiguous time in one recording."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import stat
import tempfile
from urllib.parse import quote

from .boxes import FrameKey


_SOURCE_PATH_FIELDS = ("source_optimized_fit", "source_fit", "source_dataset")


def recording_source_paths(row):
    """Full source export/fit paths can identify a recording across local re-exports."""
    for field in _SOURCE_PATH_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and os.path.isabs(value):
            yield field, os.path.normpath(value)


class SplitAssetGuard:
    """Track shared native files and recording provenance across all supplied splits."""
    def __init__(self, root):
        self.root = Path(root)
        self.images = {}
        self.labels = set()
        self.provenance = {}

    def _file_id(self, name):
        path = self.root / name
        info = path.stat()  # Follows symlinks; hardlinks share device and inode.
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Expected a regular asset file: {path}")
        return info.st_dev, info.st_ino

    def add(self, split, row):
        key = FrameKey.from_row(row)
        owner = (split, key)
        image = self._file_id(key.image)
        if self.images.setdefault(image, owner) != owner:
            raise ValueError(f"Shared source image has conflicting split or frame identity: {key.image}")
        label = self._file_id(row["label"])
        if label in self.labels:
            raise ValueError(f"Duplicate source hand label: {row['label']}")
        self.labels.add(label)

        # Numeric dataset IDs and names like processed__both can recur in unrelated exports.
        source_id = row.get("source_recording_id", key.recording_id)
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Expected a non-empty source_recording_id")
        for identity in [("source_recording_id", source_id), *recording_source_paths(row)]:
            if self.provenance.setdefault(identity, key.recording_id) != key.recording_id:
                raise ValueError(f"Shared recording provenance {identity} assigned to different recording IDs")


def _export_rows(root, recording_id, alias):
    identities = defaultdict(set)
    rows = []
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    conventions = {k: summary[k] for k in ("geometry_space", "image_space", "units", "length_unit",
                                           "geometry_unit", "units_3d") if k in summary}
    if (conventions.get("geometry_space", "physical_side") != "physical_side"
            or conventions.get("image_space", "native_raw") != "native_raw"):
        raise ValueError(f"Expected physical-side geometry and native KB image space in {summary_path}")
    with (root / "index.jsonl").open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                for field in ("recording_id", "source_recording_id", *_SOURCE_PATH_FIELDS):
                    if summary.get(field) is not None:
                        row.setdefault(field, summary[field])
                for field, value in conventions.items():
                    if row.setdefault(field, value) != value:
                        raise ValueError(f"{field} disagrees with export summary")
                for field in ("recording_id", "source_recording_id", "source_dataset_id"):
                    if row.get(field) is not None:
                        identities[field].add(str(row[field]))
                for field, path in recording_source_paths(row):
                    identities[field].add(path)
                if row.get("recording_id") is not None:
                    row.setdefault("source_recording_id", row["recording_id"])
                row["recording_id"] = recording_id
                FrameKey.from_row(row)
                if row.get("stored_image_horizontally_flipped", False):
                    raise ValueError("Expected physical native images")
                for field in ("image", "label"):
                    path = Path(row[field])
                    source_path = root / path
                    if not source_path.is_file():
                        raise ValueError(f"Missing {field} {source_path}")
                    # Use the recording alias for ordinary assets. Preserve external absolute references.
                    if source_path.is_relative_to(root):
                        row[field] = str(alias / source_path.relative_to(root))
                    else:
                        row[field] = str(source_path)
                rows.append(row)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{root}/index.jsonl:{number}: {exc}") from exc
    if not rows or any(len(values) > 1 for values in identities.values()):
        raise ValueError(f"Expected a non-empty export for exactly one real recording in {root}")
    return rows


def recording_timeline(rows):
    """Return (clock field, time by sync_index) for a complete single recording.

    Prefer reference_timestamp_ns, shared across all cameras. Without a reference
    clock, multiple cameras use sync_index; a single camera can use its own
    source_frame_index. Each sync needs at least one observation on the chosen
    clock and all declared observations must agree. Equal times stay together.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot order an empty recording")
    field = "sync_index"
    if any(row.get("reference_timestamp_ns") is not None for row in rows):
        field = "reference_timestamp_ns"
    elif (len({row["camera"] for row in rows}) == 1
          and any(row.get("source_frame_index") is not None for row in rows)):
        field = "source_frame_index"

    times = {}
    syncs = set()
    for row in rows:
        sync = row["sync_index"]
        if type(sync) is not int or sync < 0:
            raise ValueError("sync_index must be a non-negative integer")
        syncs.add(sync)
        time = row.get(field)
        if time is None:
            continue
        if type(time) is not int or time < 0:
            raise ValueError(f"Expected non-negative integer {field} for sync_index {sync}")
        if times.setdefault(sync, time) != time:
            raise ValueError(f"Conflicting {field} for sync_index {sync}")
    missing = syncs.difference(times)
    if missing:
        raise ValueError(f"Missing {field} for sync_index {min(missing)}")
    return field, times


def _time_partition(rows, times, val_fraction, test_fraction):
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Holdout fractions must leave non-empty train and val fractions")
    # Assign whole time values, including sync groups sharing the same timestamp.
    timeline = sorted(set(times.values()))
    n = len(timeline)
    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * test_fraction)) if test_fraction else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Too few distinct sync times for the requested holdouts")
    assignment = dict(zip(timeline, ["train"] * n_train + ["val"] * n_val + ["test"] * n_test, strict=True))
    by_split = defaultdict(list)
    for row in rows:
        by_split[assignment[times[row["sync_index"]]]].append(row)
    return by_split


def prepare_splits(output, recordings, *, by_time=False, val_fraction=0.15, test_fraction=0.15):
    """recordings is a sequence of (split, true recording ID, export root) triples."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    splits = {split for split, _, _ in recordings}
    if by_time and (len(recordings) != 1 or splits != {"time"}):
        raise ValueError("Contiguous time holdout requires one recording")
    if not by_time and (not {"train", "val"} <= splits or splits - {"train", "val", "test"}):
        raise ValueError("Assign whole recordings to train, val and optionally test")
    roots = [Path(path).resolve(strict=True) for _, _, path in recordings]
    ids = [name for _, name, _ in recordings]
    if (any(not isinstance(name, str) or not name.strip() for name in ids)
            or len(set(roots)) != len(roots) or len(set(ids)) != len(ids)):
        raise ValueError("Give each recording one unique ID and one distinct export root")
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("Choose a new output directory outside the source exports")

    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, strategy="contiguous_time" if by_time else "cross_recording",
                  sources=dict(zip(ids, map(str, roots))), splits={})
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        (stage / "recordings").mkdir(parents=True)
        by_split = defaultdict(list)
        for (split, recording_id, _), root in zip(recordings, roots):
            alias = Path("recordings") / ("rec-" + quote(recording_id, safe=""))
            (stage / alias).symlink_to(root, target_is_directory=True)
            rows = _export_rows(root, recording_id, alias)
            if by_time:
                report["time_field"], time_by_sync = recording_timeline(rows)
                by_split = _time_partition(rows, time_by_sync, val_fraction, test_fraction)
            else:
                by_split[split].extend(rows)

        guard = SplitAssetGuard(stage)
        for split, rows in by_split.items():
            with (stage / f"{split}.jsonl").open("w", encoding="utf-8") as dest:
                for row in rows:
                    guard.add(split, row)
                    dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            info = dict(hands=len(rows), recordings=sorted({r["recording_id"] for r in rows}),
                        frames=len({FrameKey.from_row(r) for r in rows}),
                        sync_times=len({(r["recording_id"], r["sync_index"]) for r in rows}),
                        camera_hands=dict(Counter(r["camera"] for r in rows)))
            if by_time:
                times = [time_by_sync[r["sync_index"]] for r in rows]
                info["time_range"] = [min(times), max(times)]
            report["splits"][split] = info
        (stage / "splits.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New prepared directory")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}", nargs=2, action="append", default=[], metavar=("ID", "EXPORT"))
    parser.add_argument("--time-holdout", nargs=2, metavar=("ID", "EXPORT"),
                        help="Use shared reference_timestamp_ns, falling back to sync_index for multiple cameras "
                             "or source_frame_index when available for one camera. Equal times stay together.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Use 0 to omit test in time holdout")
    args = parser.parse_args()
    recordings = [(s, name, root) for s in ("train", "val", "test") for name, root in getattr(args, s)]
    if args.time_holdout:
        if recordings:
            parser.error("--time-holdout cannot be combined with whole-recording assignments")
        recordings = [("time", *args.time_holdout)]
    report = prepare_splits(args.output, recordings, by_time=bool(args.time_holdout),
                            val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
