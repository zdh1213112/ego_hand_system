"""Frame-level organization of per-hand WiLoR pose annotations."""
from collections import defaultdict
import json


def require_both_hands(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[(item["camera"], item["sync_index"])].append(item)
    accepted, rejected = [], []
    for (camera, sync), group in grouped.items():
        sides = [int(item["side"]) for item in group]
        if len(sides) != len(set(sides)):
            raise ValueError(f"duplicate hand identity: {camera} sync={sync}")
        if set(sides) == {0, 1}:
            accepted.extend(group)
        else:
            rejected.append(dict(camera=camera, sync_index=sync,
                reason="both_hands_required", available_sides=sides))
    return accepted, rejected


def assign_image_paths(items, shared):
    frames = {}
    for index, item in enumerate(items):
        key = (item["camera"], int(item["sync_index"])) if shared else index
        image_index = frames.setdefault(key, len(frames))
        item["image_stem"] = f"{image_index:06d}"


def write_frame_index(path, rows):
    frames = {}
    for row in rows:
        frame = frames.setdefault(row["image"], {
            key: row[key] for key in ("image", "camera", "sync_index", "source_frame_index",
                "reference_timestamp_ns", "source_dataset_id", "source_group", "calibration_group") if key in row})
        frame.setdefault("hands", []).append({key: row[key] for key in ("side", "label", "index")})
    for frame in frames.values():
        frame["hands"].sort(key=lambda hand: hand["side"])
        sides = [h["side"] for h in frame["hands"]]
        if len(sides) != len(set(sides)):
            raise ValueError("duplicate hand side in a camera frame")
        frame["hand_count"] = len(sides)
    path.write_text("".join(json.dumps(frame, ensure_ascii=False)+"\n" for frame in frames.values()))
