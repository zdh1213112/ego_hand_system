"""Cache a fixed local FAH/WiLoR hand detector on selected native-image manifests."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

from .boxes import _proposal
from .data import FrameManifest


def generate_cache(dataset, splits, output, detector, *, side_by_class, provenance):
    """Run on each native image once. No box or handedness annotations enter detection."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = dict(frames=0, proposals=0, empty_frames=0)
    groups = set()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            for split in splits:
                manifest = FrameManifest(dataset, split)
                if groups & manifest.groups:
                    raise ValueError(f"Repeated (recording_id,sync_index) between supplied splits at {split}")
                groups.update(manifest.groups)
                for key in manifest.keys:
                    boxes, scores, classes = detector(manifest.read_image(key))
                    proposals = []
                    for box, score, cls in zip(boxes, scores, classes, strict=True):
                        item = dict(bbox=box.tolist(), score=float(score), side=side_by_class[int(cls)])
                        _proposal(item)
                        proposals.append(item)
                    row = dict(schema_version=1, **key._asdict(), detector=provenance, proposals=proposals)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["frames"] += 1
                    counts["proposals"] += len(proposals)
                    counts["empty_frames"] += not proposals
        # Publish the completed cache without replacing a simultaneous producer's output.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", nargs="+", choices=["train", "val", "test"], required=True)
    parser.add_argument("--detector", type=Path, required=True, help="Local WiLoR/FAH YOLO weights")
    parser.add_argument("--output", type=Path, required=True, help="New proposal JSONL")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fast", action="store_true",
                        help="Use fah.yolo fast inference (512 px); default is the video runner's Ultralytics path")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.nms_iou <= 1:
        parser.error("confidence and nms-iou must be in (0,1]")
    from fah.detector import build_hand_detector

    with args.detector.open("rb") as weights:
        sha = hashlib.file_digest(weights, "sha256").hexdigest()
    detector = build_hand_detector("wilor", weights=str(args.detector.resolve()), device=args.device,
                                   conf=args.confidence, iou=args.nms_iou, fast=args.fast)
    names = {k: str(v).lower() for k, v in detector.names.items()}
    if names == {0: "left", 1: "right"}:
        side_by_class = {0: 0, 1: 1}
    elif names == {0: "hand"}:
        side_by_class = {0: None}
    else:
        raise ValueError(f"Expected left/right or a single hand detector class, got {names}")
    provenance = dict(backend="fah.detector.wilor", fah_version=version("fah"), fast=args.fast,
                      weights_sha256=sha, confidence=args.confidence, nms_iou=args.nms_iou,
                      device=args.device, class_names=names)
    if not args.fast:
        provenance["ultralytics_version"] = version("ultralytics")
    counts = generate_cache(args.dataset, args.split, args.output, detector,
                            side_by_class=side_by_class, provenance=provenance)
    print(json.dumps(dict(output=str(args.output), **counts, detector=provenance)))
