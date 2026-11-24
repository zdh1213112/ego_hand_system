"""Download the released weights and geometry used by train_hand.py."""
import argparse
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "pretrained_models/hand"
WEIGHTS = {
    "fah": ("noahcao/FAH", "9f8a86f47a83cb23226ae7ab696c0500dabf6c82", "fah-s/model.pt"),
    "gem-x": ("nvidia/EgoHand-v1", "a153616c5192e6a8b09dbd302f682093080c5b83", "egohand_s060000.ckpt"),
}
SOMA_FILES = (
    "SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json", "SOMAHand.npz",
    "MHR/mhr_model_lod1.pt", "MHR/base_body_lod1.obj", "MHR/SOMA_wrap_lod1.obj",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=WEIGHTS)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    repo, revision, filename = WEIGHTS[args.model]
    hf_hub_download(repo, filename, revision=revision, local_dir=args.root / "weights")
    if args.model == "gem-x":
        for filename in SOMA_FILES:
            hf_hub_download("nvidia/SOMA-X", filename, revision="v0.3.3", local_dir=args.root / "assets/soma")


if __name__ == "__main__":
    main()
