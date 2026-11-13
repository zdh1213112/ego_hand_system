#!/usr/bin/env python3
"""Run the shared-MANO Orbbec pipeline alongside the existing EGO tools."""

import subprocess
import sys

from ego_hand_inference.run_hand_inference import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
