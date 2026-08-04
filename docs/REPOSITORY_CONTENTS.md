# Repository contents

Included:

- C++ calibration, session inspection, and epipolar validation code;
- Python MediaPipe stereo detection, triangulation, stabilization, MANO fitting,
  offline/live overlay rendering, GPU warm-start fitting, and 21-DOF export code;
- C++ and Python regression tests;
- pinned Conda environment definition;
- model placement and licensing notes.
- documented EGO stereo-depth, fisheye rectification, MediaPipe, and MANO
  coordinate pipeline with official references and an optimization roadmap.

Intentionally excluded from Git tracking (they may exist in the local bundle):

- EGO recordings and calibration sessions;
- generated MP4, CSV, JSON, JPG, and NPZ results;
- `build/`, `.venv/`, `python_deps/`, caches, and compiled binaries;
- MediaPipe `hand_landmarker.task`;
- MANO/SMPL-H `.pkl` files and extracted licensed archives;
- the external MANO source checkout under `third_party/MANO`.
- the local OrbbecSDK runtime under `third_party/orbbec_sdk`.

The original development directory remains unchanged. The prepared local bundle
is self-contained for the reference recording, while the Git-tracked file set
remains a small portable source package.
