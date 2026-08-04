# Model files

Model binaries may be present in a local working bundle, but are intentionally
excluded from Git by the repository `.gitignore`.

Place the MediaPipe Hand Landmarker task file at:

```text
models/hand_landmarker.task
```

Obtain it from the official Google AI Edge MediaPipe Hand Landmarker model
download page and review its terms before use.

MANO requires separately licensed assets. See [mano/README.md](mano/README.md).
Do not commit `*.task`, `*.pkl`, or extracted MANO/SMPL-H archives. A prepared
local bundle should contain only the two MANO hand files needed by this pipeline;
the much larger SMPL-H model files are not required.
