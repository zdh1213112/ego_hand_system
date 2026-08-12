# External dependencies

## Quick start

Public source dependencies are pinned Git submodules. The recommended flow
avoids downloading Basalt's nested vcpkg checkout, which is needed only for a
source build:

```bash
git clone https://github.com/zdh1213112/ego_hand_system.git
cd ego_hand_system
./scripts/setup_third_party.sh
```

For an existing checkout:

```bash
git submodule sync
git submodule update --init third_party/MANO third_party/basalt
./scripts/setup_third_party.sh
```

The machine-readable dependency list is `third_party/manifest.json`.

`setup_third_party.sh` initializes the two top-level source submodules and
downloads the fixed MediaPipe Hand Landmarker model with SHA-256 verification.
Basalt's nested vcpkg submodule is intentionally skipped unless you build Basalt
from source.

The external MANO PyTorch source is a Git submodule at:

```text
third_party/MANO/
```

The submodule contains loader code only. Register and accept the MANO license at
<https://mano.is.tue.mpg.de/>, then place `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`
in `models/mano/`. These model files are intentionally not distributed.

After downloading the official MANO archive or extracting it locally, install it
without manually searching for the two files:

```bash
python scripts/install_mano_models.py --source /path/to/MANO/archive_or_directory
```

The installer recursively locates both PKL files, copies them into `models/mano/`
and reports whether they match the project-tested MANO v1.2 checksums.

The local EGO real-time bundle also expects the Linux x86_64 OrbbecSDK 2.9.0 at:

```text
third_party/orbbec_sdk/
```

Only the SDK headers, shared library, runtime extensions, EGO configuration and
udev rule are needed. This vendor binary directory is ignored by Git. Obtain it
from the EGO device delivery package or an official Orbbec distribution and
review the applicable redistribution terms before publishing binaries.

Basalt stereo-inertial VIO source is a Git submodule pinned to release 0.1.7 at:

```text
third_party/basalt/
```

The offline runner uses a minimal local Basalt 0.1.7 x86_64 runtime at:

```text
third_party/basalt_runtime/
```

The runtime contains only `bin/basalt_vio`, `lib/libbasalt.so`, a checksum/version
record and a README. The platform-specific runtime is ignored by Git. Keep Basalt's
BSD-3-Clause license with the source and recreate the runtime from the official
v0.1.7 release or from a local source build on a fresh checkout.

WiLoR hand reconstruction source is a pinned Git submodule at:

```text
third_party/WiLoR/
```

The public checkout contains source code only. Its checkpoint, detector and
MANO mean-parameter asset are private/local assets under `models/wilor/` and
are checked with `python scripts/check_third_party.py --require-wilor`.
WiLoR is an optional parallel inference route; the default MediaPipe+MANO route
does not require its model bundle.

Run an audit at any time:

```bash
python scripts/check_third_party.py
python scripts/check_third_party.py --require-mano
python scripts/check_third_party.py --require-live
python scripts/check_third_party.py --require-basalt
python scripts/check_third_party.py --require-wilor
```

## Optional private/local asset bundle

If your organization has permission to redistribute the MANO model files and
OrbbecSDK, keep them in a private asset archive using these exact paths:

```text
models/hand_landmarker.task
models/mano/MANO_LEFT.pkl
models/mano/MANO_RIGHT.pkl
third_party/orbbec_sdk/...
third_party/basalt_runtime/...
models/wilor/wilor_final.ckpt
models/wilor/detector.pt
models/wilor/model_config.yaml
models/wilor/mano_mean_params.npz
```

Extract that archive at the repository root, then run:

```bash
python scripts/check_third_party.py --require-mano --require-live --require-basalt
```

The repository also provides a guarded installer for this private archive:

```bash
./scripts/install_local_assets.sh --archive /path/ego_hand_assets.tar.gz
```

Do not upload this private archive to a public repository unless the applicable
model, SDK and binary redistribution terms explicitly allow it.
