# External source trees

The local working bundle may contain the external MANO PyTorch source at:

```text
third_party/MANO/
```

This directory is ignored by Git to avoid vendoring a separate project and its
history accidentally. Obtain the source from its upstream repository when
creating a fresh checkout, review its license, and keep its `LICENSE` file.

The local EGO real-time bundle also expects the Linux x86_64 OrbbecSDK 2.9.0 at:

```text
third_party/orbbec_sdk/
```

Only the SDK headers, shared library, runtime extensions, EGO configuration and
udev rule are needed. This vendor binary directory is ignored by Git. Obtain it
from the EGO device delivery package or an official Orbbec distribution and
review the applicable redistribution terms before publishing binaries.

Basalt stereo-inertial VIO source is expected at:

```text
third_party/basalt/
```

The offline runner uses a minimal local Basalt 0.1.7 x86_64 runtime at:

```text
third_party/basalt_runtime/
```

The runtime contains only `bin/basalt_vio`, `lib/libbasalt.so`, a checksum/version
record and a README. Both external directories are ignored by Git. Keep Basalt's
BSD-3-Clause license with the source and recreate the runtime from the official
v0.1.7 release or from a local source build on a fresh checkout.
