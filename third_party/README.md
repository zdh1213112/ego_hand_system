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
