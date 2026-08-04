# Reference pipeline result

The repository contains the code used for an offline EGO stereo hand pipeline:

1. hardware-PTS stereo pairing;
2. KB fisheye rectification;
3. dual-view MediaPipe 21-landmark detection;
4. cross-camera association and triangulation;
5. missing-data handling and temporal stabilization;
6. licensed MANO fitting;
7. MANO mesh projection on the original fisheye video and joint-angle export.

On the development recording, the final visualization processed 398 synchronized
frame pairs and 729 visible hand instances. It produced a 1920x1080 overlay video
at approximately 33.6 CPU rendering FPS. The raw-fisheye and rectified projection
chains agreed to numerical precision (`2.54e-13 px` maximum residual).

These numbers are a reference result, not a general benchmark. Recordings, model
files, fitted parameters, videos, CSV output, and identifiable preview images are
not included in the public repository.

The exported finger values are geometric 3D bone angles. They are useful for
visualization and downstream analysis but are not medically calibrated anatomical
joint angles.
