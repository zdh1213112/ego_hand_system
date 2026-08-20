# MANO model assets

This directory may contain MANO model data in a local working bundle, but Git is
configured to ignore it. Obtain the assets through the official MANO licensing
process. The active WiLoR-compatible fitting, validation and visualization
pipeline requires only this file:

```text
MANO_RIGHT.pkl
```

`MANO_LEFT.pkl` may remain for legacy experiments, but current runtime code
does not load it.

Validate the files and the prepared EGO input before fitting:

```bash
python scripts/check_mano_assets.py \
  --model-dir models/mano \
  --mano-source third_party/MANO \
  --input output/mano_preparation/mano_input.npz
```

Do not redistribute the licensed `.pkl` files with this project.
