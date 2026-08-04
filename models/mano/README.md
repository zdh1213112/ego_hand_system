# MANO model assets

This directory may contain MANO model data in a local working bundle, but Git is
configured to ignore it. Obtain the assets through the official MANO licensing
process, then place the two files here with these exact names:

```text
MANO_LEFT.pkl
MANO_RIGHT.pkl
```

Validate the files and the prepared EGO input before fitting:

```bash
python scripts/check_mano_assets.py \
  --model-dir models/mano \
  --mano-source third_party/MANO \
  --input output/mano_preparation/mano_input.npz
```

Do not redistribute the licensed `.pkl` files with this project.
