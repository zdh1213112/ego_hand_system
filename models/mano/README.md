# MANO asset

Shared WiLoR uses `MANO_RIGHT.pkl`. Obtain it through the official MANO licensing process and install it with:

```bash
python scripts/install_mano_models.py --source /path/to/authorized/download
```

Physical left hands use one X reflection of the right-canonical model and reversed mesh triangle winding. A separate left model is not required.
