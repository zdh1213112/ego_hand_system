# Runtime sources

The only Git submodule is the pinned WiLoR checkout:

```bash
git submodule update --init third_party/WiLoR
```

Shared MANO fitting also uses WiLoR's MANO wrapper. Revisions and model requirements are recorded in `manifest.json` and each inference run's configuration.
