# External source trees

The local working bundle may contain the external MANO PyTorch source at:

```text
third_party/MANO/
```

This directory is ignored by Git to avoid vendoring a separate project and its
history accidentally. Obtain the source from its upstream repository when
creating a fresh checkout, review its license, and keep its `LICENSE` file.
