# Contributing

This repository contains companion code for the Applied Deep Learning textbook.
Keep changes scoped to reader-facing code, notebooks, and compact reference
artifacts.

Before opening a change:

```sh
make public-release-check
```

Do not commit `.env` files, API keys, downloaded datasets, model weights,
audio, DICOM files, generated run directories, or notebook outputs. If a change
requires a large generated artifact, document how to reproduce it instead of
checking it in.
