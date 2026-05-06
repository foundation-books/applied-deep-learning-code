# Notices

This companion-code repository contains original textbook code and small
reference artifacts. The MIT license in `LICENSE` applies to the original code
and documentation unless a file says otherwise.

Third-party datasets, pretrained models, model weights, generated datasets, and
downloaded challenge files are not relicensed by this repository. Follow the
terms from each upstream provider before redistributing data or artifacts.

Important upstream data notes:

- AG News / AG's News Topic Classification: the public dataset cards and source
  descriptions identify the original news corpus as research/non-commercial or
  license-unknown. Checked-in reference artifacts therefore redact raw news text.
  See https://huggingface.co/datasets/fancyzhx/ag_news.
- Food-101: the dataset contains Foodspotting images. Upstream dataset cards
  state that uses beyond scientific fair use must be negotiated with the image
  owners according to Foodspotting terms.
  See https://huggingface.co/datasets/ethz/food101.
- RSNA Pneumonia Detection Challenge: the RSNA terms permit academic,
  educational, commercial, and non-commercial use when attribution and
  non-identification requirements are followed. This repository does not bundle
  DICOM images. The optional helper that prepares a tiny YOLO subset downloads
  selected DICOM files from a Hugging Face mirror, but the official RSNA terms
  remain the governing source for use, attribution, and redistribution.
  See https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
  and the linked Pneumonia Detection Challenge Terms of Use and Attribution.
- Kvasir-SEG: the segmentation training helper can write qualitative grids and
  failure grids derived from Kvasir-SEG images and masks. The upstream dataset
  page restricts use to research and educational purposes, forbids commercial
  use without prior written permission, and requires citation of the dataset
  paper in publications that use or report results based on the data.
  See https://datasets.simula.no/kvasir-seg/.
- LibriSpeech: the ASR helper can download LibriSpeech through TorchAudio and
  write a small derived manifest with copied or noisy WAV files and transcripts.
  OpenSLR lists LibriSpeech under CC BY 4.0. Preserve required attribution when
  sharing generated subsets or results derived from the corpus.
  See https://www.openslr.org/12/.
- BCCD: the object-detection workflow can convert the BCCD repository into YOLO
  format by copying its JPEG images and annotations. The upstream repository
  identifies BCCD as MIT licensed and credits earlier data/annotation sources.
  Preserve the upstream license and notice if publishing converted datasets or
  derived image artifacts. See https://github.com/Shenggan/BCCD_Dataset.
- CIFAR-10, ACL IMDB, Kaggle/Jigsaw Toxic Comment Classification, and Kaggle
  Bag of Words Meets Bags of Popcorn data are not bundled here. Download them
  from their original providers, follow the provider license or competition
  terms, and do not commit raw challenge files, downloaded text, submissions, or
  generated outputs.
- Ultralytics YOLO: the `object-detection` dependency group can install the
  `ultralytics` package. Ultralytics publishes open-source use under AGPL-3.0
  and offers a separate Enterprise license for uses that cannot satisfy AGPL
  obligations. Review https://www.ultralytics.com/license before training,
  distributing, or integrating YOLO models.
- Qwen-ASR and vLLM: optional ASR runtimes are not installed by the shared
  Poetry environment because their supported wheels depend on the active GPU
  runtime and provider release channel. The ASR streaming helper uses the vLLM
  backend only when explicitly selected. Install these packages separately in a
  reviewed environment, record the exact package versions in generated metadata,
  and follow the provider's package and model terms for Qwen checkpoints.
- Qwen-TTS: the optional TTS runtime is not installed by the shared Poetry
  environment because provider wheels, hardware support, and model access can
  depend on the active runtime and release channel. Install it separately in a
  reviewed environment, record the exact package version in generated metadata,
  and follow the provider's package and model terms for Qwen checkpoints.
- Kaggle, Hugging Face, Qwen, Unsloth, TorchVision, and other model or dataset
  providers may impose separate model, data, or service terms.

Reference files with historical `thor1-*` names are compact reproducibility
records from a CUDA reference run. The prefix is retained for compatibility
with the book text and figure-generation scripts; public-facing metadata avoids
recording private hostnames or local home-directory paths.
