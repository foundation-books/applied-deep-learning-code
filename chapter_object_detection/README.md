# Object Detection Companion Workflow

This directory contains the reader-facing YOLO medical-imaging workflow that
accompanies the object detection chapter.

For an interactive path through the same workflow, open
`yolo_medical_detection_walkthrough.ipynb`. The notebook is a guided report
scaffold for label inspection, dry-run YOLO commands, optional training,
metric summaries, and qualitative examples. The `.py` files remain the source
of truth for repeatable command-line runs and smoke checks.

The workflow is designed to be useful before a dataset is available.  It can
write a dataset template, inspect YOLO-format labels, print reproducible
Ultralytics training, validation, test, prediction, and export commands, record
command timing, summarize `results.csv` files, and plot a compact mAP curve.
The chapter includes one thor1 reference run on a small RSNA pneumonia subset;
the workflow remains usable for rerunning or replacing that reference evidence.

## Setup

From the companion-code root, install the object-detection dependency group,
then run the examples from this chapter directory:

```sh
poetry install --with object-detection
cd chapter_object_detection
```

The bare `python` commands in this README assume the Poetry virtualenv is
active. If it is not active, prefix the `python` invocation with `poetry run`
from the same working directory.

The RSNA converter downloads DICOM files with Python's standard-library HTTP
client. It requires network access for that optional preparation step, or you
can manually place the expected DICOM files in `--cache-dir`.

Ultralytics YOLO is a separate upstream package. Ultralytics publishes
open-source use under AGPL-3.0 and offers a separate Enterprise license for
uses that cannot satisfy AGPL obligations. The MIT license for this companion
code does not relicense Ultralytics, YOLO weights, or third-party datasets.

## Workflow

Create a dataset YAML template:

```sh
python yolo_medical_workflow.py make-template --output-dir /tmp/medical-template
```

Prepare a small public microscopy fallback dataset for a classroom dry run.
Clone the MIT-licensed BCCD repository outside the textbook repo, then convert
its VOC annotations into YOLO format:

```sh
git clone https://github.com/Shenggan/BCCD_Dataset.git /tmp/BCCD_Dataset
python yolo_medical_workflow.py prepare-bccd \
  --source-dir /tmp/BCCD_Dataset \
  --output-dir /tmp/bccd-yolo
```

Prepare the RSNA pneumonia reference subset from a downloaded
`stage_2_train_labels.csv` file. This command downloads only the selected DICOM
images from the documented Hugging Face mirror and writes a YOLO dataset with
one `opacity` class:

```sh
python prepare_rsna_yolo_subset.py \
  --labels-csv /path/to/stage_2_train_labels.csv \
  --output-dir /tmp/rsna-pneumonia-yolo \
  --cache-dir /tmp/rsna-dicom-cache
```

The RSNA challenge terms, attribution, and non-identification requirements still
govern these files even when the helper downloads selected images from a mirror.
Do not publish downloaded DICOM files or converted images unless your use
satisfies the official RSNA terms.

The helper pins the Hugging Face mirror revision used for downloads, rejects
implausibly small DICOM files, and records each downloaded file's byte count and
SHA-256 digest in the generated manifest for local provenance.

Inspect prepared labels:

```sh
python yolo_medical_workflow.py inspect-labels \
  --data-yaml medical-detection.yaml \
  --split train \
  --output-dir /tmp/medical-label-audit
```

Print the baseline training command without launching training:

```sh
python yolo_medical_workflow.py yolo \
  --mode train \
  --model yolo26n.pt \
  --data-yaml medical-detection.yaml \
  --epochs 100 \
  --imgsz 640 \
  --dry-run
```

Record the exact installed package versions before a real experiment:

```sh
python yolo_medical_workflow.py env-report \
  --output-dir runs/object-detection/baseline
```

After training, summarize metrics and timing sidecars:

```sh
python yolo_medical_workflow.py summarize-runs \
  --runs-dir runs/object-detection \
  --output-dir runs/object-detection-summary
```

Create a compact validation mAP plot:

```sh
python yolo_medical_workflow.py plot-metrics \
  --runs-dir runs/object-detection \
  --output-dir runs/object-detection-plots
```

For a latency pass, run prediction on a fixed image folder. The workflow writes
`predict_timing.json` and computes milliseconds per image when the source is a
local image file or image directory:

```sh
python yolo_medical_workflow.py yolo \
  --mode predict \
  --model runs/object-detection/baseline/weights/best.pt \
  --source /path/to/held-out-images \
  --project runs/object-detection \
  --name baseline
```

Run the dependency-light smoke check from the repository root:

```sh
make object-detection-code-check
```
