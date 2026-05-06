# Automatic Speech Recognition Companion Code

This directory contains the reader-facing scaffold for the automatic speech
recognition chapter homework. The default workflow is dependency-light and uses
only the Python standard library. It does not download or run a speech model by
default; instead, it validates an ASR manifest, reads audio duration from WAV
files, scores saved ASR hypotheses against references, measures real-time
factor from recorded processing time, and writes reproducible artifacts.

The main script is:

```text
asr_transcription_eval.py
```

The optional Qwen3-ASR experiment helper is:

```text
asr_qwen3_experiment.py
```

The interactive walkthrough is:

```text
asr_transcription_walkthrough.ipynb
```

Use the script for repeatable command-line runs, dependency checks, and
repository smoke tests:

```sh
cd ..
poetry install --with asr
cd chapter_automatic_speech_recognition
python asr_transcription_eval.py --check-deps --allow-missing-deps
```

The bare `python` commands in this README assume the Poetry virtualenv is
active. If it is not active, prefix the `python` invocation with `poetry run`
from the same working directory.

From the companion-code root, `make asr-code-check` also parses both ASR Python
files without writing `__pycache__` bytecode.

Create a tiny synthetic sample dataset and score it:

```sh
python asr_transcription_eval.py --write-sample-data sample_audio
python asr_transcription_eval.py \
  --run \
  --manifest sample_audio/sample_manifest.jsonl \
  --save-artifacts \
  --artifact-dir artifacts/sample_asr
```

The sample data contains short generated WAV files, reference transcripts, and
two saved hypothesis fields: `baseline_hypothesis` and `controlled_hypothesis`.
It is only a scoring and reporting scaffold. For the homework, replace the
sample hypotheses with transcripts produced by a pretrained ASR model such as
Qwen3-ASR when the runtime environment supports it.

Manifest records should contain:

```json
{"audio_id": "clip001", "path": "audio.wav", "split": "validation", "language": "English", "reference_text": "gradient descent updates the weights", "baseline_hypothesis": "gradient decent updates weights", "baseline_processing_seconds": 0.42}
```

The script writes `scored_segments.jsonl`, `run_summary.json`, and
`error_examples.jsonl` when `--save-artifacts` is supplied. Use those measured
values to fill the homework run table.

For a GPU runtime that already has Qwen3-ASR installed, prepare a small public
LibriSpeech-derived manifest and run one offline baseline like this:

```sh
python asr_qwen3_experiment.py prepare-librispeech \
  --data-root data/librispeech \
  --output-dir artifacts/librispeech_subset \
  --limit 12 \
  --validation-count 8

python asr_qwen3_experiment.py run-qwen \
  --manifest artifacts/librispeech_subset/manifest.jsonl \
  --run-id qwen3-0p6b-transformers \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --backend transformers \
  --language-from-manifest \
  --output-jsonl artifacts/qwen_outputs.jsonl \
  --metadata-json artifacts/qwen3-0p6b-transformers-metadata.json
```

Use `build-scoring-manifest` to combine two hypothesis runs into the manifest
shape expected by `asr_transcription_eval.py`, then score it with the standard
scaffold.

`qwen-asr` and `vllm` are not pinned in the shared Poetry environment because
their supported wheels depend on the active GPU and CUDA runtime. In a reviewed
GPU environment, install the provider packages separately, for example
`pip install qwen-asr` for the Transformers backend or
`pip install "qwen-asr[vllm]"` for the vLLM backend, then record the exact
package versions, model name, model revision, hardware, and install command in
the generated metadata before comparing or publishing results.

The default script does not read `.env`. If an external ASR service or API is
used, `.env` should contain only secrets such as API keys. Audio manifests,
reference transcripts, generated hypotheses, prompts, settings, and score files
should remain ordinary data artifacts, not environment variables.

The OpenAI-compatible API helper redacts `--base-url` in generated public
metadata by default. Pass `--record-base-url` only when the endpoint itself is
safe to publish.
