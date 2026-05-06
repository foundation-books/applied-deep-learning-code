# Text-to-Speech Companion Code

This directory contains a lightweight workflow scaffold for the text-to-speech
chapter. The script is designed so repository smoke checks can run without
downloading Qwen3-TTS model weights. Full voice-cloning inference requires the
optional `qwen-tts` package, suitable hardware or API access, and model access.

The bundled `tts_eval_manifest.csv` contains ten fixed target sentences for the
chapter homework. `--write-sample-data` copies those targets into a homework
folder and creates `results.csv`, `experiment_summary.csv`, and
`run_metadata.json` templates. The summary template uses the run-level fields
from the chapter: model ID, mode, reference duration, language, target count,
total generated duration, runtime, RTF, WER/CER, speaker similarity,
naturalness, and notes.

Open `tts_voice_cloning_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when
you want to create the sample homework files, preview scoring, and keep full
Qwen3-TTS inference behind an explicit flag. The notebook intentionally has no
saved outputs.

Create sample homework files:

```bash
python qwen3_tts_voice_cloning.py --write-sample-data /tmp/tts-demo
```

Score a completed result CSV:

```bash
python qwen3_tts_voice_cloning.py --score-results /tmp/tts-demo/results.csv
```

Check optional dependencies:

```bash
cd ..
poetry install --with tts
cd chapter_text_to_speech
python qwen3_tts_voice_cloning.py --check-deps --allow-missing-deps
```

The shared Poetry `tts` group installs only the dependency-light scoring and
audio utilities used by repository checks. It does not pin `qwen-tts`, because
provider wheels and model access can depend on the active GPU runtime and
release channel. For full Qwen3-TTS inference, install the provider package
separately in a reviewed environment, record the exact package version in run
metadata, and follow the package and model terms for the selected checkpoint.
Keep reference voice recordings, copied transcripts, generated voice-clone
audio, and listener notes private unless you have the necessary consent,
attribution, and model-provider rights to publish them.

The bare `python` commands in this README assume the Poetry virtualenv is
active. If it is not active, prefix the `python` invocation with `poetry run`
from the same working directory.

Run Qwen3-TTS locally when optional dependencies and model weights are
available:

```bash
python qwen3_tts_voice_cloning.py \
  --run-qwen \
  --model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio reference.wav \
  --reference-text reference.txt \
  --targets tts_eval_manifest.csv \
  --output-dir outputs \
  --language English
```

The full run records generated audio paths, runtime, generated duration when
available, aggregate metadata, and result placeholders that can be completed
with ASR-backcheck and listener scores.
