.PHONY: help public-release-check poetry-check text-hygiene-check python-syntax-check notebook-check notebook-explanation-check notebook-public-path-check \
	mnist-code-check cnn-code-check cnn-arch-code-check cnn-revisited-code-check \
	transfer-learning-code-check rnn-code-check embeddings-code-check \
	attention-transformers-code-check vit-code-check image-generation-code-check \
	rl-intro-code-check nnt-revisited-code-check rag-code-check llm-code-check \
	lora-code-check rl-llm-code-check asr-code-check tts-code-check \
	object-detection-code-check segmentation-code-check vlm-code-check

PYTHON ?= python3
POETRY ?= poetry
TMPDIR ?= /tmp
export PYTHONDONTWRITEBYTECODE := 1
PYTHON_FILES := $(shell find . -name '*.py' -not -path '*/__pycache__/*' | sort)
PY_SYNTAX_CHECK = $(PYTHON) -B -c 'import ast,pathlib,sys; [ast.parse(pathlib.Path(path).read_text(encoding="utf-8"), filename=path) for path in sys.argv[1:]]'

help:
	@printf '%s\n' 'Available public companion-code checks:'
	@printf '%s\n' '  public-release-check          Run all public-repo validation checks'
	@printf '%s\n' '  poetry-check                  Validate pyproject.toml and poetry.lock'
	@printf '%s\n' '  text-hygiene-check            Check text files for extra final blank lines'
	@printf '%s\n' '  python-syntax-check           Parse every Python file without writing bytecode'
	@printf '%s\n' '  notebook-check                Validate notebooks as JSON and reader-facing walkthroughs'
	@printf '%s\n' '  notebook-explanation-check    Check reader-facing notebook explanation coverage'
	@printf '%s\n' '  notebook-public-path-check    Check notebook companion paths from public repo root'
	@printf '%s\n' '  mnist-code-check              Smoke-check MNIST chapter code'
	@printf '%s\n' '  cnn-code-check                Smoke-check CNN basics chapter code'
	@printf '%s\n' '  cnn-arch-code-check           Smoke-check CNN architectures chapter code'
	@printf '%s\n' '  cnn-revisited-code-check      Smoke-check CNNs revisited chapter code'
	@printf '%s\n' '  transfer-learning-code-check  Smoke-check transfer-learning chapter code'
	@printf '%s\n' '  embeddings-code-check         Smoke-check embeddings chapter code'
	@printf '%s\n' '  attention-transformers-code-check  Smoke-check attention/Transformers code'
	@printf '%s\n' '  rnn-code-check                Smoke-check recurrent neural networks code'
	@printf '%s\n' '  vit-code-check                Smoke-check Vision Transformers code'
	@printf '%s\n' '  image-generation-code-check   Smoke-check image-generation chapter code'
	@printf '%s\n' '  rl-intro-code-check           Smoke-check reinforcement-learning intro code'
	@printf '%s\n' '  nnt-revisited-code-check      Smoke-check training revisited chapter code'
	@printf '%s\n' '  rag-code-check                Smoke-check RAG chapter code'
	@printf '%s\n' '  llm-code-check                Smoke-check LLM prompt benchmark code'
	@printf '%s\n' '  lora-code-check               Smoke-check LoRA/QLoRA chapter code'
	@printf '%s\n' '  rl-llm-code-check             Smoke-check RL-for-LLM chapter code'
	@printf '%s\n' '  asr-code-check                Smoke-check ASR chapter code'
	@printf '%s\n' '  tts-code-check                Smoke-check TTS chapter code'
	@printf '%s\n' '  object-detection-code-check   Smoke-check object detection workflow'
	@printf '%s\n' '  segmentation-code-check       Smoke-check segmentation workflow'
	@printf '%s\n' '  vlm-code-check                Smoke-check vision-language-model code'

public-release-check: poetry-check text-hygiene-check python-syntax-check notebook-check \
	mnist-code-check cnn-code-check cnn-arch-code-check cnn-revisited-code-check \
	transfer-learning-code-check rnn-code-check embeddings-code-check \
	attention-transformers-code-check vit-code-check image-generation-code-check \
	rl-intro-code-check nnt-revisited-code-check rag-code-check llm-code-check \
	lora-code-check rl-llm-code-check asr-code-check tts-code-check \
	object-detection-code-check segmentation-code-check vlm-code-check

poetry-check:
	$(POETRY) check --lock

text-hygiene-check:
	$(PYTHON) -B check_text_hygiene.py

python-syntax-check:
	$(PY_SYNTAX_CHECK) $(PYTHON_FILES)

notebook-check:
	$(PYTHON) -m json.tool chapter_attention_transformers/attention_transformers_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_automatic_speech_recognition/asr_transcription_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_cnn_architectures/resnet_cifar10_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_cnn_basics/cifar10_keras3_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_cnn_revisited/convnext_food101_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_embeddings/kaggle_bag_of_embeddings_sentiment_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_from_transformer_to_llms/llm_prompt_benchmark_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_image_generation/image_generation_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_lora_qlora_adaptation/lora_unsloth_course_assistant_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_mnist_python/mnist_keras3_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_neural_network_training/cifar10_training_lab.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_neural_network_training_revisited/transformer_distillation_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_object_detection/yolo_medical_detection_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_recurrent_neural_networks/imdb_rnn_keras3_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_reinforcement_learning_intro/rl_intro_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_reinforcement_learning_llm_training/rl_lora_unsloth_reasoning_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_retrieval_augmented_generation/rag_course_assistant_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_segmentation_cnn_transformers/segmentation_medical_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_text_to_speech/tts_voice_cloning_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_transfer_learning_fastai/food101_fastai_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_vision_language_models/vlm_reasoning_walkthrough.ipynb >/dev/null
	$(PYTHON) -m json.tool chapter_vision_transformers/pretrained_vit_walkthrough.ipynb >/dev/null
	$(PYTHON) -B check_notebook_public_paths.py
	$(PYTHON) -B check_notebook_explanations.py

notebook-explanation-check:
	$(PYTHON) -B check_notebook_explanations.py

notebook-public-path-check:
	$(PYTHON) -B check_notebook_public_paths.py

mnist-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_mnist_python/mnist_shape_check.py chapter_mnist_python/mnist_keras3.py chapter_mnist_python/mnist_pytorch.py
	$(PYTHON) -m json.tool chapter_mnist_python/mnist_keras3_walkthrough.ipynb >/dev/null
	cd chapter_mnist_python && $(PYTHON) mnist_shape_check.py >/dev/null
	cd chapter_mnist_python && $(PYTHON) mnist_keras3.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_mnist_python && $(PYTHON) mnist_pytorch.py --check-deps --allow-missing-deps >/dev/null

cnn-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_cnn_basics/cifar10_keras3.py adl_plot_style.py
	$(PYTHON) -m json.tool chapter_cnn_basics/cifar10_keras3_walkthrough.ipynb >/dev/null
	cd chapter_cnn_basics && $(PYTHON) cifar10_keras3.py --check-deps --allow-missing-deps >/dev/null

cnn-arch-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_cnn_architectures/resnet50_cifar10_keras3.py chapter_cnn_architectures/resnet50_cifar10_pytorch.py adl_plot_style.py
	$(PYTHON) -m json.tool chapter_cnn_architectures/resnet_cifar10_walkthrough.ipynb >/dev/null
	cd chapter_cnn_architectures && $(PYTHON) resnet50_cifar10_keras3.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_cnn_architectures && $(PYTHON) resnet50_cifar10_pytorch.py --check-deps --allow-missing-deps >/dev/null

cnn-revisited-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_cnn_revisited/convnext_food101_pytorch.py
	$(PYTHON) -m json.tool chapter_cnn_revisited/convnext_food101_walkthrough.ipynb >/dev/null
	cd chapter_cnn_revisited && $(PYTHON) convnext_food101_pytorch.py --check-deps --allow-missing-deps >/dev/null

transfer-learning-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_transfer_learning_fastai/food101_fastai.py chapter_transfer_learning_fastai/food101_keras3.py
	$(PYTHON) -m json.tool chapter_transfer_learning_fastai/food101_fastai_walkthrough.ipynb >/dev/null
	cd chapter_transfer_learning_fastai && $(PYTHON) food101_fastai.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_transfer_learning_fastai && $(PYTHON) food101_keras3.py --check-deps --allow-missing-deps >/dev/null

rnn-code-check: poetry-check
	$(PY_SYNTAX_CHECK) imdb_sentiment_shared.py chapter_recurrent_neural_networks/imdb_rnn_keras3.py
	$(PYTHON) -m json.tool chapter_recurrent_neural_networks/imdb_rnn_keras3_walkthrough.ipynb >/dev/null
	cd chapter_recurrent_neural_networks && $(PYTHON) imdb_rnn_keras3.py --check-deps --allow-missing-deps >/dev/null

embeddings-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_embeddings/kaggle_bag_of_embeddings_sentiment.py
	$(PYTHON) -m json.tool chapter_embeddings/kaggle_bag_of_embeddings_sentiment_walkthrough.ipynb >/dev/null
	cd chapter_embeddings && $(PYTHON) kaggle_bag_of_embeddings_sentiment.py --check-deps --allow-missing-deps >/dev/null

attention-transformers-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_attention_transformers/attention_scaling_benchmark.py chapter_attention_transformers/toxic_comments_data_audit.py chapter_attention_transformers/toxic_comments_tfidf_baseline.py chapter_attention_transformers/toxic_comments_transformer.py
	$(PYTHON) -m json.tool chapter_attention_transformers/attention_transformers_walkthrough.ipynb >/dev/null
	cd chapter_attention_transformers && $(PYTHON) attention_scaling_benchmark.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_attention_transformers && $(PYTHON) toxic_comments_data_audit.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_attention_transformers && $(PYTHON) toxic_comments_tfidf_baseline.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_attention_transformers && $(PYTHON) toxic_comments_transformer.py --check-deps --allow-missing-deps >/dev/null

vit-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_vision_transformers/pretrained_vit_experiment.py chapter_vision_transformers/summarize_vit_runs.py chapter_vision_transformers/smoke_summarize_vit_runs.py
	$(PYTHON) -m json.tool chapter_vision_transformers/pretrained_vit_walkthrough.ipynb >/dev/null
	cd chapter_vision_transformers && $(PYTHON) smoke_summarize_vit_runs.py >/dev/null
	cd chapter_vision_transformers && $(PYTHON) pretrained_vit_experiment.py --check-deps --allow-missing-deps >/dev/null

image-generation-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_image_generation/image_generation_experiments.py
	$(PYTHON) -m json.tool chapter_image_generation/image_generation_walkthrough.ipynb >/dev/null
	cd chapter_image_generation && $(PYTHON) image_generation_experiments.py --check-deps --allow-missing-deps >/dev/null

rl-intro-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_reinforcement_learning_intro/gymnasium_rl_experiment.py
	$(PYTHON) -m json.tool chapter_reinforcement_learning_intro/rl_intro_walkthrough.ipynb >/dev/null
	cd chapter_reinforcement_learning_intro && $(PYTHON) gymnasium_rl_experiment.py --check-deps --allow-missing-deps >/dev/null

nnt-revisited-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_neural_network_training_revisited/transformer_distillation_experiment.py chapter_neural_network_training_revisited/regenerate_reference_plots.py
	$(PYTHON) -m json.tool chapter_neural_network_training_revisited/transformer_distillation_walkthrough.ipynb >/dev/null
	cd chapter_neural_network_training_revisited && $(PYTHON) transformer_distillation_experiment.py --check-deps --allow-missing-deps >/dev/null

rag-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_retrieval_augmented_generation/rag_course_assistant.py chapter_retrieval_augmented_generation/rag_thor1_experiments.py
	$(PYTHON) -m json.tool chapter_retrieval_augmented_generation/rag_course_assistant_walkthrough.ipynb >/dev/null
	cd chapter_retrieval_augmented_generation && $(PYTHON) rag_course_assistant.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_retrieval_augmented_generation && $(PYTHON) rag_course_assistant.py --preview-corpus >/dev/null

llm-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_from_transformer_to_llms/llm_prompt_benchmark.py
	$(PYTHON) -m json.tool chapter_from_transformer_to_llms/llm_prompt_benchmark_walkthrough.ipynb >/dev/null
	cd chapter_from_transformer_to_llms && $(PYTHON) llm_prompt_benchmark.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_from_transformer_to_llms && $(PYTHON) llm_prompt_benchmark.py --preview --prompts sample_prompts.jsonl >/dev/null

lora-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_lora_qlora_adaptation/lora_unsloth_course_assistant.py
	$(PYTHON) -m json.tool chapter_lora_qlora_adaptation/lora_unsloth_course_assistant_walkthrough.ipynb >/dev/null
	cd chapter_lora_qlora_adaptation && $(PYTHON) lora_unsloth_course_assistant.py --check-deps --allow-missing-deps >/dev/null

rl-llm-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_reinforcement_learning_llm_training/reward_functions.py chapter_reinforcement_learning_llm_training/rl_lora_unsloth_reasoning.py chapter_reinforcement_learning_llm_training/unsloth_grpo_gspo_reasoning_experiment.py
	$(PYTHON) -m json.tool chapter_reinforcement_learning_llm_training/rl_lora_unsloth_reasoning_walkthrough.ipynb >/dev/null
	cd chapter_reinforcement_learning_llm_training && $(PYTHON) rl_lora_unsloth_reasoning.py --unit-test-rewards >/dev/null
	cd chapter_reinforcement_learning_llm_training && $(PYTHON) rl_lora_unsloth_reasoning.py --check-deps --allow-missing-deps >/dev/null
	cd chapter_reinforcement_learning_llm_training && $(PYTHON) unsloth_grpo_gspo_reasoning_experiment.py --check-deps --allow-missing-deps >/dev/null

asr-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_automatic_speech_recognition/asr_transcription_eval.py chapter_automatic_speech_recognition/asr_qwen3_experiment.py
	$(PYTHON) -m json.tool chapter_automatic_speech_recognition/asr_transcription_walkthrough.ipynb >/dev/null
	cd chapter_automatic_speech_recognition && $(PYTHON) asr_transcription_eval.py --check-deps --allow-missing-deps >/dev/null
	tmpdir="$$(mktemp -d "$(TMPDIR)/adl-asr-smoke.XXXXXX")"; \
		cd chapter_automatic_speech_recognition && \
		$(PYTHON) asr_transcription_eval.py --write-sample-data "$$tmpdir/sample_audio" >/dev/null && \
		$(PYTHON) asr_transcription_eval.py --run --manifest "$$tmpdir/sample_audio/sample_manifest.jsonl" >/dev/null; \
		status="$$?"; \
		rm -rf "$$tmpdir"; \
		exit "$$status"

tts-code-check: poetry-check
	$(PY_SYNTAX_CHECK) chapter_text_to_speech/qwen3_tts_voice_cloning.py
	$(PYTHON) -m json.tool chapter_text_to_speech/tts_voice_cloning_walkthrough.ipynb >/dev/null
	cd chapter_text_to_speech && $(PYTHON) qwen3_tts_voice_cloning.py --check-deps --allow-missing-deps >/dev/null
	tmpdir="$$(mktemp -d "$(TMPDIR)/adl-tts-smoke.XXXXXX")"; \
		cd chapter_text_to_speech && \
		$(PYTHON) qwen3_tts_voice_cloning.py --write-sample-data "$$tmpdir" >/dev/null && \
		$(PYTHON) qwen3_tts_voice_cloning.py --score-results "$$tmpdir/results.csv" >/dev/null; \
		status="$$?"; \
		rm -rf "$$tmpdir"; \
		exit "$$status"

object-detection-code-check:
	$(PY_SYNTAX_CHECK) chapter_object_detection/yolo_medical_workflow.py chapter_object_detection/prepare_rsna_yolo_subset.py adl_plot_style.py
	$(PYTHON) -m json.tool chapter_object_detection/yolo_medical_detection_walkthrough.ipynb >/dev/null
	tmpdir="$$(mktemp -d "$(TMPDIR)/adl-object-detection-smoke.XXXXXX")"; \
		mplconfig="$(TMPDIR)/adl-matplotlib-cache"; \
		mkdir -p "$$mplconfig"; \
		cd chapter_object_detection && \
		MPLCONFIGDIR="$$mplconfig" $(PYTHON) yolo_medical_workflow.py smoke --output-dir "$$tmpdir" >/dev/null; \
		status="$$?"; \
		rm -rf "$$tmpdir"; \
		exit "$$status"

segmentation-code-check:
	$(PY_SYNTAX_CHECK) chapter_segmentation_cnn_transformers/segmentation_medical_workflow.py chapter_segmentation_cnn_transformers/segformer_medical_finetune.py adl_plot_style.py
	$(PYTHON) -m json.tool chapter_segmentation_cnn_transformers/segmentation_medical_walkthrough.ipynb >/dev/null
	tmpdir="$$(mktemp -d "$(TMPDIR)/adl-segmentation-smoke.XXXXXX")"; \
		cd chapter_segmentation_cnn_transformers && \
		$(PYTHON) segmentation_medical_workflow.py smoke --output-dir "$$tmpdir/workflow" >/dev/null && \
		$(PYTHON) segformer_medical_finetune.py smoke --output-dir "$$tmpdir/training" >/dev/null; \
		status="$$?"; \
		rm -rf "$$tmpdir"; \
		exit "$$status"

vlm-code-check:
	$(PY_SYNTAX_CHECK) chapter_vision_language_models/answer_parser.py chapter_vision_language_models/vlm_eval.py
	$(PYTHON) -m json.tool chapter_vision_language_models/vlm_reasoning_walkthrough.ipynb >/dev/null
	cd chapter_vision_language_models && $(PYTHON) vlm_eval.py --unit-test-parser >/dev/null
	cd chapter_vision_language_models && $(PYTHON) vlm_eval.py --check-deps --allow-missing-deps >/dev/null
	tmpdir="$$(mktemp -d "$(TMPDIR)/adl-vlm-smoke.XXXXXX")"; \
		cd chapter_vision_language_models && \
		$(PYTHON) vlm_eval.py --write-sample-data "$$tmpdir" >/dev/null && \
		$(PYTHON) vlm_eval.py --preview --prompts "$$tmpdir/prompts.jsonl" >/dev/null; \
		status="$$?"; \
		rm -rf "$$tmpdir"; \
		exit "$$status"
