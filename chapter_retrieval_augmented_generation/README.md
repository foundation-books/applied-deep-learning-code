# Retrieval-Augmented Generation Companion Code

This directory contains the reader-facing scaffold for the retrieval-augmented
generation chapter homework. It is intentionally dependency-light: the default
run uses only the Python standard library, a small sample corpus, lexical
retrieval, deterministic hashed dense vectors, hybrid retrieval, and an
extractive citation-based answer writer. This makes the retrieval diagnostics
useful even on a CPU-only machine with no API keys.

The main script is:

```text
rag_course_assistant.py
```

The interactive walkthrough is:

```text
rag_course_assistant_walkthrough.ipynb
```

Open the notebook from a repository checkout in Jupyter, Colab, or Kaggle when
you want to follow the homework one step at a time: inspect the sample corpus,
chunk the documents, run lexical and dense retrieval, generate citation-based
RAG answers, and read the saved artifacts. The notebook intentionally has no
saved outputs and calls `rag_course_assistant.py` instead of duplicating the
implementation.

Use the `.py` script for repeatable command-line runs, dependency checks, and
repository smoke tests.

The heavier reference runner is:

```text
rag_reference_experiments.py
```

It is intended for chapter artifact generation on a reviewed local GPU runtime,
where small Transformer encoders and a local generator can measure neural
retrieval, reranking, generation latency, prompt-injection behavior, index
mismatch, and approximate vector-search scaling. Keep the reader-facing homework
scaffold in `rag_course_assistant.py`; use this heavier runner only for
book-facing reference evidence.

The committed reference CSV summaries cited by the chapter are included in this
companion-code directory:

```text
reference_artifacts/rag-reference-retrieval-summary.csv
reference_artifacts/rag-reference-generation-summary.csv
```

When you rerun `rag_reference_experiments.py`, its default output directory is under
this chapter's ignored `artifacts/` directory. Generated CSV/JSONL files will not
appear as tracked files unless you intentionally copy them
elsewhere. Generated artifact filenames use the neutral `rag-reference-*`
prefix for companion-code consistency.

Generation does not trust Hugging Face remote model code by default. Pass
`--trust-remote-code` only after reviewing the selected model repository.

Repository checks can run:

```sh
python rag_course_assistant.py --check-deps --allow-missing-deps
```

From the companion-code root, `make rag-code-check` also parses both RAG Python
files without writing `__pycache__` bytecode.

Run the sample end-to-end experiment:

```sh
python rag_course_assistant.py --run --save-artifacts
```

Preview the chunk records before running retrieval:

```sh
python rag_course_assistant.py --preview-corpus
```

Run a controlled retrieval comparison with a different chunking rule:

```sh
python rag_course_assistant.py \
  --run \
  --chunk-tokens 60 \
  --overlap-tokens 15 \
  --top-k 4 \
  --rag-method dense \
  --controlled-method hybrid \
  --save-artifacts \
  --artifact-dir artifacts/chunk60_top4
```

The default sample inputs are:

```text
sample_corpus/*.md
sample_questions.jsonl
```

For the homework, replace those files with a small course corpus and a held-out
question set. Question records should contain a `question_id`, `question`,
`gold_doc_id`, and `gold_terms` list. The script writes `chunks.jsonl`,
`retrieval_results.jsonl`, `answers.jsonl`, and `run_summary.json` when
`--save-artifacts` is supplied. Use those measured values to fill the chapter
homework table.

The default script does not read `.env`. If an instructor allows students to
replace the extractive answer writer with an API-based generator, `.env` should
contain only secrets such as API keys. Corpus files, prompts, questions,
retrieval outputs, and scores should remain ordinary tracked or ignored data
files, not environment variables.
