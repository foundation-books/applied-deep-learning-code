# RAG Reference Artifacts

These CSV files are the compact reference summaries cited by the RAG chapter
table. They were produced by `rag_reference_experiments.py` on the sample corpus
with top-3 retrieval, eight held-out questions, MiniLM retrieval, and
`Qwen/Qwen2.5-0.5B-Instruct` generation.

The heavier runner writes generated outputs to `artifacts/rag-reference/` by
default. That directory is ignored by Git, so these checked-in summaries give
readers a stable copy of the reference evidence without requiring a GPU rerun.
