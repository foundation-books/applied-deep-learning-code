#!/usr/bin/env python3
"""Small retrieval-augmented course assistant scaffold.

The default run is deliberately local and dependency-light. It uses a small
sample corpus, lexical retrieval, deterministic hashed dense vectors, hybrid
retrieval, and an extractive citation-based answer writer. The scaffold is meant
to make retrieval diagnostics, citations, answer scoring, and latency visible
before students replace any component with a heavier model or service.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9_']+")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
CORPUS_SUFFIXES = {".md", ".txt", ".tex"}
OPTIONAL_PACKAGES = ["numpy"]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    chunk_id: str
    source_path: str
    version: str
    start_token: int
    end_token: int
    text: str


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    gold_doc_id: str
    gold_terms: tuple[str, ...]
    grading_notes: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stable_id(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return normalized or "document"


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_status() -> dict[str, bool]:
    return {name: package_available(name) for name in OPTIONAL_PACKAGES}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def first_heading_or_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            return match.group(1).strip()
    return path.stem.replace("_", " ").title()


def section_texts(title: str, text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = title
    buffer: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
            current_heading = match.group(1).strip() or title
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))
    return [(heading, body) for heading, body in sections if tokenize(body)]


def chunk_tokens(tokens: list[str], chunk_tokens_count: int, overlap_tokens: int) -> list[tuple[int, int]]:
    if chunk_tokens_count <= 0:
        raise ValueError("--chunk-tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("--overlap-tokens must be nonnegative")
    if overlap_tokens >= chunk_tokens_count:
        raise ValueError("--overlap-tokens must be smaller than --chunk-tokens")

    step = chunk_tokens_count - overlap_tokens
    spans: list[tuple[int, int]] = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_tokens_count, len(tokens))
        if start >= end:
            break
        spans.append((start, end))
        if end == len(tokens):
            break
    return spans


def load_corpus(
    corpus_dir: Path,
    chunk_tokens_count: int,
    overlap_tokens: int,
    version: str,
) -> list[Chunk]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    chunks: list[Chunk] = []
    for path in sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in CORPUS_SUFFIXES):
        if any(part.startswith(".") for part in path.parts):
            continue
        raw_text = path.read_text(encoding="utf-8")
        title = first_heading_or_title(path, raw_text)
        doc_id = stable_id(path.stem)
        chunk_index = 1
        for section, body in section_texts(title, raw_text):
            tokens = tokenize(body)
            for start, end in chunk_tokens(tokens, chunk_tokens_count, overlap_tokens):
                chunk_id = f"{doc_id}:c{chunk_index:03d}"
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        title=title,
                        section=section,
                        chunk_id=chunk_id,
                        source_path=str(path),
                        version=version,
                        start_token=start,
                        end_token=end,
                        text=" ".join(tokens[start:end]),
                    )
                )
                chunk_index += 1
    if not chunks:
        raise ValueError(f"No chunkable Markdown, text, or TeX files found in {corpus_dir}")
    return chunks


def read_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            question = str(record.get("question", "")).strip()
            if not question:
                raise ValueError(f"{path}:{line_number} missing nonempty question")
            raw_terms = record.get("gold_terms", [])
            if isinstance(raw_terms, str):
                gold_terms = tuple(tokenize(raw_terms))
            else:
                gold_terms = tuple(str(term).lower() for term in raw_terms)
            questions.append(
                Question(
                    question_id=str(record.get("question_id", f"q{line_number:03d}")),
                    question=question,
                    gold_doc_id=str(
                        record.get("gold_doc_id", record.get("expected_source_id", ""))
                    ),
                    gold_terms=gold_terms,
                    grading_notes=str(record.get("grading_notes", "")),
                )
            )
    if not questions:
        raise ValueError(f"No question records found in {path}")
    return questions


def chunk_dict(chunk: Chunk) -> dict[str, object]:
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "chunk_id": chunk.chunk_id,
        "source_path": chunk.source_path,
        "version": chunk.version,
        "start_token": chunk.start_token,
        "end_token": chunk.end_token,
        "text": chunk.text,
    }


def build_token_indexes(chunks: list[Chunk]) -> tuple[dict[str, Counter[str]], Counter[str], float]:
    chunk_terms: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []
    for chunk in chunks:
        counts = Counter(tokenize(chunk.text))
        chunk_terms[chunk.chunk_id] = counts
        lengths.append(sum(counts.values()))
        for term in counts:
            document_frequency[term] += 1
    average_length = sum(lengths) / len(lengths)
    return chunk_terms, document_frequency, average_length


def bm25_scores(
    query_tokens: list[str],
    chunks: list[Chunk],
    chunk_terms: dict[str, Counter[str]],
    document_frequency: Counter[str],
    average_length: float,
) -> dict[str, float]:
    n_chunks = len(chunks)
    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    for chunk in chunks:
        counts = chunk_terms[chunk.chunk_id]
        length = max(sum(counts.values()), 1)
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * length / average_length)
            score += idf * (tf * (k1 + 1.0)) / denominator
        scores[chunk.chunk_id] = score
    return scores


def hashed_vector(tokens: list[str], dimension: int) -> list[float]:
    if dimension <= 0:
        raise ValueError("--dense-dim must be positive")
    vector = [0.0] * dimension
    for token, count in Counter(tokens).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimension
        sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def dense_scores(
    query_tokens: list[str],
    chunks: list[Chunk],
    chunk_vectors: dict[str, list[float]],
    dense_dim: int,
) -> dict[str, float]:
    query_vector = hashed_vector(query_tokens, dense_dim)
    return {chunk.chunk_id: dot(query_vector, chunk_vectors[chunk.chunk_id]) for chunk in chunks}


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if math.isclose(minimum, maximum):
        return {key: 0.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def rank_chunks(
    chunks: list[Chunk],
    scores: dict[str, float],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(by_id[chunk_id], scores[chunk_id]) for chunk_id in ranked_ids[:top_k]]


def query_coverage_score(query_tokens: list[str], chunk: Chunk) -> float:
    chunk_token_set = set(tokenize(chunk.text))
    query_token_set = set(query_tokens)
    if not query_token_set:
        return 0.0
    unigram_coverage = len(query_token_set & chunk_token_set) / len(query_token_set)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    chunk_tokens = tokenize(chunk.text)
    chunk_bigrams = set(zip(chunk_tokens, chunk_tokens[1:]))
    if not query_bigrams:
        return unigram_coverage
    bigram_coverage = len(query_bigrams & chunk_bigrams) / len(query_bigrams)
    return 0.8 * unigram_coverage + 0.2 * bigram_coverage


def retrieve(
    method: str,
    question: Question,
    chunks: list[Chunk],
    chunk_terms: dict[str, Counter[str]],
    document_frequency: Counter[str],
    average_length: float,
    chunk_vectors: dict[str, list[float]],
    dense_dim: int,
    top_k: int,
    candidate_k: int,
) -> list[tuple[Chunk, float]]:
    query_tokens = tokenize(question.question)
    lexical = bm25_scores(query_tokens, chunks, chunk_terms, document_frequency, average_length)
    dense = dense_scores(query_tokens, chunks, chunk_vectors, dense_dim)

    if method == "lexical":
        return rank_chunks(chunks, lexical, top_k)
    if method == "dense":
        return rank_chunks(chunks, dense, top_k)

    lexical_norm = normalize(lexical)
    dense_norm = normalize(dense)
    hybrid = {
        chunk.chunk_id: 0.5 * lexical_norm[chunk.chunk_id] + 0.5 * dense_norm[chunk.chunk_id]
        for chunk in chunks
    }
    if method == "hybrid":
        return rank_chunks(chunks, hybrid, top_k)
    if method == "rerank":
        candidates = rank_chunks(chunks, hybrid, max(candidate_k, top_k))
        reranked = {
            chunk.chunk_id: 0.7 * hybrid[chunk.chunk_id] + 0.3 * query_coverage_score(query_tokens, chunk)
            for chunk, _ in candidates
        }
        return rank_chunks([chunk for chunk, _ in candidates], reranked, top_k)
    raise ValueError(f"Unknown retrieval method: {method}")


def is_acceptable(question: Question, chunk: Chunk) -> bool:
    if question.gold_doc_id and chunk.doc_id == question.gold_doc_id:
        return True
    if question.gold_terms:
        text = chunk.text.lower()
        return all(term.lower() in text for term in question.gold_terms)
    return False


def retrieval_metrics(rows: list[dict[str, object]], questions: list[Question]) -> dict[str, object]:
    by_method: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method[str(row["method"])][str(row["question_id"])].append(row)

    summary: dict[str, object] = {}
    for method, method_rows in by_method.items():
        hits = 0
        reciprocal_ranks: list[float] = []
        latencies: list[float] = []
        for question in questions:
            ranked_rows = sorted(method_rows[question.question_id], key=lambda row: int(row["rank"]))
            latencies.extend(float(row["retrieval_latency_ms"]) for row in ranked_rows[:1])
            first_rank = 0
            for row in ranked_rows:
                if bool(row["acceptable"]):
                    first_rank = int(row["rank"])
                    break
            if first_rank:
                hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)
        summary[method] = {
            "recall_at_k": hits / len(questions),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "retrieval_latency_ms_p50": percentile(latencies, 50),
            "retrieval_latency_ms_p95": percentile(latencies, 95),
        }
    return summary


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil((percent / 100.0) * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def citation_ids(answer: str) -> list[str]:
    return re.findall(r"\[([a-z0-9_:-]+)\]", answer.lower())


def prompt_only_answer(question: Question) -> str:
    return (
        "A prompt-only baseline can give general applied advice, but it has no "
        "retrieved course evidence. It should suggest checking the validation "
        "protocol, changing one factor at a time, and avoiding final test-set "
        "tuning when discussing model improvement."
    )


def rag_answer(question: Question, retrieved: list[tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "The retrieved sources are insufficient, so I cannot give a grounded answer."
    primary = retrieved[0][0]
    secondary = retrieved[1][0] if len(retrieved) > 1 else primary
    primary_words = primary.text.split()[:34]
    secondary_words = secondary.text.split()[:24]
    return (
        f"According to [{primary.chunk_id}], {' '.join(primary_words)}. "
        f"The supporting context in [{secondary.chunk_id}] adds that "
        f"{' '.join(secondary_words)}. Use these sources to answer the question "
        "and state when the retrieved evidence is insufficient."
    )


def score_answer(
    answer: str,
    question: Question,
    retrieved_chunks: list[Chunk],
) -> dict[str, object]:
    answer_lower = answer.lower()
    term_matches = sum(1 for term in question.gold_terms if term.lower() in answer_lower)
    term_fraction = term_matches / len(question.gold_terms) if question.gold_terms else 0.0
    cited_ids = citation_ids(answer)
    retrieved_ids = {chunk.chunk_id.lower() for chunk in retrieved_chunks}
    retrieved_by_id = {chunk.chunk_id.lower(): chunk for chunk in retrieved_chunks}
    cited_chunks = [retrieved_by_id[cited_id] for cited_id in cited_ids if cited_id in retrieved_by_id]

    technical_correctness = 4.0 * term_fraction
    uses_retrieved_evidence = 2.0 if cited_chunks else 0.0
    correct_source_cited = 2.0 if any(is_acceptable(question, chunk) for chunk in cited_chunks) else 0.0
    word_count = len(answer.split())
    concision = 1.0 if word_count <= 160 else 0.5
    source_awareness = 1.0 if cited_chunks or "insufficient" in answer_lower else 0.0
    total = technical_correctness + uses_retrieved_evidence + correct_source_cited + concision + source_awareness

    return {
        "score": round(total, 3),
        "technical_correctness": round(technical_correctness, 3),
        "uses_retrieved_evidence": uses_retrieved_evidence,
        "citation_correctness": correct_source_cited,
        "concision": concision,
        "source_awareness": source_awareness,
        "matched_gold_terms": term_matches,
        "gold_terms": list(question.gold_terms),
        "citations": cited_ids,
        "all_citations_in_context": all(cited_id in retrieved_ids for cited_id in cited_ids),
    }


def retrieval_row(
    question: Question,
    method: str,
    rank: int,
    chunk: Chunk,
    score: float,
    retrieval_latency_ms: float,
) -> dict[str, object]:
    return {
        "question_id": question.question_id,
        "question": question.question,
        "method": method,
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "score": round(score, 6),
        "acceptable": is_acceptable(question, chunk),
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "text_excerpt": " ".join(chunk.text.split()[:36]),
    }


def answer_row(
    question: Question,
    run_name: str,
    method: str,
    answer: str,
    retrieved: list[tuple[Chunk, float]],
    generation_latency_ms: float,
) -> dict[str, object]:
    retrieved_chunks = [chunk for chunk, _ in retrieved]
    scores = score_answer(answer, question, retrieved_chunks)
    return {
        "question_id": question.question_id,
        "question": question.question,
        "run_name": run_name,
        "retrieval_method": method,
        "answer": answer,
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks],
        "generation_latency_ms": round(generation_latency_ms, 3),
        **scores,
    }


def summarize_answers(rows: list[dict[str, object]]) -> dict[str, object]:
    by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run_name"])].append(row)
    summary: dict[str, object] = {}
    for run_name, run_rows in by_run.items():
        scores = [float(row["score"]) for row in run_rows]
        latencies = [float(row["generation_latency_ms"]) for row in run_rows]
        faithful = [bool(row["all_citations_in_context"]) for row in run_rows if row["citations"]]
        summary[run_name] = {
            "average_answer_score": sum(scores) / len(scores),
            "generation_latency_ms_p50": percentile(latencies, 50),
            "generation_latency_ms_p95": percentile(latencies, 95),
            "faithful_citation_fraction": (
                sum(1 for value in faithful if value) / len(faithful) if faithful else 0.0
            ),
        }
    return summary


def preview_corpus(args: argparse.Namespace) -> None:
    chunks = load_corpus(
        args.corpus_dir,
        args.chunk_tokens,
        args.overlap_tokens,
        args.corpus_version,
    )
    for chunk in chunks[: args.preview_count]:
        print(json.dumps(chunk_dict(chunk), sort_keys=True))
    print(f"Previewed {min(len(chunks), args.preview_count)} of {len(chunks)} chunks.")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    chunks = load_corpus(
        args.corpus_dir,
        args.chunk_tokens,
        args.overlap_tokens,
        args.corpus_version,
    )
    questions = read_questions(args.questions)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    for required_method in (args.rag_method, args.controlled_method):
        if required_method and required_method not in methods:
            methods.append(required_method)

    chunk_terms, document_frequency, average_length = build_token_indexes(chunks)
    chunk_vectors = {
        chunk.chunk_id: hashed_vector(tokenize(chunk.text), args.dense_dim)
        for chunk in chunks
    }

    retrieval_rows: list[dict[str, object]] = []
    retrieved_by_question_method: dict[tuple[str, str], list[tuple[Chunk, float]]] = {}
    for question in questions:
        for method in methods:
            start = time.perf_counter()
            retrieved = retrieve(
                method,
                question,
                chunks,
                chunk_terms,
                document_frequency,
                average_length,
                chunk_vectors,
                args.dense_dim,
                args.top_k,
                args.candidate_k,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            retrieved_by_question_method[(question.question_id, method)] = retrieved
            for rank, (chunk, score) in enumerate(retrieved, start=1):
                retrieval_rows.append(retrieval_row(question, method, rank, chunk, score, elapsed_ms))

    answer_rows: list[dict[str, object]] = []
    for question in questions:
        start = time.perf_counter()
        baseline = prompt_only_answer(question)
        baseline_elapsed = (time.perf_counter() - start) * 1000.0
        answer_rows.append(
            answer_row(question, "prompt_only", "none", baseline, [], baseline_elapsed)
        )

        for run_name, method in (
            ("rag_baseline", args.rag_method),
            ("controlled_improvement", args.controlled_method),
        ):
            retrieved = retrieved_by_question_method[(question.question_id, method)]
            start = time.perf_counter()
            answer = rag_answer(question, retrieved)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            answer_rows.append(answer_row(question, run_name, method, answer, retrieved, elapsed_ms))

    summary = {
        "settings": {
            "corpus_dir": str(args.corpus_dir),
            "questions": str(args.questions),
            "corpus_version": args.corpus_version,
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "dense_encoder": f"hashing-bow-{args.dense_dim}",
            "methods": methods,
            "rag_method": args.rag_method,
            "controlled_method": args.controlled_method,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "optional_dependencies": dependency_status(),
        },
        "corpus": {
            "documents": len({chunk.doc_id for chunk in chunks}),
            "chunks": len(chunks),
            "average_chunk_tokens": sum(len(tokenize(chunk.text)) for chunk in chunks) / len(chunks),
        },
        "retrieval": retrieval_metrics(retrieval_rows, questions),
        "answers": summarize_answers(answer_rows),
    }

    if args.save_artifacts:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.artifact_dir / "chunks.jsonl", [chunk_dict(chunk) for chunk in chunks])
        write_jsonl(args.artifact_dir / "retrieval_results.jsonl", retrieval_rows)
        write_jsonl(args.artifact_dir / "answers.jsonl", answer_rows)
        write_json(args.artifact_dir / "run_summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--corpus-dir", type=Path, default=here / "sample_corpus")
    parser.add_argument("--questions", type=Path, default=here / "sample_questions.jsonl")
    parser.add_argument("--artifact-dir", type=Path, default=here / "artifacts")
    parser.add_argument("--corpus-version", default="sample-v1")
    parser.add_argument("--chunk-tokens", type=int, default=80)
    parser.add_argument("--overlap-tokens", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--dense-dim", type=int, default=64)
    parser.add_argument("--methods", default="lexical,dense,hybrid")
    parser.add_argument("--rag-method", default="lexical")
    parser.add_argument("--controlled-method", default="hybrid")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--run", action="store_true", help="Run retrieval, answer scoring, and latency measurement.")
    parser.add_argument("--preview-corpus", action="store_true", help="Print chunk records without running retrieval.")
    parser.add_argument("--save-artifacts", action="store_true", help="Write chunks, retrieval results, answers, and summary JSON.")
    parser.add_argument("--check-deps", action="store_true", help="Print optional dependency status and exit.")
    parser.add_argument("--allow-missing-deps", action="store_true", help="Accepted for repository smoke checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_deps:
        print(json.dumps({"optional_dependencies": dependency_status()}, indent=2, sort_keys=True))
        return 0
    if args.preview_corpus:
        preview_corpus(args)
        return 0
    if args.run or not any((args.preview_corpus, args.check_deps)):
        run_experiment(args)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
