"""Reward helpers for the RL-for-LLM-training homework scaffold."""

from __future__ import annotations

import re
from dataclasses import dataclass


STRICT_ANSWER_RE = re.compile(r"^\s*<answer>\s*([^<]*?)\s*</answer>\s*$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class RewardBreakdown:
    correctness: float
    format_score: float
    length_penalty: float

    @property
    def total(self) -> float:
        return self.correctness + self.format_score + self.length_penalty

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "format_score": self.format_score,
            "length_penalty": self.length_penalty,
            "total": self.total,
        }


def extract_answer(text: str) -> str | None:
    """Extract an answer only when the completion is exactly one answer tag."""
    match = STRICT_ANSWER_RE.fullmatch(text)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer or None


def has_strict_answer_format(text: str) -> bool:
    """Return whether text contains exactly one nonempty answer tag and no prose."""
    return extract_answer(text) is not None


def score_completion(completion: str, expected_answer: str, max_tokens: int = 80) -> RewardBreakdown:
    """Score one completion with correctness, format, and length components."""
    extracted = extract_answer(completion)
    correctness = 1.0 if extracted == expected_answer else 0.0
    format_score = 0.2 if has_strict_answer_format(completion) else 0.0
    token_count = len(completion.split())
    length_penalty = -0.01 * max(0, token_count - max_tokens)
    return RewardBreakdown(
        correctness=correctness,
        format_score=format_score,
        length_penalty=length_penalty,
    )
