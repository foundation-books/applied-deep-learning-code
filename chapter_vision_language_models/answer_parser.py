"""Answer parsing and scoring helpers for the VLM chapter.

The parser keeps two measurements separate:

* exact-match answer quality, after small normalization steps;
* valid-format compliance, meaning exactly one non-empty <answer>...</answer>
  tag is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction


ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
LOOSE_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?"
)
NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")
FRACTION_RE = re.compile(r"([-+]?\d+)\s*/\s*(\d+)")


@dataclass(frozen=True)
class ParseResult:
    """Structured result for one generated completion."""

    raw_text: str
    answer: str | None
    normalized_answer: str | None
    tag_count: int
    valid_format: bool
    parser_error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "tag_count": self.tag_count,
            "valid_format": self.valid_format,
            "parser_error": self.parser_error,
        }


@dataclass(frozen=True)
class ScoreResult:
    """Answer score plus format score for one completion."""

    parsed: ParseResult
    reference_answer: str
    normalized_reference_answer: str
    exact_match: bool
    correctness_reward: float
    format_reward: float

    @property
    def total_reward(self) -> float:
        return self.correctness_reward + self.format_reward

    def to_dict(self) -> dict[str, object]:
        return {
            **self.parsed.to_dict(),
            "reference_answer": self.reference_answer,
            "normalized_reference_answer": self.normalized_reference_answer,
            "exact_match": self.exact_match,
            "correctness_reward": self.correctness_reward,
            "format_reward": self.format_reward,
            "total_reward": self.total_reward,
        }


def _decimal_to_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    if value == value.to_integral_value():
        return str(int(value))
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".")


def normalize_answer(answer: str) -> str:
    """Normalize simple numeric and text answers for exact matching."""

    cleaned = " ".join(answer.strip().strip(".;,").split())
    fraction_match = FRACTION_RE.fullmatch(cleaned)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator == 0:
            return cleaned.lower()
        fraction = Fraction(numerator, denominator)
        return f"{fraction.numerator}/{fraction.denominator}"

    if NUMERIC_RE.fullmatch(cleaned):
        try:
            return _decimal_to_text(Decimal(cleaned.replace(",", "")))
        except InvalidOperation:
            return cleaned.lower()

    return cleaned.lower()


def _extract_untagged_answer(text: str) -> str | None:
    """Extract a best-effort answer from text without valid answer tags."""

    stripped = text.strip()
    if not stripped:
        return None
    numbers = LOOSE_NUMBER_RE.findall(stripped)
    if numbers:
        return numbers[-1].strip()
    return stripped.strip(".;,") or None


def parse_answer(text: str, allow_untagged: bool = True) -> ParseResult:
    """Parse a completion according to the chapter's answer-tag contract."""

    matches = ANSWER_TAG_RE.findall(text)
    tag_count = len(matches)

    if tag_count == 1:
        answer = matches[0].strip()
        if not answer:
            return ParseResult(
                raw_text=text,
                answer=None,
                normalized_answer=None,
                tag_count=tag_count,
                valid_format=False,
                parser_error="empty_answer",
            )
        return ParseResult(
            raw_text=text,
            answer=answer,
            normalized_answer=normalize_answer(answer),
            tag_count=tag_count,
            valid_format=True,
            parser_error=None,
        )

    if tag_count > 1:
        first_answer = matches[0].strip() or None
        return ParseResult(
            raw_text=text,
            answer=first_answer,
            normalized_answer=normalize_answer(first_answer) if first_answer else None,
            tag_count=tag_count,
            valid_format=False,
            parser_error="multiple_answer_tags",
        )

    answer = _extract_untagged_answer(text) if allow_untagged else None
    return ParseResult(
        raw_text=text,
        answer=answer,
        normalized_answer=normalize_answer(answer) if answer else None,
        tag_count=0,
        valid_format=False,
        parser_error="missing_answer_tag",
    )


def score_completion(
    completion: str,
    reference_answer: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
) -> ScoreResult:
    """Score one VLM completion for exact answer and format validity."""

    parsed = parse_answer(completion)
    normalized_reference = normalize_answer(reference_answer)
    exact_match = (
        parsed.normalized_answer is not None
        and parsed.normalized_answer == normalized_reference
    )
    return ScoreResult(
        parsed=parsed,
        reference_answer=reference_answer,
        normalized_reference_answer=normalized_reference,
        exact_match=exact_match,
        correctness_reward=correctness_weight if exact_match else 0.0,
        format_reward=format_weight if parsed.valid_format else 0.0,
    )


def unit_test_parser() -> None:
    """Run the parser cases required by the homework scaffold."""

    cases = [
        ("<answer>42</answer>", "42", True, True, None),
        ("The answer is 42.", "42", True, False, "missing_answer_tag"),
        ("<answer>41</answer>", "42", False, True, None),
        (
            "<answer>42</answer> and maybe <answer>41</answer>",
            "42",
            True,
            False,
            "multiple_answer_tags",
        ),
        ("<answer></answer>", "42", False, False, "empty_answer"),
        ("<answer> 4.0 </answer>", "4", True, True, None),
        ("<answer>1/2</answer>", "1/2", True, True, None),
        ("<ANSWER> Twelve </ANSWER>", "twelve", True, True, None),
        ("Final: <answer>42", "42", True, False, "missing_answer_tag"),
    ]

    for completion, reference, expected_exact, expected_format, expected_error in cases:
        score = score_completion(completion, reference)
        assert score.exact_match is expected_exact, completion
        assert score.parsed.valid_format is expected_format, completion
        assert score.parsed.parser_error == expected_error, completion


if __name__ == "__main__":
    unit_test_parser()
    print("parser unit tests passed")
