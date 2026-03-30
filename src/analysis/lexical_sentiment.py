"""Lexicon-based sentiment analysis (Methodology A) for earnings call outlook sections.

This script scans JSON transcripts in data/raw/earnings_calls/, isolates outlook-related
paragraphs, computes a Loughran-McDonald style bullishness score, and appends results to:
    data/processed/lexical_sentiment_results.csv

Usage:
    python src/analysis/lexical_sentiment.py
"""

from __future__ import annotations

import csv
import json
import re
import string
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "earnings_calls"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_CSV = PROCESSED_DIR / "lexical_sentiment_results.csv"

METHOD = "Lexical_LM"

NEGATION_WORDS = {"not", "no", "never", "neither", "cannot"}

# Fallback LM-style stems (used when no external list is provided).
# You can override these with files in data/raw/lm/positive_stems.txt and
# data/raw/lm/negative_stems.txt (one stem per line).
FALLBACK_POSITIVE_STEMS = {
    "achiev",
    "advanc",
    "benefit",
    "confiden",
    "construct",
    "efficien",
    "enhanc",
    "expand",
    "favorable",
    "growth",
    "improv",
    "innov",
    "lead",
    "momentum",
    "opportun",
    "optim",
    "outperform",
    "progress",
    "resilien",
    "strength",
    "strong",
    "success",
    "upside",
}

FALLBACK_NEGATIVE_STEMS = {
    "advers",
    "challeng",
    "constraint",
    "declin",
    "decreas",
    "deterior",
    "difficult",
    "disrupt",
    "downturn",
    "headwind",
    "impair",
    "inflation",
    "loss",
    "negative",
    "pressur",
    "risk",
    "slow",
    "soft",
    "uncertain",
    "volatil",
    "weak",
}

OUTLOOK_START_PATTERN = re.compile(
    r"(outlook|guidance|looking\s+ahead|looking\s+forward|move\s+ahead\s+into\s+the\s+\w+\s+quarter)",
    re.IGNORECASE,
)

OUTLOOK_CONTEXT_PATTERN = re.compile(
    r"(for\s+the\s+(next|first|second|third|fourth)\s+quarter|for\s+fiscal\s+\d{4}|we\s+expect|we\s+anticipate)",
    re.IGNORECASE,
)

QNA_BOUNDARY_PATTERN = re.compile(
    r"(question\s*-?\s*and\s*-?\s*answer|q\s*&\s*a|question\s+and\s+answer\s+session|operator.*first\s+question|open\s+.*questions)",
    re.IGNORECASE,
)

QNA_OPERATOR_PROMPT_PATTERN = re.compile(
    r"(your\s+(first|next|last)\s+question|question\s+comes\s+from|we\s+will\s+now\s+take\s+questions)",
    re.IGNORECASE,
)


def load_stems() -> tuple[set[str], set[str]]:
    """Load LM stems from optional files, else use fallback lists."""
    lm_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "lm"
    pos_file = lm_dir / "positive_stems.txt"
    neg_file = lm_dir / "negative_stems.txt"

    if pos_file.exists() and neg_file.exists():
        pos = {
            ln.strip().lower()
            for ln in pos_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
        neg = {
            ln.strip().lower()
            for ln in neg_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
        print(f"Loaded LM stems from files: +{len(pos)} / -{len(neg)}")
        return pos, neg

    print("LM stem files not found in data/raw/lm; using fallback stem lists.")
    return set(FALLBACK_POSITIVE_STEMS), set(FALLBACK_NEGATIVE_STEMS)


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation/noise, and tokenize into word tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z]+", text)


def stem_match(token: str, stems: set[str]) -> bool:
    """Prefix match token against a set of stems."""
    return any(token.startswith(stem) for stem in stems)


def parse_symbol_and_date(path: Path) -> tuple[str, str] | None:
    """Parse SYMBOL and YYYY-MM-DD from SYMBOL_YYYY-MM-DD.json naming."""
    m = re.match(r"^([A-Z]+)_(\d{4}-\d{2}-\d{2})\.json$", path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


def isolate_outlook_paragraphs(transcript: list[dict]) -> list[tuple[int, str, str, str]]:
    """Extract outlook paragraphs using marker-based start and Q&A boundary end.

    Returns list of tuples: (paragraph_idx, speaker, title, content)
    """
    selected: list[tuple[int, str, str, str]] = []
    in_outlook = False

    for idx, para in enumerate(transcript):
        content = str(para.get("content", "")).strip()
        speaker = str(para.get("speaker", "Unknown")).strip() or "Unknown"
        title = str(para.get("title", "")).strip()
        title_lower = title.lower()
        speaker_lower = speaker.lower()

        if not content:
            continue

        # Once outlook has started, stop when Q&A begins.
        if in_outlook:
            if QNA_BOUNDARY_PATTERN.search(content):
                break
            if speaker_lower == "operator" and QNA_OPERATOR_PROMPT_PATTERN.search(content):
                break
            if "analyst" in title_lower:
                break

        if QNA_BOUNDARY_PATTERN.search(content):
            continue

        is_start = bool(OUTLOOK_START_PATTERN.search(content))
        has_context = bool(OUTLOOK_CONTEXT_PATTERN.search(content))

        if is_start:
            in_outlook = True
            selected.append((idx, speaker, title, content))
            continue

        if in_outlook:
            selected.append((idx, speaker, title, content))
            continue

        # Also allow high-confidence CFO guidance context lines to begin outlook.
        if has_context and ("cfo" in title.lower() or "chief financial officer" in title.lower()):
            in_outlook = True
            selected.append((idx, speaker, title, content))

    return selected


def score_outlook_text(
    outlook_paragraphs: list[tuple[int, str, str, str]],
    positive_stems: set[str],
    negative_stems: set[str],
) -> tuple[float, int, int, int]:
    """Compute bullishness score with negation rule.

    Score = (PosCount - NegCount) / TotalWordsInOutlook
    """
    outlook_text = " ".join(p[3] for p in outlook_paragraphs)
    tokens = tokenize(outlook_text)

    pos_count = 0
    neg_count = 0

    for i, token in enumerate(tokens):
        if stem_match(token, positive_stems):
            window_start = max(0, i - 3)
            prior_window = tokens[window_start:i]
            if any(w in NEGATION_WORDS for w in prior_window):
                neg_count += 1
            else:
                pos_count += 1
        elif stem_match(token, negative_stems):
            neg_count += 1

    total_words = len(tokens)
    score = (pos_count - neg_count) / total_words if total_words > 0 else 0.0
    return score, pos_count, neg_count, total_words


def append_result(symbol: str, reported_date: str, score: float) -> None:
    """Append one sentiment result row to the central CSV file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["symbol", "reported_date", "methodology", "outlook_sentiment_score"])
        writer.writerow([symbol, reported_date, METHOD, f"{score:.8f}"])


def main() -> None:
    positive_stems, negative_stems = load_stems()

    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {RAW_DIR}")
        return

    print(f"Processing {len(json_files)} transcript files from: {RAW_DIR}")

    processed = 0
    for path in json_files:
        parsed = parse_symbol_and_date(path)
        if not parsed:
            print(f"Skipping non-standard filename: {path.name}")
            continue

        symbol, reported_date = parsed

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON {path.name}: {exc}")
            continue

        transcript = payload.get("transcript", [])
        if not isinstance(transcript, list):
            print(f"Skipping {path.name}: transcript is not a list")
            continue

        outlook_paragraphs = isolate_outlook_paragraphs(transcript)

        print(f"\n[{path.name}] Outlook paragraph candidates: {len(outlook_paragraphs)}")
        for idx, speaker, title, content in outlook_paragraphs:
            preview = re.sub(r"\s+", " ", content)[:180]
            print(f"  - Paragraph #{idx} | {speaker} ({title}) | {preview}")

        score, pos_count, neg_count, total_words = score_outlook_text(
            outlook_paragraphs,
            positive_stems,
            negative_stems,
        )

        append_result(symbol, reported_date, score)
        processed += 1
        print(
            f"  => Score={score:.8f} | Pos={pos_count} Neg={neg_count} TotalWords={total_words}"
        )

    print(f"\nDone. Appended results for {processed} files to {OUT_CSV}")


if __name__ == "__main__":
    main()
