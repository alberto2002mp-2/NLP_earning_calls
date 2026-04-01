"""Lexicon-based sentiment analysis (Methodology A) for earnings call outlook sections.

This script scans JSON transcripts in data/raw/earnings_calls/**, isolates outlook-related
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
    r"(outlook|guidance|forecast|projection|looking\s+(ahead|forward)|move\s+ahead\s+into\s+the\s+\w+\s+quarter)",
    re.IGNORECASE,
)

OUTLOOK_CONTEXT_PATTERN = re.compile(
    r"(for\s+the\s+(next|first|second|third|fourth)\s+quarter|fiscal\s+\d{4}|we\s+expect|we\s+anticipate|between\s+\d+\s+and\s+\d+)",
    re.IGNORECASE,
)

QNA_BOUNDARY_PATTERN = re.compile(
    r"(question\s*-?\s*and\s*-?\s*answer|q\s*&\s*a|operator.*first\s+question|open\s+the\s+call\s+to\s+questions)",
    re.IGNORECASE,
)


def _is_cfo_title(title: str) -> bool:
    """Return True if speaker title indicates CFO role."""
    t = title.lower()
    # Match acronym and common long-form variants.
    if re.search(r"\bcfo\b", t):
        return True
    if "chief financial officer" in t:
        return True
    if "cfo" in t:
        return True
    if "finance chief" in t:
        return True
    return False


def _is_ceo_title(title: str) -> bool:
    """Return True if speaker title indicates CEO role."""
    t = title.lower()
    return "ceo" in t or "chief executive officer" in t


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


def _collect_outlook_for_role(
    transcript: list[dict],
    role: str,
    context_grace_paragraphs: int,
) -> dict:
    """Collect outlook paragraphs with robust confirmation and boundary slicing.

    States:
    - WAITING: wait for role speaker
    - DISCOVERING_CONFIRMATION: start found; wait for context within grace window
    - COLLECTING: include paragraphs until analyst/Q&A boundary
    - STOPPED: terminate
    """
    selected: list[tuple[int, str, str, str]] = []
    pending: list[tuple[int, str, str, str]] = []

    state = "WAITING"
    confirmation_deadline = -1

    has_role_speaker = False
    start_matched = False
    context_confirmed = False
    boundary_collision = False

    for idx, para in enumerate(transcript):
        content = str(para.get("content", "")).strip()
        speaker = str(para.get("speaker", "Unknown")).strip() or "Unknown"
        title = str(para.get("title", "")).strip()
        title_lower = title.lower()

        if not content:
            continue

        is_role = _is_cfo_title(title) if role == "CFO" else _is_ceo_title(title)
        has_role_speaker = has_role_speaker or is_role

        start_match = OUTLOOK_START_PATTERN.search(content)
        context_matches = len(list(OUTLOOK_CONTEXT_PATTERN.finditer(content)))
        boundary_match = QNA_BOUNDARY_PATTERN.search(content)
        is_analyst = "analyst" in title_lower

        # Resilient slicing: if a role paragraph has both start and boundary,
        # collect only text before boundary and stop immediately.
        if is_role and start_match and boundary_match:
            prefix = content[: boundary_match.start()].strip()
            if prefix:
                start_matched = True
                context_confirmed = context_confirmed or context_matches >= 1
                selected.append((idx, speaker, title, prefix))
                state = "STOPPED"
                break

        if state == "WAITING":
            if is_role and start_match:
                start_matched = True
                pending.append((idx, speaker, title, content))
                confirmation_deadline = idx + context_grace_paragraphs

                # CEO fallback rule: confirm when context appears in same paragraph
                # or the immediate subsequent paragraph.
                if role == "CEO" and context_matches >= 1:
                    context_confirmed = True
                    selected.extend(pending)
                    pending.clear()
                    state = "COLLECTING"
                elif role == "CFO" and context_matches >= 1:
                    context_confirmed = True
                    selected.extend(pending)
                    pending.clear()
                    state = "COLLECTING"
                else:
                    state = "DISCOVERING_CONFIRMATION"
            continue

        if state == "DISCOVERING_CONFIRMATION":
            # Boundary before confirmation means collision only if we cannot slice.
            if is_analyst or boundary_match:
                # Recall safeguard: preserve pending start paragraphs even if
                # Q&A begins before explicit context confirmation.
                if pending:
                    selected.extend(pending)
                    pending.clear()
                else:
                    boundary_collision = start_matched and not context_confirmed
                state = "STOPPED"
                break

            pending.append((idx, speaker, title, content))

            if context_matches >= 1:
                context_confirmed = True
                selected.extend(pending)
                pending.clear()
                state = "COLLECTING"
                continue

            if idx >= confirmation_deadline:
                # Context not confirmed in grace window, reset.
                pending.clear()
                state = "WAITING"
                continue

        if state == "COLLECTING":
            if is_analyst or boundary_match:
                state = "STOPPED"
                break
            selected.append((idx, speaker, title, content))
            continue

    return {
        "selected": selected,
        "has_role_speaker": has_role_speaker,
        "start_matched": start_matched,
        "context_confirmed": context_confirmed,
        "boundary_collision": boundary_collision,
    }


def isolate_outlook_paragraphs(
    transcript: list[dict],
) -> tuple[list[tuple[int, str, str, str]], str, str]:
    """Hierarchical outlook isolation with CFO-first and CEO fallback.

    Returns:
        (outlook_paragraphs, source_speaker_title, status_note)
        source_speaker_title in {"CFO", "CEO", "None"}
    """
    cfo_result = _collect_outlook_for_role(
        transcript,
        role="CFO",
        context_grace_paragraphs=3,
    )
    if cfo_result["selected"]:
        return cfo_result["selected"], "CFO", "Success_CFO"

    # CEO fallback: require start and context in same or subsequent paragraph.
    ceo_result = _collect_outlook_for_role(
        transcript,
        role="CEO",
        context_grace_paragraphs=1,
    )
    if ceo_result["selected"]:
        return ceo_result["selected"], "CEO", "Success_CEO"

    any_speaker = cfo_result["has_role_speaker"] or ceo_result["has_role_speaker"]
    any_start = cfo_result["start_matched"] or ceo_result["start_matched"]
    any_context = cfo_result["context_confirmed"] or ceo_result["context_confirmed"]
    any_boundary_collision = (
        cfo_result["boundary_collision"] or ceo_result["boundary_collision"]
    )

    if not any_speaker:
        status_note = "Fail_No_Speaker"
    elif any_boundary_collision:
        status_note = "Fail_Boundary_Collision"
    elif not any_start:
        status_note = "Fail_No_Start"
    elif not any_context:
        status_note = "Fail_No_Context"
    else:
        status_note = "Fail_No_Context"

    return [], "None", status_note


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


def _ensure_output_header() -> None:
    """Ensure CSV header contains source_speaker_title and status_note."""
    expected_header = [
        "symbol",
        "reported_date",
        "methodology",
        "outlook_sentiment_score",
        "source_speaker_title",
        "status_note",
    ]

    if not OUT_CSV.exists():
        return

    rows = list(csv.reader(OUT_CSV.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return

    header = rows[0]
    if header == expected_header:
        return

    migrated_rows = []
    for row in rows[1:]:
        if len(row) >= 6:
            migrated_rows.append(row[:6])
        elif len(row) == 5:
            migrated_rows.append(row + ["Unknown"])
        elif len(row) == 4:
            migrated_rows.append(row + ["Unknown", "Unknown"])
        else:
            continue

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(expected_header)
        writer.writerows(migrated_rows)


def append_result(
    symbol: str,
    reported_date: str,
    score: float,
    source_speaker_title: str,
    status_note: str,
) -> None:
    """Append one sentiment result row to the central CSV file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "symbol",
                    "reported_date",
                    "methodology",
                    "outlook_sentiment_score",
                    "source_speaker_title",
                    "status_note",
                ]
            )
        writer.writerow(
            [
                symbol,
                reported_date,
                METHOD,
                f"{score:.8f}",
                source_speaker_title,
                status_note,
            ]
        )


def main() -> None:
    positive_stems, negative_stems = load_stems()
    _ensure_output_header()

    json_files = sorted(RAW_DIR.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {RAW_DIR}")
        return

    print(f"Processing {len(json_files)} transcript files from: {RAW_DIR} (recursive)")

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

        outlook_paragraphs, source_title, status_note = isolate_outlook_paragraphs(transcript)

        print(f"Company {symbol} - Outlook found via {source_title} ({status_note})")

        print(f"\n[{path.name}] Outlook paragraph candidates: {len(outlook_paragraphs)}")
        for idx, speaker, title, content in outlook_paragraphs:
            preview = re.sub(r"\s+", " ", content)[:180]
            print(f"  - Paragraph #{idx} | {speaker} ({title}) | {preview}")

        score, pos_count, neg_count, total_words = score_outlook_text(
            outlook_paragraphs,
            positive_stems,
            negative_stems,
        )

        append_result(symbol, reported_date, score, source_title, status_note)
        processed += 1
        print(
            f"  => Score={score:.8f} | Pos={pos_count} Neg={neg_count} TotalWords={total_words}"
        )

    print(f"\nDone. Appended results for {processed} files to {OUT_CSV}")


if __name__ == "__main__":
    main()
