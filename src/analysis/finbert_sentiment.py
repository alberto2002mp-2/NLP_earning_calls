"""Transformer-based sentiment analysis (Methodology B) for earnings call outlook sections.

This script scans JSON transcripts in data/raw/earnings_calls/, isolates outlook-related
paragraphs using the same high-precision logic as lexical_sentiment.py, scores each
sentence with ProsusAI/finbert, and writes results to:
    data/processed/finbert_sentiment_results.csv

Usage:
    python src/analysis/finbert_sentiment.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "earnings_calls"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_CSV = PROCESSED_DIR / "finbert_sentiment_results.csv"
OUT_STANDARDIZED_CSV = PROCESSED_DIR / "finbert_sentiment_standardized.csv"

METHOD = "FinBERT"
MODEL_NAME = "ProsusAI/finbert"
DEFAULT_SEED = 42

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

        # Same-paragraph safeguard: preserve outlook text right before Q&A boundary.
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

                if context_matches >= 1:
                    context_confirmed = True
                    selected.extend(pending)
                    pending.clear()
                    state = "COLLECTING"
                else:
                    state = "DISCOVERING_CONFIRMATION"
            continue

        if state == "DISCOVERING_CONFIRMATION":
            # Boundary before confirmation means collision only if nothing pending.
            if is_analyst or boundary_match:
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
        (outlook_paragraphs, source_speaker, status_note)
        source_speaker in {"CFO", "CEO", "None"}
    """
    cfo_result = _collect_outlook_for_role(
        transcript,
        role="CFO",
        context_grace_paragraphs=3,
    )
    if cfo_result["selected"]:
        return cfo_result["selected"], "CFO", "Success_CFO"

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


def _ensure_sentence_tokenizer() -> None:
    """Ensure punkt resources are available for nltk.sent_tokenize."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                # Keep going; sent_tokenize call will raise if still unavailable.
                pass


def split_sentences(outlook_paragraphs: list[tuple[int, str, str, str]]) -> list[str]:
    """Tokenize outlook text into individual non-empty sentences."""
    outlook_text = " ".join(p[3] for p in outlook_paragraphs).strip()
    if not outlook_text:
        return []

    sentences = nltk.sent_tokenize(outlook_text)
    return [s.strip() for s in sentences if s and s.strip()]


def set_deterministic(seed: int) -> None:
    """Configure deterministic behavior for reproducible FinBERT runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _probabilities_from_logits(
    logits: torch.Tensor,
    id2label: dict[int, str],
) -> tuple[float, float, float]:
    """Map model logits into (positive, neutral, negative) probabilities."""
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()

    label_probs = {id2label[i].lower(): probs[i].item() for i in range(len(probs))}

    positive = 0.0
    neutral = 0.0
    negative = 0.0

    for label, prob in label_probs.items():
        if "positive" in label:
            positive = prob
        elif "neutral" in label:
            neutral = prob
        elif "negative" in label:
            negative = prob

    return positive, neutral, negative


def score_outlook_sentences(
    sentences: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> float:
    """Score outlook as weighted average of sentence-level FinBERT sentiment.

    Label mapping:
    - Positive -> +1
    - Neutral -> 0
    - Negative -> -1
    """
    if not sentences:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    for sentence in sentences:
        encoded = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits

        p_pos, p_neu, p_neg = _probabilities_from_logits(logits, id2label)
        sentence_score = (1.0 * p_pos) + (0.0 * p_neu) + (-1.0 * p_neg)

        # Weight by sentence length so short fragments do not dominate.
        weight = max(1, len(re.findall(r"[A-Za-z]+", sentence)))
        weighted_sum += sentence_score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def standardize_scores(df: pd.DataFrame) -> pd.Series:
    """Return z-scored sentiment values for later index aggregation.

    This is included for index-construction workflows from lecture guidance.
    The standardized series is computed but not written to the output CSV to keep
    the CSV schema focused on per-call methodology results.
    """
    series = pd.to_numeric(df["outlook_sentiment_score"], errors="coerce").fillna(0.0)
    mean = float(series.mean())
    std = float(series.std(ddof=0))

    if std == 0.0:
        return pd.Series([0.0] * len(series), index=df.index)
    return (series - mean) / std


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FinBERT sentiment analysis over isolated outlook sections."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic execution (default: 42).",
    )
    parser.add_argument(
        "--write-standardized",
        action="store_true",
        help=(
            "Also write standardized z-scores to a separate CSV for index aggregation."
        ),
    )
    parser.add_argument(
        "--standardized-csv",
        type=Path,
        default=OUT_STANDARDIZED_CSV,
        help=(
            "Path for standardized score CSV (used when --write-standardized is set)."
        ),
    )
    return parser


def _print_run_summary(df: pd.DataFrame, source_counts: Counter, status_counts: Counter) -> None:
    """Print concise coverage and quality statistics for the batch run."""
    print("\nRun summary")
    print(f"Rows written: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique() if not df.empty else 0}")
    print(f"Source speaker counts: {dict(source_counts)}")
    print(f"Isolation status counts: {dict(status_counts)}")

    if not df.empty:
        valid_scores = pd.to_numeric(df["outlook_sentiment_score"], errors="coerce")
        sentence_counts = pd.to_numeric(df["sentence_count"], errors="coerce").fillna(0)
        print(
            "Score stats: "
            f"mean={valid_scores.mean():.6f}, std={valid_scores.std(ddof=0):.6f}, "
            f"min={valid_scores.min():.6f}, max={valid_scores.max():.6f}"
        )
        print(
            "Sentence stats: "
            f"mean={sentence_counts.mean():.2f}, min={int(sentence_counts.min())}, "
            f"max={int(sentence_counts.max())}"
        )


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {RAW_DIR}")

    set_deterministic(args.seed)
    _ensure_sentence_tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {MODEL_NAME} on {device} (seed={args.seed})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        print("No transcript JSON files found in data/raw/earnings_calls")
        return

    rows: list[dict] = []
    source_counts: Counter = Counter()
    status_counts: Counter = Counter()

    for path in tqdm(files, desc="FinBERT batch", unit="file"):
        parsed = parse_symbol_and_date(path)
        if not parsed:
            continue
        symbol, reported_date = parsed

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Skipping unreadable JSON {path.name}: {exc}")
            continue

        transcript = payload.get("transcript") or []
        if not isinstance(transcript, list):
            transcript = []

        outlook_paragraphs, source_speaker, status_note = isolate_outlook_paragraphs(
            transcript
        )
        sentences = split_sentences(outlook_paragraphs)
        score = score_outlook_sentences(sentences, tokenizer, model, device)

        source_counts[source_speaker] += 1
        status_counts[status_note] += 1

        rows.append(
            {
                "symbol": symbol,
                "reported_date": reported_date,
                "methodology": METHOD,
                "outlook_sentiment_score": score,
                "source_speaker": source_speaker,
                "sentence_count": len(sentences),
            }
        )

    result_df = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "reported_date",
            "methodology",
            "outlook_sentiment_score",
            "source_speaker",
            "sentence_count",
        ],
    )

    # Compute standardized values for downstream index aggregation workflows.
    standardized_scores = standardize_scores(result_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    if args.write_standardized:
        standardized_df = result_df.copy()
        standardized_df["outlook_sentiment_zscore"] = standardized_scores
        args.standardized_csv.parent.mkdir(parents=True, exist_ok=True)
        standardized_df.to_csv(args.standardized_csv, index=False)
        print(f"Saved standardized scores to {args.standardized_csv}")

    print(f"Saved {len(result_df)} rows to {OUT_CSV}")
    _print_run_summary(result_df, source_counts, status_counts)


if __name__ == "__main__":
    main()
