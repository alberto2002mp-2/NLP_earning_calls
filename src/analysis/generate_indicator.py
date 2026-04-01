"""Generate Point-in-Time (PIT) sector sentiment indicators for 2024.

Inputs:
- data/processed/finbert_sentiment_results.csv
- data/processed/lexical_sentiment_results.csv
- data/tech_companies_table.csv
- data/industrial_companies_table.csv

Outputs:
- data/processed/it_finbert_indicator.csv
- data/processed/industrials_finbert_indicator.csv
- data/processed/it_lexical_indicator.csv
- data/processed/industrials_lexical_indicator.csv
- data/processed/pit_indicator_evolution.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

FINBERT_PATH = PROCESSED_DIR / "finbert_sentiment_results.csv"
LEXICAL_PATH = PROCESSED_DIR / "lexical_sentiment_results.csv"
TECH_WEIGHTS_PATH = DATA_DIR / "tech_companies_table.csv"
INDUSTRIAL_WEIGHTS_PATH = DATA_DIR / "industrial_companies_table.csv"

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2024-12-31")
DAILY_INDEX = pd.date_range(START_DATE, END_DATE, freq="D")
AUDIT_DATES = [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-05-01"), pd.Timestamp("2024-08-01")]

COMPANY_TO_TICKER = {
    "Accenture": "ACN",
    "Adobe": "ADBE",
    "Advanced Micro Devices": "AMD",
    "Apple": "AAPL",
    "Applied Materials": "AMAT",
    "Broadcom": "AVGO",
    "Cisco Systems": "CSCO",
    "Intel": "INTC",
    "IBM": "IBM",
    "Intuit": "INTU",
    "Micron Technology": "MU",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Oracle": "ORCL",
    "Palantir Technologies": "PLTR",
    "QUALCOMM": "QCOM",
    "Salesforce": "CRM",
    "ServiceNow": "NOW",
    "Texas Instruments": "TXN",
    "3M Co": "MMM",
    "AMETEK Inc": "AME",
    "Automatic Data Processing Inc": "ADP",
    "Axon Enterprise Inc": "AXON",
    "Boeing Co/The": "BA",
    "Carrier Global Corp": "CARR",
    "Caterpillar Inc": "CAT",
    "Cintas Corp": "CTAS",
    "Copart Inc": "CPRT",
    "CSX Corp": "CSX",
    "Cummins Inc": "CMI",
    "Deere & Co": "DE",
    "Eaton Corp PLC": "ETN",
    "Emerson Electric Co": "EMR",
    "Fastenal Co": "FAST",
    "FedEx Corp": "FDX",
    "GE Vernova Inc": "GEV",
    "General Dynamics Corp": "GD",
    "General Electric Co": "GE",
    "Honeywell International Inc": "HON",
    "Howmet Aerospace Inc": "HWM",
    "Illinois Tool Works Inc": "ITW",
    "Johnson Controls International": "JCI",
    "L3Harris Technologies Inc": "LHX",
    "Lockheed Martin Corp": "LMT",
    "Norfolk Southern Corp": "NSC",
    "Northrop Grumman Corp": "NOC",
    "PACCAR Inc": "PCAR",
    "Parker-Hannifin Corp": "PH",
    "Paychex Inc": "PAYX",
    "Quanta Services Inc": "PWR",
    "Republic Services Inc": "RSG",
    "RTX Corp": "RTX",
    "Trane Technologies PLC": "TT",
    "TransDigm Group Inc": "TDG",
    "Uber Technologies Inc": "UBER",
    "Union Pacific Corp": "UNP",
    "United Parcel Service Inc": "UPS",
    "United Rentals Inc": "URI",
    "Waste Management Inc": "WM",
    "WW Grainger Inc": "GWW",
}


def _clean_percent(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().replace("%", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _load_wide_weights(path: Path) -> tuple[list[str], list[pd.Timestamp], dict[str, dict[pd.Timestamp, float]]]:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["Company"] = df["Company"].astype(str).str.strip()
    df = df[df["Company"] != ""]

    date_columns = [c for c in df.columns if c != "Company"]
    rebalance_dates = [pd.to_datetime(c, dayfirst=True) for c in date_columns]

    symbol_to_weights: dict[str, dict[pd.Timestamp, float]] = {}
    for _, row in df.iterrows():
        company = row["Company"]
        symbol = COMPANY_TO_TICKER.get(company)
        if not symbol:
            continue

        entries: dict[pd.Timestamp, float] = {}
        for col, dt in zip(date_columns, rebalance_dates):
            entries[dt] = _clean_percent(row[col])
        symbol_to_weights[symbol] = entries

    return sorted(symbol_to_weights.keys()), sorted(rebalance_dates), symbol_to_weights


def build_daily_step_weights(path: Path) -> pd.DataFrame:
    symbols, rebalance_dates, symbol_to_weights = _load_wide_weights(path)
    out = pd.DataFrame(0.0, index=DAILY_INDEX, columns=symbols)

    for symbol in symbols:
        for i, dt in enumerate(rebalance_dates):
            next_dt = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else END_DATE + pd.Timedelta(days=1)
            period_start = max(START_DATE, dt)
            period_end = min(END_DATE, next_dt - pd.Timedelta(days=1))
            if period_start > period_end:
                continue
            out.loc[period_start:period_end, symbol] = symbol_to_weights[symbol].get(dt, 0.0)

    # Re-weight to 100% inside sector each day.
    row_sum = out.sum(axis=1)
    normalized = out.div(row_sum.replace(0.0, np.nan), axis=0) * 100.0
    normalized = normalized.fillna(0.0)
    return normalized


def build_persistent_panel(
    sentiment_df: pd.DataFrame,
    tickers: list[str],
    score_col: str,
    relevance_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_panel = pd.DataFrame(np.nan, index=DAILY_INDEX, columns=tickers)
    relevance_panel = pd.DataFrame(np.nan, index=DAILY_INDEX, columns=tickers)

    for ticker in tickers:
        company_rows = sentiment_df[sentiment_df["symbol"] == ticker].copy()
        if company_rows.empty:
            continue

        company_rows = company_rows.sort_values("reported_date")

        for _, row in company_rows.iterrows():
            dt = row["reported_date"]
            if dt < START_DATE or dt > END_DATE:
                continue
            score_panel.at[dt, ticker] = float(row[score_col])
            relevance_panel.at[dt, ticker] = float(row[relevance_col])

        score_panel[ticker] = score_panel[ticker].ffill()
        relevance_panel[ticker] = relevance_panel[ticker].ffill()

    return score_panel, relevance_panel


def aggregate_daily_sentiment(
    weights: pd.DataFrame,
    score_panel: pd.DataFrame,
    relevance_panel: pd.DataFrame,
) -> pd.Series:
    aligned_scores = score_panel.reindex(index=weights.index, columns=weights.columns)
    aligned_relevance = relevance_panel.reindex(index=weights.index, columns=weights.columns)

    weighted_relevance = weights * aligned_relevance
    numerator = (aligned_scores * weighted_relevance).sum(axis=1, min_count=1)
    denominator = weighted_relevance.sum(axis=1, min_count=1)

    series = numerator / denominator.replace(0.0, np.nan)
    return series


def pit_expanding_zscore(series: pd.Series, warmup_days: int = 10) -> pd.Series:
    z = pd.Series(0.0, index=series.index)
    valid_history: list[float] = []

    for dt, value in series.items():
        if pd.notna(value):
            valid_history.append(float(value))

        if len(valid_history) < warmup_days or pd.isna(value):
            z.loc[dt] = 0.0
            continue

        arr = np.array(valid_history, dtype=float)
        std = float(arr.std(ddof=0))
        if std == 0.0:
            z.loc[dt] = 0.0
            continue

        mean = float(arr.mean())
        z.loc[dt] = (float(value) - mean) / std

    return z


def compute_indicator(
    sector_name: str,
    method_name: str,
    sentiment_df: pd.DataFrame,
    weights_daily: pd.DataFrame,
    relevance_mode: str,
) -> pd.DataFrame:
    sector_tickers = list(weights_daily.columns)

    filtered = sentiment_df[sentiment_df["symbol"].isin(sector_tickers)].copy()
    filtered["reported_date"] = pd.to_datetime(filtered["reported_date"])
    filtered["outlook_sentiment_score"] = pd.to_numeric(
        filtered["outlook_sentiment_score"], errors="coerce"
    ).fillna(0.0)

    if relevance_mode == "sentence_count":
        filtered["relevance"] = pd.to_numeric(
            filtered["sentence_count"], errors="coerce"
        ).fillna(0.0)
    else:
        filtered["relevance"] = 1.0

    scores, relevance = build_persistent_panel(
        filtered,
        sector_tickers,
        score_col="outlook_sentiment_score",
        relevance_col="relevance",
    )

    raw_sentiment = aggregate_daily_sentiment(weights_daily, scores, relevance)
    zscore = pit_expanding_zscore(raw_sentiment, warmup_days=10)

    out = pd.DataFrame(
        {
            "date": DAILY_INDEX,
            "sector": sector_name,
            "methodology": method_name,
            "raw_weighted_sentiment": raw_sentiment.values,
            "pit_zscore": zscore.values,
        }
    )
    return out


def print_audit(indicator_df: pd.DataFrame, label: str) -> None:
    audit = indicator_df.set_index("date").reindex(AUDIT_DATES)
    print(f"\nAudit weighted sentiment ({label}):")
    for dt in AUDIT_DATES:
        val = audit.at[dt, "raw_weighted_sentiment"]
        shown = "NaN" if pd.isna(val) else f"{float(val):.6f}"
        print(f"  {dt.date()}: {shown}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tech_weights_daily = build_daily_step_weights(TECH_WEIGHTS_PATH)
    industrial_weights_daily = build_daily_step_weights(INDUSTRIAL_WEIGHTS_PATH)

    finbert = pd.read_csv(FINBERT_PATH)
    lexical = pd.read_csv(LEXICAL_PATH)

    it_finbert = compute_indicator(
        sector_name="Information_Technology",
        method_name="FinBERT",
        sentiment_df=finbert,
        weights_daily=tech_weights_daily,
        relevance_mode="sentence_count",
    )
    industrial_finbert = compute_indicator(
        sector_name="Industrials_Transport",
        method_name="FinBERT",
        sentiment_df=finbert,
        weights_daily=industrial_weights_daily,
        relevance_mode="sentence_count",
    )

    it_lexical = compute_indicator(
        sector_name="Information_Technology",
        method_name="Lexical_LM",
        sentiment_df=lexical,
        weights_daily=tech_weights_daily,
        relevance_mode="unit",
    )
    industrial_lexical = compute_indicator(
        sector_name="Industrials_Transport",
        method_name="Lexical_LM",
        sentiment_df=lexical,
        weights_daily=industrial_weights_daily,
        relevance_mode="unit",
    )

    it_finbert.to_csv(PROCESSED_DIR / "it_finbert_indicator.csv", index=False)
    industrial_finbert.to_csv(PROCESSED_DIR / "industrials_finbert_indicator.csv", index=False)
    it_lexical.to_csv(PROCESSED_DIR / "it_lexical_indicator.csv", index=False)
    industrial_lexical.to_csv(PROCESSED_DIR / "industrials_lexical_indicator.csv", index=False)

    print_audit(it_finbert, "IT FinBERT")
    print_audit(industrial_finbert, "Industrials FinBERT")
    print_audit(it_lexical, "IT Lexical")
    print_audit(industrial_lexical, "Industrials Lexical")

    plt.figure(figsize=(12, 6))
    plt.plot(it_finbert["date"], it_finbert["pit_zscore"], label="IT FinBERT", linewidth=1.8)
    plt.plot(
        industrial_finbert["date"],
        industrial_finbert["pit_zscore"],
        label="Industrials FinBERT",
        linewidth=1.8,
    )
    plt.plot(it_lexical["date"], it_lexical["pit_zscore"], label="IT Lexical", linewidth=1.2, alpha=0.9)
    plt.plot(
        industrial_lexical["date"],
        industrial_lexical["pit_zscore"],
        label="Industrials Lexical",
        linewidth=1.2,
        alpha=0.9,
    )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Evolution of Bullishness: PIT Z-Scores (2024)")
    plt.xlabel("Date")
    plt.ylabel("PIT Z-Score")
    plt.legend()
    plt.tight_layout()
    plot_path = PROCESSED_DIR / "pit_indicator_evolution.png"
    plt.savefig(plot_path, dpi=180)

    print("\nSaved outputs:")
    print(f"  {PROCESSED_DIR / 'it_finbert_indicator.csv'}")
    print(f"  {PROCESSED_DIR / 'industrials_finbert_indicator.csv'}")
    print(f"  {PROCESSED_DIR / 'it_lexical_indicator.csv'}")
    print(f"  {PROCESSED_DIR / 'industrials_lexical_indicator.csv'}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
