"""Fetch 2024 sector prices and plot PIT indicators against market benchmarks.

Creates two charts:
- IT FinBERT + IT Lexical + ^SP500-45
- Industrials FinBERT + Industrials Lexical + ^SP500-20
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDICATOR_CSV = PROCESSED_DIR / "pit_indicator_evolution_data.csv"

PRICE_CSV = PROCESSED_DIR / "sector_prices_2024.csv"
IT_MERGED_CSV = PROCESSED_DIR / "it_indicator_with_price_2024.csv"
IND_MERGED_CSV = PROCESSED_DIR / "industrials_indicator_with_price_2024.csv"

IT_PLOT = PROCESSED_DIR / "it_indicators_vs_sp500_45.png"
IND_PLOT = PROCESSED_DIR / "industrials_indicators_vs_sp500_20.png"
IT_PLOT_RAW = PROCESSED_DIR / "it_indicators_vs_sp500_45_raw.png"
IND_PLOT_RAW = PROCESSED_DIR / "industrials_indicators_vs_sp500_20_raw.png"
IT_PLOT_RAW_DAILY_PCT = PROCESSED_DIR / "it_indicators_vs_sp500_45_raw_daily_pct.png"
IND_PLOT_RAW_DAILY_PCT = PROCESSED_DIR / "industrials_indicators_vs_sp500_20_raw_daily_pct.png"
IT_PLOT_RAW_DAILY_LOG = PROCESSED_DIR / "it_indicators_vs_sp500_45_raw_daily_log.png"
IND_PLOT_RAW_DAILY_LOG = PROCESSED_DIR / "industrials_indicators_vs_sp500_20_raw_daily_log.png"
IT_PLOT_RAW_WEEKLY_PCT = PROCESSED_DIR / "it_indicators_vs_sp500_45_raw_weekly_pct.png"
IND_PLOT_RAW_WEEKLY_PCT = PROCESSED_DIR / "industrials_indicators_vs_sp500_20_raw_weekly_pct.png"
IT_PLOT_RAW_MONTHLY_PCT = PROCESSED_DIR / "it_indicators_vs_sp500_45_raw_monthly_pct.png"
IND_PLOT_RAW_MONTHLY_PCT = PROCESSED_DIR / "industrials_indicators_vs_sp500_20_raw_monthly_pct.png"


def fetch_prices_2024() -> pd.DataFrame:
    symbols = ["^SP500-45", "^SP500-20"]
    data = yf.download(
        symbols,
        start="2024-01-01",
        end="2025-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if data is None or data.empty:
        raise ValueError("No price data returned from yfinance for requested symbols.")

    # Handle both single-index and multi-index return formats.
    price_df = pd.DataFrame(index=data.index)
    for sym in symbols:
        if (sym, "Close") in data.columns:
            price_df[sym] = data[(sym, "Close")]
        elif "Close" in data.columns and len(symbols) == 1:
            price_df[sym] = data["Close"]
        else:
            raise ValueError(f"Could not find Close prices for {sym} from yfinance.")

    price_df.index = pd.to_datetime(price_df.index)
    price_df.index.name = "date"
    return price_df


def normalize_to_100(series: pd.Series) -> pd.Series:
    first_valid = series.dropna().iloc[0]
    return (series / first_valid) * 100.0


def resampled_percent_change(series: pd.Series, rule: str, full_dates: pd.DatetimeIndex) -> pd.Series:
    """Compute percent change on a resampled close series and align back to full dates."""
    resampled = series.resample(rule).last()
    changed = resampled.pct_change() * 100.0
    return changed.reindex(full_dates).ffill()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    indicator = pd.read_csv(INDICATOR_CSV)
    indicator["date"] = pd.to_datetime(indicator["date"])
    indicator = indicator.sort_values("date")

    prices = fetch_prices_2024()
    prices.to_csv(PRICE_CSV)

    full_dates = pd.DatetimeIndex(indicator["date"])
    prices_daily = prices.reindex(full_dates).ffill()

    merged = indicator.merge(
        prices_daily.reset_index().rename(columns={"index": "date"}),
        on="date",
        how="left",
    )

    merged["sp500_45_norm100"] = normalize_to_100(merged["^SP500-45"])
    merged["sp500_20_norm100"] = normalize_to_100(merged["^SP500-20"])
    merged["sp500_45_weekly_pct_change"] = resampled_percent_change(
        merged.set_index("date")["^SP500-45"], "W-FRI", full_dates
    ).values
    merged["sp500_20_weekly_pct_change"] = resampled_percent_change(
        merged.set_index("date")["^SP500-20"], "W-FRI", full_dates
    ).values
    merged["sp500_45_monthly_pct_change"] = resampled_percent_change(
        merged.set_index("date")["^SP500-45"], "ME", full_dates
    ).values
    merged["sp500_20_monthly_pct_change"] = resampled_percent_change(
        merged.set_index("date")["^SP500-20"], "ME", full_dates
    ).values

    it_df = merged[
        [
            "date",
            "it_finbert_pit_zscore",
            "it_lexical_pit_zscore",
            "^SP500-45",
            "sp500_45_norm100",
            "sp500_45_weekly_pct_change",
            "sp500_45_monthly_pct_change",
        ]
    ].copy()
    it_df["sp500_45_daily_pct_change"] = it_df["^SP500-45"].pct_change() * 100.0
    it_df["sp500_45_daily_log_change"] = (
        it_df["^SP500-45"].replace(0, np.nan).transform(np.log).diff() * 100.0
    )

    ind_df = merged[
        [
            "date",
            "industrials_finbert_pit_zscore",
            "industrials_lexical_pit_zscore",
            "^SP500-20",
            "sp500_20_norm100",
            "sp500_20_weekly_pct_change",
            "sp500_20_monthly_pct_change",
        ]
    ].copy()
    ind_df["sp500_20_daily_pct_change"] = ind_df["^SP500-20"].pct_change() * 100.0
    ind_df["sp500_20_daily_log_change"] = (
        ind_df["^SP500-20"].replace(0, np.nan).transform(np.log).diff() * 100.0
    )

    it_df.to_csv(IT_MERGED_CSV, index=False)
    ind_df.to_csv(IND_MERGED_CSV, index=False)

    # IT plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(it_df["date"], it_df["it_finbert_pit_zscore"], label="IT FinBERT PIT Z", linewidth=1.8)
    ax1.plot(it_df["date"], it_df["it_lexical_pit_zscore"], label="IT Lexical PIT Z", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(it_df["date"], it_df["sp500_45_norm100"], label="^SP500-45 (Norm=100)", color="black", linewidth=1.8)

    ax1.set_title("IT Indicators vs S&P 500 Information Technology (2024)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Index Level (Start=100)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IT_PLOT, dpi=180)
    plt.close(fig)

    # IT plot with raw (non-normalized) index levels.
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(it_df["date"], it_df["it_finbert_pit_zscore"], label="IT FinBERT PIT Z", linewidth=1.8)
    ax1.plot(it_df["date"], it_df["it_lexical_pit_zscore"], label="IT Lexical PIT Z", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(it_df["date"], it_df["^SP500-45"], label="^SP500-45 (Raw)", color="black", linewidth=1.8)

    ax1.set_title("IT Indicators vs S&P 500 Information Technology (2024, Raw Index)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Index Level (Raw)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IT_PLOT_RAW, dpi=180)
    plt.close(fig)

    # IT plot with raw daily percent changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(it_df["date"], it_df["it_finbert_pit_zscore"], label="IT FinBERT PIT Z", linewidth=1.8)
    ax1.plot(it_df["date"], it_df["it_lexical_pit_zscore"], label="IT Lexical PIT Z", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        it_df["date"],
        it_df["sp500_45_daily_pct_change"],
        label="^SP500-45 Daily % Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("IT Indicators vs S&P 500 Information Technology (Raw Daily % Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Daily % Change")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IT_PLOT_RAW_DAILY_PCT, dpi=180)
    plt.close(fig)

    # IT plot with raw daily log changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(it_df["date"], it_df["it_finbert_pit_zscore"], label="IT FinBERT PIT Z", linewidth=1.8)
    ax1.plot(it_df["date"], it_df["it_lexical_pit_zscore"], label="IT Lexical PIT Z", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        it_df["date"],
        it_df["sp500_45_daily_log_change"],
        label="^SP500-45 Daily Log Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("IT Indicators vs S&P 500 Information Technology (Raw Daily Log Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Daily Log Change (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IT_PLOT_RAW_DAILY_LOG, dpi=180)
    plt.close(fig)

    # IT plot with raw weekly percent changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(it_df["date"], it_df["it_finbert_pit_zscore"], label="IT FinBERT PIT Z", linewidth=1.8)
    ax1.plot(it_df["date"], it_df["it_lexical_pit_zscore"], label="IT Lexical PIT Z", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        it_df["date"],
        it_df["sp500_45_weekly_pct_change"],
        label="^SP500-45 Weekly % Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("IT Indicators vs S&P 500 Information Technology (Raw Weekly % Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Weekly % Change")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IT_PLOT_RAW_WEEKLY_PCT, dpi=180)
    plt.close(fig)

    # IT plot with raw monthly percent changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(it_df["date"], it_df["it_finbert_pit_zscore"], label="IT FinBERT PIT Z", linewidth=1.8)
    ax1.plot(it_df["date"], it_df["it_lexical_pit_zscore"], label="IT Lexical PIT Z", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        it_df["date"],
        it_df["sp500_45_monthly_pct_change"],
        label="^SP500-45 Monthly % Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("IT Indicators vs S&P 500 Information Technology (Raw Monthly % Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Monthly % Change")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IT_PLOT_RAW_MONTHLY_PCT, dpi=180)
    plt.close(fig)

    # Industrials plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        ind_df["date"],
        ind_df["industrials_finbert_pit_zscore"],
        label="Industrials FinBERT PIT Z",
        linewidth=1.8,
    )
    ax1.plot(
        ind_df["date"],
        ind_df["industrials_lexical_pit_zscore"],
        label="Industrials Lexical PIT Z",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        ind_df["date"],
        ind_df["sp500_20_norm100"],
        label="^SP500-20 (Norm=100)",
        color="black",
        linewidth=1.8,
    )

    ax1.set_title("Industrials Indicators vs S&P 500 Industrials (2024)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Index Level (Start=100)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IND_PLOT, dpi=180)
    plt.close(fig)

    # Industrials plot with raw (non-normalized) index levels.
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        ind_df["date"],
        ind_df["industrials_finbert_pit_zscore"],
        label="Industrials FinBERT PIT Z",
        linewidth=1.8,
    )
    ax1.plot(
        ind_df["date"],
        ind_df["industrials_lexical_pit_zscore"],
        label="Industrials Lexical PIT Z",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        ind_df["date"],
        ind_df["^SP500-20"],
        label="^SP500-20 (Raw)",
        color="black",
        linewidth=1.8,
    )

    ax1.set_title("Industrials Indicators vs S&P 500 Industrials (2024, Raw Index)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Index Level (Raw)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IND_PLOT_RAW, dpi=180)
    plt.close(fig)

    # Industrials plot with raw daily percent changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        ind_df["date"],
        ind_df["industrials_finbert_pit_zscore"],
        label="Industrials FinBERT PIT Z",
        linewidth=1.8,
    )
    ax1.plot(
        ind_df["date"],
        ind_df["industrials_lexical_pit_zscore"],
        label="Industrials Lexical PIT Z",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        ind_df["date"],
        ind_df["sp500_20_daily_pct_change"],
        label="^SP500-20 Daily % Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("Industrials Indicators vs S&P 500 Industrials (Raw Daily % Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Daily % Change")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IND_PLOT_RAW_DAILY_PCT, dpi=180)
    plt.close(fig)

    # Industrials plot with raw daily log changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        ind_df["date"],
        ind_df["industrials_finbert_pit_zscore"],
        label="Industrials FinBERT PIT Z",
        linewidth=1.8,
    )
    ax1.plot(
        ind_df["date"],
        ind_df["industrials_lexical_pit_zscore"],
        label="Industrials Lexical PIT Z",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        ind_df["date"],
        ind_df["sp500_20_daily_log_change"],
        label="^SP500-20 Daily Log Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("Industrials Indicators vs S&P 500 Industrials (Raw Daily Log Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Daily Log Change (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IND_PLOT_RAW_DAILY_LOG, dpi=180)
    plt.close(fig)

    # Industrials plot with raw weekly percent changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        ind_df["date"],
        ind_df["industrials_finbert_pit_zscore"],
        label="Industrials FinBERT PIT Z",
        linewidth=1.8,
    )
    ax1.plot(
        ind_df["date"],
        ind_df["industrials_lexical_pit_zscore"],
        label="Industrials Lexical PIT Z",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        ind_df["date"],
        ind_df["sp500_20_weekly_pct_change"],
        label="^SP500-20 Weekly % Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("Industrials Indicators vs S&P 500 Industrials (Raw Weekly % Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Weekly % Change")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IND_PLOT_RAW_WEEKLY_PCT, dpi=180)
    plt.close(fig)

    # Industrials plot with raw monthly percent changes (price side only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        ind_df["date"],
        ind_df["industrials_finbert_pit_zscore"],
        label="Industrials FinBERT PIT Z",
        linewidth=1.8,
    )
    ax1.plot(
        ind_df["date"],
        ind_df["industrials_lexical_pit_zscore"],
        label="Industrials Lexical PIT Z",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax2.plot(
        ind_df["date"],
        ind_df["sp500_20_monthly_pct_change"],
        label="^SP500-20 Monthly % Change",
        color="black",
        linewidth=1.6,
    )
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)

    ax1.set_title("Industrials Indicators vs S&P 500 Industrials (Raw Monthly % Change)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PIT Z-Score")
    ax2.set_ylabel("Monthly % Change")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(IND_PLOT_RAW_MONTHLY_PCT, dpi=180)
    plt.close(fig)

    print("Saved files:")
    print(f"  {PRICE_CSV}")
    print(f"  {IT_MERGED_CSV}")
    print(f"  {IND_MERGED_CSV}")
    print(f"  {IT_PLOT}")
    print(f"  {IND_PLOT}")
    print(f"  {IT_PLOT_RAW}")
    print(f"  {IND_PLOT_RAW}")
    print(f"  {IT_PLOT_RAW_DAILY_PCT}")
    print(f"  {IND_PLOT_RAW_DAILY_PCT}")
    print(f"  {IT_PLOT_RAW_DAILY_LOG}")
    print(f"  {IND_PLOT_RAW_DAILY_LOG}")
    print(f"  {IT_PLOT_RAW_WEEKLY_PCT}")
    print(f"  {IND_PLOT_RAW_WEEKLY_PCT}")
    print(f"  {IT_PLOT_RAW_MONTHLY_PCT}")
    print(f"  {IND_PLOT_RAW_MONTHLY_PCT}")


if __name__ == "__main__":
    main()
