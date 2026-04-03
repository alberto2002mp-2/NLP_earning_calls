#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtester import BacktestConfig, EventBacktester
from llm_backend import OpenAIBackend
from market_data import get_price_series, get_shares_and_mcap
from sec_fundamentals import SecConfig, build_quarter_table
from valuation_agent import ValuationAgent, ValuationAgentConfig, ValuationInputs


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

FINBERT_SENTIMENT_CSV = PROCESSED_DIR / "finbert_sentiment_results.csv"
LEXICAL_SENTIMENT_CSV = PROCESSED_DIR / "lexical_sentiment_results.csv"

OUT_VAL_INPUTS = PROCESSED_DIR / "nvda_pit_valuation_inputs.csv"
OUT_DECISIONS = PROCESSED_DIR / "nvda_agent_decisions.csv"
OUT_BACKTEST_WITH = PROCESSED_DIR / "nvda_backtest_with_sentiment.csv"
OUT_BACKTEST_WITHOUT = PROCESSED_DIR / "nvda_backtest_without_sentiment.csv"
OUT_BACKTEST_SUMMARY = PROCESSED_DIR / "nvda_backtest_summary.csv"

OUT_PLOT_VALUATIONS = PROCESSED_DIR / "nvda_pit_valuations.png"
OUT_PLOT_DECISIONS = PROCESSED_DIR / "nvda_agent_decisions.png"
OUT_PLOT_PORTFOLIO = PROCESSED_DIR / "nvda_backtest_portfolio.png"
OUT_PLOT_SENTIMENT = PROCESSED_DIR / "nvda_sentiment_scores.png"

TICKER = "NVDA"
WINDOWS = [
    (pd.Timestamp("2024-02-05"), pd.Timestamp("2024-02-10")),
    (pd.Timestamp("2024-08-05"), pd.Timestamp("2024-08-10")),
]


def align_to_trading_day(prices: pd.Series, dt: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(dt)
    idx = prices.index
    if dt in idx:
        return dt
    later = idx[idx >= dt]
    if len(later) == 0:
        return pd.Timestamp(idx[-1])
    return pd.Timestamp(later[0])


def pick_window_trading_days(prices: pd.Series) -> pd.DatetimeIndex:
    picked = []
    for start, end in WINDOWS:
        seg = prices.loc[(prices.index >= start) & (prices.index <= end)]
        picked.extend(list(seg.index))
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(picked))))


def ensure_price_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            series = prices["Close"]
        elif prices.shape[1] == 1:
            series = prices.iloc[:, 0]
        else:
            raise ValueError(f"Unexpected price DataFrame columns: {list(prices.columns)}")
    else:
        series = prices

    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.astype(float)


def _load_sentiment_pair() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    deviation_note = None

    if FINBERT_SENTIMENT_CSV.exists() and LEXICAL_SENTIMENT_CSV.exists():
        fin = pd.read_csv(FINBERT_SENTIMENT_CSV)
        lex = pd.read_csv(LEXICAL_SENTIMENT_CSV)
        return fin, lex, deviation_note

    # Documented fallback if exercise 1 outputs are not found.
    deviation_note = (
        "Exercise 1 sentiment CSVs not found; using fallback price-momentum sentiment."
    )
    return pd.DataFrame(), pd.DataFrame(), deviation_note


def get_pti_sentiment(
    asof: pd.Timestamp,
    finbert_df: pd.DataFrame,
    lexical_df: pd.DataFrame,
    ticker: str,
) -> tuple[Optional[float], Optional[float]]:
    if finbert_df.empty or lexical_df.empty:
        return None, None

    f = finbert_df.copy()
    l = lexical_df.copy()

    f = f[f["symbol"] == ticker].copy()
    l = l[l["symbol"] == ticker].copy()

    if f.empty or l.empty:
        return None, None

    f["reported_date"] = pd.to_datetime(f["reported_date"], errors="coerce")
    l["reported_date"] = pd.to_datetime(l["reported_date"], errors="coerce")

    f = f[(f["reported_date"].notna()) & (f["reported_date"] <= asof)].sort_values("reported_date")
    l = l[(l["reported_date"].notna()) & (l["reported_date"] <= asof)].sort_values("reported_date")

    fin = None if f.empty else float(pd.to_numeric(f.iloc[-1]["outlook_sentiment_score"], errors="coerce"))
    lex = None if l.empty else float(pd.to_numeric(l.iloc[-1]["outlook_sentiment_score"], errors="coerce"))

    return fin, lex


def fallback_sentiment_from_prices(prices: pd.Series, asof: pd.Timestamp) -> tuple[float, float]:
    hist = prices.loc[:asof].tail(6)
    if len(hist) < 2:
        return 0.0, 0.0
    ret = float(hist.iloc[-1] / hist.iloc[0] - 1.0)
    # Map to bounded pseudo sentiment to avoid extreme values.
    s = float(np.tanh(ret * 20.0))
    return s, s


def build_inputs_by_date(
    prices: pd.Series,
    quarter_table: pd.DataFrame,
    trade_days: pd.DatetimeIndex,
    finbert_df: pd.DataFrame,
    lexical_df: pd.DataFrame,
    with_sentiment: bool,
    fallback_shares: Optional[float],
    fallback_mcap: Optional[float],
) -> Dict[pd.Timestamp, ValuationInputs]:
    out: Dict[pd.Timestamp, ValuationInputs] = {}

    for dt in trade_days:
        px = float(prices.loc[dt])
        hist = prices.loc[:dt].tail(5)
        hist_dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in hist.index]
        hist_vals = [float(v) for v in hist.values]

        fin, lex = get_pti_sentiment(dt, finbert_df, lexical_df, TICKER)
        if with_sentiment:
            if fin is None or lex is None:
                fin_fallback, lex_fallback = fallback_sentiment_from_prices(prices, dt)
                fin = fin if fin is not None else fin_fallback
                lex = lex if lex is not None else lex_fallback
        else:
            fin, lex = None, None

        out[pd.Timestamp(dt)] = ValuationInputs(
            asof=pd.Timestamp(dt),
            ticker=TICKER,
            price=px,
            market_cap=fallback_mcap,
            shares_outstanding=fallback_shares,
            quarter_table=quarter_table,
            finbert_sentiment=fin,
            lexical_sentiment=lex,
            recent_price_dates=hist_dates,
            recent_prices_5d=hist_vals,
        )

    return out


def decision_table(agent: ValuationAgent, inputs_by_date: Dict[pd.Timestamp, ValuationInputs], label: str) -> pd.DataFrame:
    rows = []
    for dt in sorted(inputs_by_date.keys()):
        vin = inputs_by_date[dt]
        metrics = agent.compute_metrics(vin)
        try:
            decision = agent.decide(vin)
        except RuntimeError as exc:
            # Robust fallback: if LLM provider is unavailable/quota-limited,
            # force rule-only decision path so pipeline still completes.
            emsg = str(exc).lower()
            if "rate limit" in emsg or "quota" in emsg or "429" in emsg:
                prev = agent.cfg.use_llm
                agent.cfg.use_llm = False
                decision = agent.decide(vin)
                agent.cfg.use_llm = prev
            else:
                raise
        rows.append(
            {
                "config": label,
                "date": dt,
                "ticker": vin.ticker,
                "price": vin.price,
                "ttm_revenue": metrics.get("ttm_revenue"),
                "ttm_net_income": metrics.get("ttm_net_income"),
                "ttm_fcf": metrics.get("ttm_fcf"),
                "ttm_ps": metrics.get("ps"),
                "ttm_pe": metrics.get("pe"),
                "ttm_pfcf": metrics.get("p_fcf"),
                "finbert_sentiment": vin.finbert_sentiment,
                "lexical_sentiment": vin.lexical_sentiment,
                "recent_5d_prices": "|".join(f"{d}:{p:.2f}" for d, p in zip(vin.recent_price_dates or [], vin.recent_prices_5d or [])),
                "action": decision.action,
                "confidence": decision.confidence,
                "score": decision.score,
                "thesis": decision.thesis,
            }
        )
    return pd.DataFrame(rows)


def summarize_backtest(label: str, bt_df: pd.DataFrame, bt: EventBacktester) -> dict:
    cumret = float(bt_df["portfolio_value"].iloc[-1] / bt_df["portfolio_value"].iloc[0] - 1.0)
    sharpe = float(bt.compute_sharpe(bt_df["returns"]))
    return {
        "config": label,
        "start_date": bt_df.index.min().date().isoformat(),
        "end_date": bt_df.index.max().date().isoformat(),
        "cumulative_return": cumret,
        "sharpe": sharpe,
    }


def main() -> None:
    load_dotenv(ROOT / "Code Archive-Fund Agent" / ".env")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    market_data_source = "Yahoo Finance (yfinance)"
    fundamentals_source = "SEC CompanyFacts"

    prices = get_price_series(TICKER, start="2023-01-01", end="2024-09-01")
    prices = ensure_price_series(prices)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="first")]

    trade_days = pick_window_trading_days(prices)
    if len(trade_days) == 0:
        raise RuntimeError("No trading days found inside required windows.")

    sec_cfg = SecConfig(user_agent="ValuationAgent nvda_pti@example.com")
    quarter_table = build_quarter_table(TICKER, sec_cfg)
    fallback_shares, fallback_mcap = get_shares_and_mcap(TICKER)

    fin_df, lex_df, deviation_note = _load_sentiment_pair()

    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
    llm_enabled = openai_key_present

    if llm_enabled:
        llm = OpenAIBackend(
            chat_model="gpt-4.1-mini",
            embed_model="text-embedding-3-small",
            temperature=0.2,
            max_output_tokens=600,
        )
    else:
        # Backend object still required by agent constructor; use OpenAIBackend only when key exists.
        # We run rule-only mode if key is absent.
        class _NoLLM:
            def chat(self, messages):
                return {"content": ""}

            def embed(self, texts):
                return np.zeros((len(texts), 8), dtype=np.float32)

        llm = _NoLLM()

    inputs_without = build_inputs_by_date(
        prices,
        quarter_table,
        trade_days,
        fin_df,
        lex_df,
        with_sentiment=False,
        fallback_shares=fallback_shares,
        fallback_mcap=fallback_mcap,
    )
    inputs_with = build_inputs_by_date(
        prices,
        quarter_table,
        trade_days,
        fin_df,
        lex_df,
        with_sentiment=True,
        fallback_shares=fallback_shares,
        fallback_mcap=fallback_mcap,
    )

    cfg_without = ValuationAgentConfig(use_llm=llm_enabled, use_sentiment=False)
    cfg_with = ValuationAgentConfig(use_llm=llm_enabled, use_sentiment=True, sentiment_weight=0.20)

    agent_without = ValuationAgent(llm=llm, config=cfg_without, filing_rag=None)
    agent_with = ValuationAgent(llm=llm, config=cfg_with, filing_rag=None)

    # Required PIT valuation metric table on relevant trading days.
    pit_table = decision_table(agent_with, inputs_with, label="with_sentiment")[
        [
            "date",
            "ticker",
            "price",
            "ttm_revenue",
            "ttm_net_income",
            "ttm_fcf",
            "ttm_ps",
            "ttm_pe",
            "ttm_pfcf",
            "finbert_sentiment",
            "lexical_sentiment",
            "recent_5d_prices",
        ]
    ].copy()
    pit_table.to_csv(OUT_VAL_INPUTS, index=False)

    decisions_without = decision_table(agent_without, inputs_without, "without_sentiment")
    decisions_with = decision_table(agent_with, inputs_with, "with_sentiment")
    decisions = pd.concat([decisions_without, decisions_with], ignore_index=True)
    decisions.to_csv(OUT_DECISIONS, index=False)

    bt_prices = prices.loc[(prices.index >= trade_days.min()) & (prices.index <= trade_days.max())].copy()
    bt = EventBacktester(
        prices=bt_prices,
        cfg=BacktestConfig(
            initial_cash=100_000.0,
            transaction_cost_bps=10.0,
            trade_size_units=5.0,
            allow_short=True,
        ),
    )

    bt_without = bt.run(
        ticker=TICKER,
        agent=agent_without,
        valuation_inputs_by_event_date=inputs_without,
    )
    bt_with = bt.run(
        ticker=TICKER,
        agent=agent_with,
        valuation_inputs_by_event_date=inputs_with,
    )

    bt_without.to_csv(OUT_BACKTEST_WITHOUT)
    bt_with.to_csv(OUT_BACKTEST_WITH)

    summary = pd.DataFrame(
        [
            summarize_backtest("without_sentiment", bt_without, bt),
            summarize_backtest("with_sentiment", bt_with, bt),
        ]
    )
    summary.to_csv(OUT_BACKTEST_SUMMARY, index=False)

    print("\nNVDA PIT valuation metric table (required windows):")
    print(pit_table.to_string(index=False))

    print("\nAgent outputs (without sentiment):")
    print(decisions_without[["date", "action", "confidence", "score", "thesis"]].to_string(index=False))

    print("\nAgent outputs (with sentiment):")
    print(decisions_with[["date", "action", "confidence", "score", "thesis"]].to_string(index=False))

    print("\nBacktest summary:")
    print(summary.to_string(index=False))

    print("\nDirect comparison:")
    base_sharpe = float(summary.loc[summary["config"] == "without_sentiment", "sharpe"].iloc[0])
    sent_sharpe = float(summary.loc[summary["config"] == "with_sentiment", "sharpe"].iloc[0])
    delta = sent_sharpe - base_sharpe
    direction = "improved" if delta > 0 else ("degraded" if delta < 0 else "unchanged")
    print(f"Sharpe delta (with - without): {delta:.6f} -> sentiment {direction} performance.")

    print("\nData sources:")
    print(f"- Prices: {market_data_source}")
    print(f"- Fundamentals: {fundamentals_source}")
    print(f"- Sentiment: {FINBERT_SENTIMENT_CSV.name}, {LEXICAL_SENTIMENT_CSV.name}")

    if not llm_enabled:
        print("\nBLOCKER: OPENAI_API_KEY not set, so ChatGPT calls were disabled and rule-only mode was used.")
    if deviation_note:
        print(f"\nDEVIATION: {deviation_note}")

    print("\nSaved outputs:")
    print(f"- {OUT_VAL_INPUTS}")
    print(f"- {OUT_DECISIONS}")
    print(f"- {OUT_BACKTEST_WITHOUT}")
    print(f"- {OUT_BACKTEST_WITH}")
    print(f"- {OUT_BACKTEST_SUMMARY}")

    generate_plots(pit_table, decisions_without, decisions_with, bt_without, bt_with)

    print(f"\nSaved plots:")
    print(f"- {OUT_PLOT_VALUATIONS}")
    print(f"- {OUT_PLOT_DECISIONS}")
    print(f"- {OUT_PLOT_PORTFOLIO}")
    print(f"- {OUT_PLOT_SENTIMENT}")


def generate_plots(
    pit: pd.DataFrame,
    dec_without: pd.DataFrame,
    dec_with: pd.DataFrame,
    bt_without: pd.DataFrame,
    bt_with: pd.DataFrame,
) -> None:
    """Produce four PNG plots from agent outputs."""
    MARKER_STYLE = {"buy": ("^", "green"), "sell": ("v", "red"), "hold": ("o", "orange")}
    DATE_FMT = mdates.DateFormatter("%b %d")

    # ── 1. PIT Valuation Multiples ──────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    pit = pit.copy()
    pit["date"] = pd.to_datetime(pit["date"])

    axes[0].plot(pit["date"], pit["price"], color="navy", linewidth=1.8, label="Price (USD)")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(pit["date"], pit["ttm_pe"], color="steelblue", width=0.8, label="TTM P/E")
    axes[1].set_ylabel("P/E")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(pit["date"], pit["ttm_pfcf"], color="tomato", width=0.8, label="TTM P/FCF")
    axes[2].set_ylabel("P/FCF")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    axes[3].bar(pit["date"], pit["ttm_ps"], color="mediumseagreen", width=0.8, label="TTM P/S")
    axes[3].set_ylabel("P/S")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)
    axes[3].xaxis.set_major_formatter(DATE_FMT)
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("NVDA – Point-in-Time Valuation Multiples", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PLOT_VALUATIONS, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Agent Decisions overlaid on Price ────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for ax, df, title in [
        (ax_top, dec_without, "Without Sentiment"),
        (ax_bot, dec_with, "With Sentiment"),
    ]:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        ax.plot(df["date"], df["price"], color="navy", linewidth=1.5, zorder=1, label="Price")
        for action, (marker, color) in MARKER_STYLE.items():
            sub = df[df["action"] == action]
            if not sub.empty:
                ax.scatter(sub["date"], sub["price"], marker=marker, color=color, s=120, zorder=3, label=action.capitalize())
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Price (USD)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_bot.xaxis.set_major_formatter(DATE_FMT)
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.suptitle("NVDA – Agent Action Signals on Price", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PLOT_DECISIONS, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Backtest Portfolio Value ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for df, label, color in [
        (bt_without, "Without Sentiment", "steelblue"),
        (bt_with, "With Sentiment", "darkorange"),
    ]:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        ax.plot(df.index, df["portfolio_value"], label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_title("NVDA – Backtest Portfolio Value: With vs Without Sentiment", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DATE_FMT)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_PLOT_PORTFOLIO, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Sentiment Scores ──────────────────────────────────────────────────
    pit_sent = pit[["date", "finbert_sentiment", "lexical_sentiment"]].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))

    if not pit_sent.empty:
        ax.plot(pit_sent["date"], pit_sent["finbert_sentiment"], label="FinBERT", color="mediumpurple", linewidth=1.8, marker="o", markersize=5)
        ax.plot(pit_sent["date"], pit_sent["lexical_sentiment"], label="Lexical (LM)", color="teal", linewidth=1.8, linestyle="--", marker="s", markersize=5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score")
        ax.set_title("NVDA – PIT Sentiment Scores (FinBERT vs Lexical)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DATE_FMT)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_PLOT_SENTIMENT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
