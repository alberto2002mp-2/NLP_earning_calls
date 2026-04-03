#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:51:39 2026

@author: fabriziocoiai
"""
# main.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from llm_backend import GeminiBackend
from sec_fundamentals import SecConfig, build_quarter_table, build_filing_date_events
from market_data import get_price_series
from valuation_agent import ValuationAgent, ValuationInputs, ValuationAgentConfig
from backtester import EventBacktester, BacktestConfig
from filing_rag import FilingRAG


def align_to_trading_day(prices: pd.Series, dt: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(dt)
    idx = prices.index
    if dt in idx:
        return dt
    later = idx[idx >= dt]
    if len(later) == 0:
        return pd.Timestamp(idx[-1])
    return pd.Timestamp(later[0])


def main():
    ticker = "AAPL"
    start = "2022-01-01"
    end = "2026-01-01"

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY first.")

    sec_cfg = SecConfig(user_agent="ValuationAgent xyz@gmail.com")

    prices = get_price_series(ticker, start=start, end=end)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="first")]

    llm = GeminiBackend(
        chat_model="gemini-2.0-flash",
        embed_model="gemini-embedding-001",
        temperature=0.2,
        max_output_tokens=800,
    )

    quarter_table = build_quarter_table(ticker, sec_cfg)
    filing_dates = build_filing_date_events(quarter_table)

    filing_rag = FilingRAG()

    print("Adding one small filing snippet to RAG...")
    filing_rag.add_document(
        llm,
        doc_id="aapl_test_doc",
        ticker=ticker,
        filed=pd.Timestamp("2025-05-02"),
        source="10-Q",
        text="Management discussed services growth, gross margin resilience, and ongoing capital returns."
    )
    print("RAG document added.")
    
    #################
    #REAL SEC filing
    
#     filing_docs = fetch_and_prepare_filing_texts(
#     ticker=ticker,
#     cfg=text_cfg,
#     forms=["10-Q"],
#     date_from="2025-01-01",
#     date_to="2025-12-31",
# )

# print("Fetched filing docs:", len(filing_docs))

# for doc in filing_docs[:1]:
#     print("Adding filing:", doc["doc_id"], doc["filed"])
#     filing_rag.add_document(
#         llm,
#         doc_id=doc["doc_id"],
#         ticker=doc["ticker"],
#         filed=doc["filed"],
#         source=doc["source"],
#         text=doc["text"][:10000],
#     )
    
    #####
     

    agent = ValuationAgent(
        llm=llm,
        config=ValuationAgentConfig(
            use_llm=False,  # use True for LMM to work
            cheap_threshold_ps=3.0,
            expensive_threshold_ps=8.0,
            cheap_threshold_pe=15.0,
            expensive_threshold_pe=30.0,
            cheap_threshold_pfcf=12.0,
            expensive_threshold_pfcf=25.0,
        ),
        filing_rag=filing_rag,
    )

    valuation_inputs_by_event_date = {}
    for fd in filing_dates:
        fd = pd.Timestamp(fd)
        if fd < prices.index.min() or fd > prices.index.max():
            continue

        trade_dt = align_to_trading_day(prices, fd)

        val = prices.loc[trade_dt]
        price = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)

        vin = ValuationInputs(
            asof=trade_dt,
            ticker=ticker,
            price=price,
            market_cap=None,
            shares_outstanding=None,
            quarter_table=quarter_table,
        )
        valuation_inputs_by_event_date[pd.Timestamp(trade_dt)] = vin

    bt = EventBacktester(
        prices=prices,
        cfg=BacktestConfig(
            initial_cash=100_000.0,
            transaction_cost_bps=10.0,
            trade_size_units=5.0,
            allow_short=True,
        ),
    )

    results = bt.run(
        ticker=ticker,
        agent=agent,
        valuation_inputs_by_event_date=valuation_inputs_by_event_date,
    )

    print(results.tail(15)[["price", "action", "position", "portfolio_value", "decision_score", "decision_confidence"]])
    print("\nCumulative return:", results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0] - 1)

    last_decision = results.dropna(subset=["decision_thesis"]).tail(1)
    if not last_decision.empty:
        print("\nLast thesis:\n", last_decision["decision_thesis"].iloc[0])


if __name__ == "__main__":
    main()