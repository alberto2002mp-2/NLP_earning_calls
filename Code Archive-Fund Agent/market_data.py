#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:48:50 2026

@author: fabriziocoiai
"""
# market_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf


@dataclass
class MarketSnapshot:
    date: pd.Timestamp
    price: float
    shares_outstanding: Optional[float]
    market_cap: Optional[float]


def get_price_series(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}.")
    # use close
    return df["Close"].dropna().astype(float)


def get_shares_and_mcap(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    t = yf.Ticker(ticker)
    info = getattr(t, "info", {}) or {}
    shares = info.get("sharesOutstanding", None)
    mcap = info.get("marketCap", None)
    try:
        shares = float(shares) if shares is not None else None
    except Exception:
        shares = None
    try:
        mcap = float(mcap) if mcap is not None else None
    except Exception:
        mcap = None
    return shares, mcap