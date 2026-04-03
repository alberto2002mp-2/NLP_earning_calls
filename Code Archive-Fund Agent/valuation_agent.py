#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:49:55 2026

@author: fabriziocoiai
"""
# valuation_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from llm_backend import LLMBackend
from sec_fundamentals import ttm_from_quarters
from filing_rag import FilingRAG


@dataclass
class ValuationInputs:
    asof: pd.Timestamp
    ticker: str
    price: float
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    quarter_table: pd.DataFrame
    finbert_sentiment: Optional[float] = None
    lexical_sentiment: Optional[float] = None
    recent_price_dates: Optional[List[str]] = None
    recent_prices_5d: Optional[List[float]] = None


@dataclass
class ValuationDecision:
    ticker: str
    asof: pd.Timestamp
    action: str
    confidence: float
    score: float
    thesis: str
    key_points: List[str]
    risks: List[str]
    metrics: Dict[str, Optional[float]]


@dataclass
class ValuationAgentConfig:
    output_actions: Tuple[str, ...] = ("buy", "sell", "hold")
    horizon: str = "medium"
    cheap_threshold_ps: float = 3.0
    expensive_threshold_ps: float = 8.0
    cheap_threshold_pe: float = 15.0
    expensive_threshold_pe: float = 30.0
    cheap_threshold_pfcf: float = 12.0
    expensive_threshold_pfcf: float = 25.0
    use_llm: bool = True
    use_sentiment: bool = False
    sentiment_weight: float = 0.20
    rag_top_k: int = 4


class ValuationAgent:
    def __init__(
        self,
        llm: LLMBackend,
        config: Optional[ValuationAgentConfig] = None,
        filing_rag: Optional[FilingRAG] = None,
    ):
        self.llm = llm
        self.cfg = config or ValuationAgentConfig()
        self.filing_rag = filing_rag

    @staticmethod
    def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or abs(b) < 1e-12:
            return None
        return float(a / b)

    def compute_metrics(self, vin: ValuationInputs) -> Dict[str, Optional[float]]:
        q = vin.quarter_table.dropna(subset=["end"]).sort_values("end")
        if q.empty:
            return {
                "ttm_revenue": None,
                "ttm_op_income": None,
                "ttm_net_income": None,
                "ttm_fcf": None,
                "shares_outstanding": None,
                "market_cap": None,
                "ps": None,
                "pe": None,
                "p_fcf": None,
            }

        eligible = q[q["filed"].notna() & (q["filed"] <= vin.asof)]
        if eligible.empty:
            eligible = q
        last_end = eligible["end"].max()

        ttm = ttm_from_quarters(q, pd.Timestamp(last_end))

        # Point-in-time market cap using point-in-time shares if possible
        shares = ttm.get("shares_outstanding")
        if shares is None:
            shares = vin.shares_outstanding

        market_cap = None
        if shares is not None:
            market_cap = float(vin.price) * float(shares)
        elif vin.market_cap is not None:
            market_cap = float(vin.market_cap)

        ps = self._safe_div(market_cap, ttm["ttm_revenue"])
        pe = self._safe_div(market_cap, ttm["ttm_net_income"])
        p_fcf = self._safe_div(market_cap, ttm["ttm_fcf"])

        return {
            **ttm,
            "market_cap": market_cap,
            "ps": ps,
            "pe": pe,
            "p_fcf": p_fcf,
        }

    @staticmethod
    def _sentiment_component(vin: ValuationInputs) -> float:
        """Combine FinBERT and lexical sentiment onto a bounded [-1, 1] scale."""
        vals: List[float] = []
        if vin.finbert_sentiment is not None:
            vals.append(float(max(-1.0, min(1.0, vin.finbert_sentiment))))
        if vin.lexical_sentiment is not None:
            # Lexical values are small; smooth-scale into comparable range.
            vals.append(float(np.tanh(vin.lexical_sentiment * 50.0)))
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _rule_prior(self, metrics: Dict[str, Optional[float]], vin: Optional[ValuationInputs] = None) -> float:
        score = 0.0
        n = 0

        ps = metrics.get("ps")
        if isinstance(ps, (int, float)) and ps is not None:
            n += 1
            if ps <= self.cfg.cheap_threshold_ps:
                score += 0.35
            elif ps >= self.cfg.expensive_threshold_ps:
                score -= 0.35

        pe = metrics.get("pe")
        if isinstance(pe, (int, float)) and pe is not None:
            n += 1
            if pe <= self.cfg.cheap_threshold_pe:
                score += 0.35
            elif pe >= self.cfg.expensive_threshold_pe:
                score -= 0.35

        p_fcf = metrics.get("p_fcf")
        if isinstance(p_fcf, (int, float)) and p_fcf is not None:
            n += 1
            if p_fcf <= self.cfg.cheap_threshold_pfcf:
                score += 0.30
            elif p_fcf >= self.cfg.expensive_threshold_pfcf:
                score -= 0.30

        base = 0.0 if n == 0 else float(score)

        if self.cfg.use_sentiment and vin is not None:
            s_comp = self._sentiment_component(vin)
            base += float(self.cfg.sentiment_weight) * s_comp

        return max(-1.0, min(1.0, float(base)))

    @staticmethod
    def _parse_key_lines(raw: str) -> Dict[str, str]:
        keys = ["ACTION", "CONFIDENCE", "SCORE", "THESIS", "POINTS", "RISKS"]
        parsed = {k: "" for k in keys}
        for line in (raw or "").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().upper()
            if k in parsed:
                parsed[k] = v.strip()
        return parsed

    @staticmethod
    def _to_float(x: str, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _retrieve_filing_context(self, vin: ValuationInputs) -> str:
        if self.filing_rag is None:
            return "(no filing RAG attached)"

        chunks = self.filing_rag.retrieve(
            self.llm,
            ticker=vin.ticker,
            query=(
                f"{vin.ticker} valuation, margins, growth, risks, cash flow, "
                f"capital allocation, guidance"
            ),
            asof=vin.asof,
            top_k=self.cfg.rag_top_k,
        )
        if not chunks:
            return "(no filing text retrieved)"

        return "\n\n".join(
            f"[{c.source} filed={c.filed.date()} doc={c.doc_id}] {c.text}"
            for c in chunks
        )

    def decide(self, vin: ValuationInputs) -> ValuationDecision:
        metrics = self.compute_metrics(vin)
        prior = self._rule_prior(metrics, vin)
        filing_context = self._retrieve_filing_context(vin)

        history_pairs = []
        if vin.recent_price_dates and vin.recent_prices_5d:
            for d, p in zip(vin.recent_price_dates, vin.recent_prices_5d):
                history_pairs.append(f"{d}: {p:.2f}")
        history_block = "; ".join(history_pairs) if history_pairs else "(missing)"

        if getattr(self.cfg, "use_llm", True) is False:
            action = "buy" if prior > 0.25 else ("sell" if prior < -0.25 else "hold")
            return ValuationDecision(
                ticker=vin.ticker,
                asof=vin.asof,
                action=action,
                confidence=0.55,
                score=prior,
                thesis="LLM disabled: using rule-based valuation prior only.",
                key_points=[
                    f"P/S={metrics.get('ps')}",
                    f"P/E={metrics.get('pe')}",
                    f"P/FCF={metrics.get('p_fcf')}",
                ],
                risks=["LLM disabled; qualitative context missing."],
                metrics=metrics,
            )

        prompt = f"""
You are a valuation-focused equity analyst.

Ticker: {vin.ticker}
As-of date: {vin.asof.date()}
Price: {vin.price:.2f}
Horizon: {self.cfg.horizon}

Point-in-time valuation metrics:
- Market Cap: {metrics.get("market_cap")}
- TTM Revenue: {metrics.get("ttm_revenue")}
- TTM Operating Income: {metrics.get("ttm_op_income")}
- TTM Net Income: {metrics.get("ttm_net_income")}
- TTM Free Cash Flow: {metrics.get("ttm_fcf")}
- Shares Outstanding: {metrics.get("shares_outstanding")}
- P/S: {metrics.get("ps")}
- P/E: {metrics.get("pe")}
- P/FCF: {metrics.get("p_fcf")}

Rule-based valuation prior (range -1..+1): {prior}

Sentiment inputs:
- FinBERT sentiment: {vin.finbert_sentiment}
- Lexical sentiment: {vin.lexical_sentiment}

Recent price history (last 5 trading days):
{history_block}

Retrieved filing context:
{filing_context}

Task:
Assess whether the stock is undervalued, fairly valued, or overvalued based on valuation,
cash generation, profitability, and management commentary. Be conservative.

Output exactly:
ACTION: <buy|sell|hold>
CONFIDENCE: <0..1 float>
SCORE: <-1..+1 float>
THESIS: <1-3 sentences>
POINTS: <semicolon-separated>
RISKS: <semicolon-separated>
""".strip()

        resp = self.llm.chat([
            {"role": "system", "content": "Be precise, numerate, and conservative."},
            {"role": "user", "content": prompt},
        ])

        raw = (resp.get("content") or "").strip()
        p = self._parse_key_lines(raw)

        action = p["ACTION"].lower()
        if action not in self.cfg.output_actions:
            action = "buy" if prior > 0.25 else ("sell" if prior < -0.25 else "hold")

        confidence = max(0.0, min(1.0, self._to_float(p["CONFIDENCE"], 0.55)))
        score = max(-1.0, min(1.0, self._to_float(p["SCORE"], prior)))

        thesis = p["THESIS"] or "Valuation signal is weak or incomplete; staying neutral."
        points = [x.strip() for x in (p["POINTS"] or "").split(";") if x.strip()]
        risks = [x.strip() for x in (p["RISKS"] or "").split(";") if x.strip()]

        return ValuationDecision(
            ticker=vin.ticker,
            asof=vin.asof,
            action=action,
            confidence=confidence,
            score=score,
            thesis=thesis,
            key_points=points,
            risks=risks,
            metrics=metrics,
        )