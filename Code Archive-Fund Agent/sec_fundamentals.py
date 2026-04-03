#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:47:28 2026

@author: fabriziocoiai
"""
# sec_fundamentals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import pandas as pd
import requests


SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


@dataclass
class SecConfig:
    user_agent: str
    sleep_seconds: float = 0.2
    timeout: int = 30


def _cik10(cik: str) -> str:
    return str(cik).zfill(10)


def _sec_get_json(url: str, cfg: SecConfig) -> Dict[str, Any]:
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    r = requests.get(url, headers=headers, timeout=cfg.timeout)
    r.raise_for_status()
    time.sleep(cfg.sleep_seconds)
    return r.json()


def ticker_to_cik(ticker: str, cfg: SecConfig) -> str:
    data = _sec_get_json(SEC_TICKER_CIK_URL, cfg)
    t = ticker.upper()
    for _, row in data.items():
        if str(row.get("ticker", "")).upper() == t:
            return str(int(row["cik_str"]))
    raise ValueError(f"Ticker {ticker} not found in SEC mapping.")


def fetch_companyfacts(cik: str, cfg: SecConfig) -> Dict[str, Any]:
    url = SEC_COMPANYFACTS_URL.format(cik10=_cik10(cik))
    return _sec_get_json(url, cfg)


def _extract_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str, unit: str) -> pd.DataFrame:
    facts = companyfacts.get("facts", {}).get(taxonomy, {}).get(tag, {})
    units = facts.get("units", {})
    vals = units.get(unit, [])

    rows: List[Dict[str, Any]] = []
    for x in vals:
        end = x.get("end")
        val = x.get("val")
        if end is None or val is None:
            continue
        rows.append({
            "end": pd.Timestamp(end),
            "filed": pd.Timestamp(x["filed"]) if x.get("filed") else pd.NaT,
            "val": float(val),
            "form": x.get("form"),
            "fy": x.get("fy"),
            "fp": x.get("fp"),
            "frame": x.get("frame"),
        })

    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val", "form", "fy", "fp", "frame"])
    return pd.DataFrame(rows).sort_values(["end", "filed"]).reset_index(drop=True)


def _extract_usd_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str) -> pd.DataFrame:
    return _extract_series(companyfacts, taxonomy, tag, "USD")


def _extract_shares_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str) -> pd.DataFrame:
    # Most share counts use "shares" as the unit
    return _extract_series(companyfacts, taxonomy, tag, "shares")


def _latest_per_end(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Keep the earliest filing for each period end to preserve point-in-time behavior
    # and avoid selecting restated values filed years later.
    return (
        df.sort_values(["end", "filed"], na_position="last")
        .groupby("end", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _first_nonempty_tag(
    companyfacts: Dict[str, Any],
    taxonomy: str,
    tags: List[str],
    unit: str = "USD",
) -> pd.DataFrame:
    for tag in tags:
        if unit == "USD":
            df = _latest_per_end(_extract_usd_series(companyfacts, taxonomy, tag))
        elif unit == "shares":
            df = _latest_per_end(_extract_shares_series(companyfacts, taxonomy, tag))
        else:
            raise ValueError(f"Unsupported unit: {unit}")

        if not df.empty:
            print(f"Using tag: {tag}")
            return df

    return pd.DataFrame(columns=["end", "filed", "val", "form", "fy", "fp", "frame"])


def _quarter_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer 10-Q rows, but keep 10-K if nothing else exists.
    """
    if df.empty:
        return df

    q_mask = df["form"].astype(str).str.contains("10-Q", case=False, na=False)
    q_df = df[q_mask].copy()
    if not q_df.empty:
        return q_df.sort_values(["end", "filed"]).reset_index(drop=True)

    return df.sort_values(["end", "filed"]).reset_index(drop=True)


def _rename(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["end", col, f"{col}_filed", f"{col}_fy", f"{col}_fp", f"{col}_form"])
    out = df[["end", "filed", "val", "fy", "fp", "form"]].copy()
    out = out.rename(columns={
        "val": col,
        "filed": f"{col}_filed",
        "fy": f"{col}_fy",
        "fp": f"{col}_fp",
        "form": f"{col}_form",
    })
    return out


def ytd_to_quarterly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convert YTD flow data into true quarterly flow data.

    Assumptions:
    - Q1 is already quarter-only
    - Q2 = Q2_YTD - Q1_YTD
    - Q3 = Q3_YTD - Q2_YTD
    - FY = FY_YTD - Q3_YTD
    """
    if df.empty:
        return df

    work = df.copy().sort_values(["fy", "end"]).reset_index(drop=True)
    out_col = f"{value_col}_quarter"
    work[out_col] = work[value_col]

    for fy, grp in work.groupby("fy", dropna=False):
        grp = grp.sort_values("end")
        prev_ytd = None

        for idx, row in grp.iterrows():
            fp = str(row.get("fp", "")).upper()
            val = row[value_col]

            if pd.isna(val):
                work.loc[idx, out_col] = pd.NA
                continue

            if fp == "Q1":
                work.loc[idx, out_col] = val
                prev_ytd = val
            elif fp in ("Q2", "Q3", "FY", "Q4"):
                if prev_ytd is not None:
                    work.loc[idx, out_col] = val - prev_ytd
                else:
                    work.loc[idx, out_col] = pd.NA
                prev_ytd = val
            else:
                # fallback: leave as-is
                work.loc[idx, out_col] = val

    return work


def build_quarter_table(ticker: str, cfg: SecConfig) -> pd.DataFrame:
    """
    Output columns:
      end, filed, revenue, op_income, net_income, ocf, capex, fcf, shares_outstanding
    """
    cik = ticker_to_cik(ticker, cfg)
    facts = fetch_companyfacts(cik, cfg)

    # Revenue with fallbacks
    rev = _first_nonempty_tag(
        facts,
        "us-gaap",
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ],
        unit="USD",
    )

    op_inc = _extract_usd_series(facts, "us-gaap", "OperatingIncomeLoss")
    net_inc = _extract_usd_series(facts, "us-gaap", "NetIncomeLoss")
    ocf = _extract_usd_series(facts, "us-gaap", "NetCashProvidedByUsedInOperatingActivities")
    capex = _extract_usd_series(facts, "us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")

    # Shares from DEI with fallbacks
    shares = _first_nonempty_tag(
        facts,
        "dei",
        [
            "EntityCommonStockSharesOutstanding",
            "EntityCommonStockSharesIssued",
        ],
        unit="shares",
    )

    rev = _latest_per_end(_quarter_only(rev))
    op_inc = _latest_per_end(_quarter_only(op_inc))
    net_inc = _latest_per_end(_quarter_only(net_inc))
    ocf = _latest_per_end(_quarter_only(ocf))
    capex = _latest_per_end(_quarter_only(capex))
    shares = _latest_per_end(shares.sort_values(["end", "filed"]))

    rev_r = _rename(rev, "revenue")
    op_r = _rename(op_inc, "op_income")
    net_r = _rename(net_inc, "net_income")
    ocf_r = _rename(ocf, "ocf")
    capex_r = _rename(capex, "capex")
    shares_r = _rename(shares, "shares_outstanding")

    table = rev_r
    for piece in [op_r, net_r, ocf_r, capex_r, shares_r]:
        table = table.merge(piece, on="end", how="outer")

    # One filing date per quarter
    table["filed"] = pd.NaT
    for col in [
        "revenue_filed",
        "net_income_filed",
        "op_income_filed",
        "ocf_filed",
        "capex_filed",
        "shares_outstanding_filed",
    ]:
        if col in table.columns:
            table["filed"] = table["filed"].fillna(table[col])

    # Carry fy/fp for OCF/CAPEX so YTD conversion works
    table["fy"] = table.get("ocf_fy")
    table["fp"] = table.get("ocf_fp")

    # Convert OCF and CAPEX from YTD to quarter-only
    ocf_tmp = table[["end", "fy", "fp", "ocf"]].copy()
    capex_tmp = table[["end", "fy", "fp", "capex"]].copy()

    ocf_tmp = ytd_to_quarterly(ocf_tmp, "ocf")
    capex_tmp = ytd_to_quarterly(capex_tmp, "capex")

    table = table.merge(
        ocf_tmp[["end", "ocf_quarter"]],
        on="end",
        how="left",
    ).merge(
        capex_tmp[["end", "capex_quarter"]],
        on="end",
        how="left",
    )

    # Use quarter-only versions for FCF
    table["fcf"] = pd.NA
    mask = table["ocf_quarter"].notna() & table["capex_quarter"].notna()
    table.loc[mask, "fcf"] = table.loc[mask, "ocf_quarter"] - table.loc[mask, "capex_quarter"]

    # Optionally overwrite displayed ocf/capex with quarter-only values
    table["ocf"] = table["ocf_quarter"]
    table["capex"] = table["capex_quarter"]

    keep_cols = [
        "end",
        "revenue_filed", "revenue",
        "op_income_filed", "op_income",
        "net_income_filed", "net_income",
        "ocf_filed", "ocf",
        "capex_filed", "capex",
        "shares_outstanding_filed", "shares_outstanding",
        "filed",
        "fcf",
        "fy",
        "fp",
    ]
    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()

    table["end"] = pd.to_datetime(table["end"])
    table["filed"] = pd.to_datetime(table["filed"], errors="coerce")

    for col in ["revenue", "op_income", "net_income", "ocf", "capex", "fcf", "shares_outstanding"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    table = table.dropna(subset=["revenue", "op_income", "net_income", "ocf", "capex", "fcf"], how="all")
    table = table.sort_values("end").reset_index(drop=True)
    return table


def ttm_from_quarters(q: pd.DataFrame, asof_end: pd.Timestamp) -> Dict[str, Optional[float]]:
    if q.empty:
        return {
            "ttm_revenue": None,
            "ttm_op_income": None,
            "ttm_net_income": None,
            "ttm_ocf": None,
            "ttm_fcf": None,
            "shares_outstanding": None,
        }

    q2 = q.copy()
    q2["end"] = pd.to_datetime(q2["end"])
    q2 = q2[q2["end"] <= pd.Timestamp(asof_end)].sort_values("end")

    if q2.empty:
        return {
            "ttm_revenue": None,
            "ttm_op_income": None,
            "ttm_net_income": None,
            "ttm_ocf": None,
            "ttm_fcf": None,
            "shares_outstanding": None,
        }

    last4 = q2.tail(4).copy()

    def ttm_sum(col: str) -> Optional[float]:
        if col not in q2.columns:
            return None
        vals = q2[col].dropna().tail(4)
        if len(vals) < 4:
            return None
        return float(vals.sum())

    shares = None
    if "shares_outstanding" in q2.columns:
        s = q2["shares_outstanding"].dropna()
        if len(s) > 0:
            shares = float(s.iloc[-1])

    return {
        "ttm_revenue": ttm_sum("revenue"),
        "ttm_op_income": ttm_sum("op_income"),
        "ttm_net_income": ttm_sum("net_income"),
        "ttm_ocf": ttm_sum("ocf"),
        "ttm_fcf": ttm_sum("fcf"),
        "shares_outstanding": shares,
    }


def build_filing_date_events(q: pd.DataFrame) -> List[pd.Timestamp]:
    if q.empty or "filed" not in q.columns:
        return []
    filed = (
        pd.to_datetime(q["filed"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [pd.Timestamp(x) for x in filed]