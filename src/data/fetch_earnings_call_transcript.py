"""Download earnings call transcripts from Alpha Vantage and save JSON output.

Batch Mode (All 2024 Release Dates): 
    python src/data/fetch_earnings_call_transcript.py
    
    Discovers all earnings released in 2024 for the configured symbol list,
    fetches each transcript, and saves as SYMBOL_YYYY-MM-DD.json
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests


BASE_URL = "https://www.alphavantage.co/query"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "earnings_calls"


def _is_rate_limited_payload(data: dict) -> bool:
    """Return True if Alpha Vantage response indicates free-tier rate limiting."""
    info = str(data.get("Information", ""))
    note = str(data.get("Note", ""))
    text = f"{info} {note}".lower()
    return "api call frequency" in text or "rate limit" in text


def _month_to_quarter(month: int) -> int:
    """Map month number to quarter number."""
    if 1 <= month <= 3:
        return 1
    if 4 <= month <= 6:
        return 2
    if 7 <= month <= 9:
        return 3
    return 4


def fetch_earnings_call_transcript(symbol: str, quarter: str, api_key: str) -> dict:
    """Fetch a single quarter earnings call transcript payload from Alpha Vantage."""
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": symbol.upper(),
        "quarter": quarter,
        "apikey": api_key,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    if not data:
        # Empty payloads can occur under free-tier pressure; wait and retry once.
        time.sleep(15)
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if _is_rate_limited_payload(data):
        raise RuntimeError(data["Information"])

    return data


def get_reported_dates_for_year(symbol: str, year: int, api_key: str) -> list[tuple[str, str]]:
    """Fetch all earnings with reported dates in the given calendar year.

    Returns a list of tuples: (reportedDate, fiscalDateEnding) for all entries
    where reportedDate falls between {year}-01-01 and {year}-12-31.
    """
    params = {
        "function": "EARNINGS",
        "symbol": symbol.upper(),
        "apikey": api_key,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data or _is_rate_limited_payload(data):
        # Free-tier safety retry path when response is empty or rate-limited.
        time.sleep(15)
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if _is_rate_limited_payload(data):
        raise RuntimeError(data.get("Information") or data.get("Note") or "Rate limited")

    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"

    results = []
    for item in data.get("quarterlyEarnings", []):
        reported_date = item.get("reportedDate")
        fiscal_date = item.get("fiscalDateEnding")

        if not reported_date or not fiscal_date:
            continue

        # Check if reportedDate is within the calendar year range
        if year_start <= reported_date <= year_end:
            results.append((reported_date, fiscal_date))

    return results


def save_transcript_json(data: dict, symbol: str, reported_date: str) -> Path:
    """Persist API response to data/raw/earnings_calls/SYMBOL_YYYY-MM-DD.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{symbol.upper()}_{reported_date}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path


def main() -> None:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Set ALPHAVANTAGE_API_KEY environment variable."
        )

    symbols = [
        "ACN",   # Accenture
        "ADBE",  # Adobe
        "AMD",   # Advanced Micro Devices
        "AAPL",  # Apple
        "AMAT",  # Applied Materials
        "AVGO",  # Broadcom
        "CSCO",  # Cisco Systems
        "INTC",  # Intel
        "IBM",   # IBM
        "INTU",  # Intuit
        "MU",    # Micron Technology
        "MSFT",  # Microsoft
        "NVDA",  # NVIDIA
        "ORCL",  # Oracle
        "PLTR",  # Palantir Technologies
        "QCOM",  # QUALCOMM
        "CRM",   # Salesforce
        "NOW",   # ServiceNow
        "TXN",   # Texas Instruments
    ]
    year = 2024

    total_transcripts = 0
    completed = 0
    failed = 0

    print(f"Fetching all earnings calls released in {year} for {len(symbols)} symbols...")
    print()

    for symbol in symbols:
        print(f"Discovering {year} release dates for {symbol}...")

        try:
            # Discover all {year} earnings release dates for this symbol
            date_tuples = get_reported_dates_for_year(symbol, year, api_key)
            total_transcripts += len(date_tuples)

            if not date_tuples:
                print(f"  No {year} earnings found for {symbol}")
                print()
                continue

            print(f"  Found {len(date_tuples)} earnings release(s) in {year}")

            # Rate limit: pause before starting transcript fetch loop
            time.sleep(20)

            # Fetch transcript for each discovered date
            for reported_date, fiscal_date in date_tuples:
                completed += 1

                # Calculate quarter from fiscal_date
                try:
                    fiscal_dt = datetime.strptime(fiscal_date, "%Y-%m-%d")
                except ValueError:
                    print(f"  ✗ Invalid fiscal date format: {fiscal_date}")
                    failed += 1
                    continue

                quarter_num = _month_to_quarter(fiscal_dt.month)
                quarter_key = f"{fiscal_dt.year}Q{quarter_num}"

                print(f"  [{completed}] Fetching {symbol} {reported_date} (fiscal: {quarter_key})...")

                try:
                    # Fetch transcript using quarter derived from fiscal date
                    data = fetch_earnings_call_transcript(symbol, quarter_key, api_key)
                    data["reported_date"] = reported_date

                    # Save with reported_date in filename
                    output_path = save_transcript_json(data, symbol, reported_date)
                    print(f"    ✓ Saved to {output_path}")

                except Exception as e:
                    failed += 1
                    print(f"    ✗ Error: {type(e).__name__}: {str(e)}")

                # Rate limit: pause between calls (20s for free tier: 5 calls/min)
                time.sleep(20)

            print()

        except Exception as e:
            print(f"  ✗ Failed to discover dates: {type(e).__name__}: {str(e)}")
            print()

    print(f"Batch complete: {completed}/{total_transcripts} transcripts fetched, {failed} failed.")


if __name__ == "__main__":
    main()
