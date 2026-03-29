"""
Fetch historical price and market cap data for a given ticker from Yahoo Finance
and save the results to data/raw/.

Usage:
    python src/data/fetch_market_data.py
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* between *start* and *end* (inclusive).

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "AAPL".
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format (exclusive in yfinance, so add 1 day if needed).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Open, High, Low, Close, Volume, Dividends, Stock Splits.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=True)
    df.index = df.index.tz_localize(None)  # strip timezone for clean CSV output
    return df


def fetch_market_cap_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Compute daily market capitalisation for *ticker*.

    Market cap = Adjusted Close * Shares Outstanding.
    Shares outstanding is a point-in-time figure from Yahoo Finance info and is
    applied uniformly; for higher precision use a dedicated fundamentals provider.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Close and MarketCap.
    """
    tk = yf.Ticker(ticker)
    price_df = tk.history(start=start, end=end, auto_adjust=True)
    price_df.index = price_df.index.tz_localize(None)

    shares_outstanding = tk.info.get("sharesOutstanding")
    if shares_outstanding is None:
        raise ValueError(
            f"Could not retrieve sharesOutstanding for {ticker} from Yahoo Finance."
        )

    market_cap_df = price_df[["Close"]].copy()
    market_cap_df["MarketCap"] = market_cap_df["Close"] * shares_outstanding
    return market_cap_df


def save_data(df: pd.DataFrame, filename: str) -> Path:
    """Save *df* to ``data/raw/<filename>`` as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    filename : str
        Output filename, e.g. "AAPL_price.csv".

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / filename
    df.to_csv(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download historical price and market cap data from Yahoo Finance."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (defaults to today)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Fetching price data for {ticker} from {args.start} to {args.end} …")
    price_df = fetch_price_data(ticker, args.start, args.end)
    price_path = save_data(price_df, f"{ticker}_price.csv")
    print(f"  Saved {len(price_df)} rows → {price_path}")

    print(f"Fetching market cap data for {ticker} …")
    market_cap_df = fetch_market_cap_data(ticker, args.start, args.end)
    market_cap_path = save_data(market_cap_df, f"{ticker}_market_cap.csv")
    print(f"  Saved {len(market_cap_df)} rows → {market_cap_path}")

    print("Done.")


if __name__ == "__main__":
    main()
