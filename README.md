# NLP_earning_calls
NLP sentiment indicator based on earning calls transcripts of S&P 500 companies.

## Project Structure

```text
NLP_earning_calls/
├── data/
│   ├── raw/
│   │   └── earnings_calls/
│   └── processed/
├── notebooks/
├── src/
│   └── data/
│       ├── fetch_market_data.py
│       └── fetch_earnings_call_transcript.py
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Fetch Earnings Call Transcript (Alpha Vantage)

Set your API key:

```powershell
$env:ALPHAVANTAGE_API_KEY="YOUR_API_KEY"
```

Run the script:

```powershell
python src/data/fetch_earnings_call_transcript.py --symbol IBM --quarter 2024Q1
```

It will print the API response and save it to:

`data/raw/earnings_calls/IBM_2024Q1.json`
