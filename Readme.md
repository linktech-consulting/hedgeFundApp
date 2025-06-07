# hedgeFundApp
Experimental Test For Wealth Management
This is demo version of Wealth Management Dashboard for Portfolio Managers

## Environment setup
1. `python -m venv venv`
2. `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. `pip install -r requirements.txt`
4. `streamlit run Dashboard.py`

The aim of this project is to provide detailed analysis of stock news for the users so that they can make timely decisions in order to manage portfolios well.

## Flask portfolio analysis
A simple Flask application is included in `app.py`. Start it with:

```bash
python app.py
```

Send a POST request to `/analyze` with a JSON payload containing a list of `tickers`, optional `weights`, and a `benchmark` symbol. The service downloads historical prices using `yfinance` and returns portfolio metrics compared to the benchmark.
