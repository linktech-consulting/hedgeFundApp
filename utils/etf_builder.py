import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def fetch_close_history(ticker, start, end):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end)
        if 'Close' in hist.columns:
            return hist['Close']
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to fetch data for {ticker}: {e}")
        return None

def fetch_market_caps(tickers):
    market_caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap", None)
            if mc is not None:
                market_caps[ticker] = mc
        except Exception as e:
            st.warning(f"Failed to fetch market cap for {ticker}: {e}")
    return market_caps

def fetch_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        eps = info.get('trailingEps', None)
        pe = info.get('trailingPE', None)
        earnings_growth = info.get('earningsQuarterlyGrowth', None)  # quarterly growth rate
        return {'EPS': eps, 'PE': pe, 'EarningsGrowth': earnings_growth}
    except Exception as e:
        st.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {'EPS': None, 'PE': None, 'EarningsGrowth': None}

@st.cache_data
def load_symbols(csv_path):
    df = pd.read_csv(csv_path)
    symbols = [symbol + ".NS" for symbol in df['SYMBOL'].dropna().unique().tolist()]
    return symbols

def weighted_average(series, weights):
    mask = series.notna()
    if mask.sum() == 0:
        return np.nan
    w = np.array(weights)[mask]
    w /= w.sum()
    return np.dot(series[mask], w)

def format_metric(val, is_percent=False):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    return f"{val:.2%}" if is_percent else f"{val:.2f}"

def etf_builder_dashboard():
    st.title("ETF Builder Module")

    csv_path = "EQUITY_L.csv"  # Adjust path if needed
    stock_list = load_symbols(csv_path)
    selected_stocks = st.multiselect("Select stocks to build your ETF:", stock_list)

    if not selected_stocks:
        st.warning("Please select at least one stock.")
        return

    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    weight_method = st.radio(
        "Select portfolio weighting method",
        ("Equal Weight", "Market Cap Weight")
    )

    close_data = {}
    missing_tickers = []
    for ticker in selected_stocks:
        data = fetch_close_history(ticker, start_date, end_date)
        if data is None or data.empty:
            missing_tickers.append(ticker)
        else:
            close_data[ticker] = data

    if missing_tickers:
        st.warning(f"Close price data missing for tickers: {', '.join(missing_tickers)}")

    if not close_data:
        st.error("No valid Close price data found for selected tickers.")
        return

    data = pd.DataFrame(close_data).dropna()
    returns = data.pct_change().dropna()
    valid_tickers = list(close_data.keys())

    if weight_method == "Equal Weight":
        weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))
    else:
        market_caps = fetch_market_caps(valid_tickers)
        valid_caps_tickers = list(market_caps.keys())
        if not valid_caps_tickers:
            st.error("Market cap data not found for selected stocks. Using equal weights.")
            weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))
        else:
            data = data[valid_caps_tickers]
            returns = returns[valid_caps_tickers]
            valid_tickers = valid_caps_tickers
            total_market_cap = sum(market_caps.values())
            weights = np.array([market_caps[t] / total_market_cap for t in valid_tickers])

    st.write("Portfolio Weights:")
    for t, w in zip(valid_tickers, weights):
        st.write(f"{t}: {w:.2%}")

    portfolio_returns = returns.dot(weights)

    nifty_ticker = yf.Ticker("^NSEI")
    nifty_hist = nifty_ticker.history(start=start_date, end=end_date)
    if 'Close' not in nifty_hist.columns or nifty_hist['Close'].empty:
        st.error("NIFTY 50 Close price data not found.")
        return
    nifty = nifty_hist['Close'].dropna()
    nifty_returns = nifty.pct_change().dropna()

    combined_index = portfolio_returns.index.intersection(nifty_returns.index)
    portfolio_returns = portfolio_returns.loc[combined_index]
    nifty_returns = nifty_returns.loc[combined_index]

    cum_portfolio = (1 + portfolio_returns).cumprod()
    cum_nifty = (1 + nifty_returns).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cum_portfolio, label="ETF Portfolio")
    ax.plot(cum_nifty, label="NIFTY 50")
    ax.set_title("Cumulative Returns Comparison")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    st.pyplot(fig)

    # Fundamentals fetch & weighted average
    fundamentals = []
    for ticker in valid_tickers:
        fundamentals.append(fetch_fundamentals(ticker))
    fund_df = pd.DataFrame(fundamentals, index=valid_tickers)

    avg_eps = weighted_average(fund_df['EPS'], weights)
    avg_pe = weighted_average(fund_df['PE'], weights)
    avg_growth = weighted_average(fund_df['EarningsGrowth'], weights)

    # NIFTY fundamentals placeholders (replace with real if available)
    nifty_eps = 90.0
    nifty_pe = 25.0
    nifty_growth = 0.12

    st.write("### Fundamental Metrics Comparison")

    fund_metrics = {
        "Metric": ["EPS", "PE Ratio", "Quarterly Earnings Growth"],
        "ETF Portfolio": [
            format_metric(avg_eps),
            format_metric(avg_pe),
            format_metric(avg_growth, is_percent=True)
        ],
        "NIFTY 50 (Estimate)": [
            format_metric(nifty_eps),
            format_metric(nifty_pe),
            format_metric(nifty_growth, is_percent=True)
        ]
    }
    fund_df_compare = pd.DataFrame(fund_metrics)
    st.table(fund_df_compare)

    # Performance metrics functions
    def annualized_return(r, periods_per_year=252):
        return (1 + r).prod() ** (periods_per_year / len(r)) - 1

    def annualized_volatility(r, periods_per_year=252):
        return r.std() * np.sqrt(periods_per_year)

    def sharpe_ratio(r, risk_free_rate=0, periods_per_year=252):
        ann_ret = annualized_return(r, periods_per_year)
        ann_vol = annualized_volatility(r, periods_per_year)
        if ann_vol == 0:
            return np.nan
        return (ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(r):
        cum_return = (1 + r).cumprod()
        peak = cum_return.cummax()
        epsilon = 1e-10
        drawdown = (cum_return - peak) / (peak + epsilon)
        return drawdown.min()

    metrics = {
        "ETF Portfolio": {
            "Annualized Return": annualized_return(portfolio_returns),
            "Annualized Volatility": annualized_volatility(portfolio_returns),
            "Sharpe Ratio": sharpe_ratio(portfolio_returns),
            "Max Drawdown": max_drawdown(portfolio_returns),
        },
        "NIFTY 50": {
            "Annualized Return": annualized_return(nifty_returns),
            "Annualized Volatility": annualized_volatility(nifty_returns),
            "Sharpe Ratio": sharpe_ratio(nifty_returns),
            "Max Drawdown": max_drawdown(nifty_returns),
        }
    }

    perf_df = pd.DataFrame(metrics)
    st.write("### Performance Metrics")
    st.dataframe(perf_df.style.format("{:.2%}"))

    corr = portfolio_returns.corr(nifty_returns)
    st.write(f"Correlation between ETF portfolio and NIFTY 50 returns: {corr:.2f}")
