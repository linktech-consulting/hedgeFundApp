import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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

def portfolio_analyzer():
    st.title("📊 Portfolio Analyzer")

    uploaded_file = st.file_uploader("Upload your portfolio CSV file", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file with your portfolio data.")
        return

    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required_cols = {"Ticker", "Quantity"}
    if not required_cols.issubset(df.columns.str.strip()):
        st.error(f"Uploaded file must contain columns: {required_cols}")
        return

    df['Ticker'] = df['Ticker'].str.strip().apply(lambda x: x if x.endswith(".NS") else x + ".NS")

    st.write("### Uploaded Portfolio")
    st.dataframe(df)

    def fetch_price(ticker):
        try:
            price = yf.Ticker(ticker).history(period="1d")['Close'][-1]
            return price
        except Exception:
            return np.nan

    with st.spinner("Fetching current prices..."):
        df['Current Price'] = df['Ticker'].apply(fetch_price)
    df.dropna(subset=['Current Price'], inplace=True)

    df['Market Value'] = df['Quantity'] * df['Current Price']
    total_value = df['Market Value'].sum()
    df['Weight'] = df['Market Value'] / total_value

    # Summary cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"₹{total_value:,.2f}")
    col2.metric("Number of Holdings", len(df))
    col3.metric("Highest Weight", f"{df['Weight'].max():.2%}")

    st.markdown("---")

    st.write("### Portfolio Holdings")
    st.dataframe(df[['Ticker', 'Quantity', 'Current Price', 'Market Value', 'Weight']].style.format({
        'Current Price': '₹{:.2f}',
        'Market Value': '₹{:,.2f}',
        'Weight': '{:.2%}'
    }))

    # Portfolio allocation pie chart
    fig1 = px.pie(df, values='Weight', names='Ticker', title='Portfolio Allocation')
    st.plotly_chart(fig1, use_container_width=True)

    # Fetch 1-year historical prices
    st.write("### Portfolio Returns vs NIFTY")

    price_histories = {}
    with st.spinner("Fetching historical prices..."):
        for ticker in df['Ticker']:
            try:
                price_histories[ticker] = yf.Ticker(ticker).history(period="1y")['Close']
            except Exception:
                st.warning(f"Failed to fetch historical data for {ticker}")

    if not price_histories:
        st.warning("No historical data fetched for returns analysis.")
        return

    prices_df = pd.DataFrame(price_histories).dropna()
    returns_df = prices_df.pct_change().dropna()
    weights = df.set_index('Ticker').loc[returns_df.columns, 'Weight'].values
    portfolio_returns = returns_df.dot(weights)

    # Fetch NIFTY returns
    nifty_ticker = yf.Ticker("^NSEI")
    nifty_hist = nifty_ticker.history(period="1y")['Close'].dropna()
    nifty_returns = nifty_hist.pct_change().dropna()

    # Align indexes
    combined_index = portfolio_returns.index.intersection(nifty_returns.index)
    portfolio_returns = portfolio_returns.loc[combined_index]
    nifty_returns = nifty_returns.loc[combined_index]

    # Cumulative returns plot
    cum_portfolio = (1 + portfolio_returns).cumprod() - 1
    cum_nifty = (1 + nifty_returns).cumprod() - 1

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(cum_portfolio.index, cum_portfolio.values, label="Portfolio")
    ax2.plot(cum_nifty.index, cum_nifty.values, label="NIFTY 50")
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title("Cumulative Returns (1 Year)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Histogram of daily returns
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.hist(portfolio_returns, bins=50, alpha=0.75, label="Portfolio Returns")
    ax3.hist(nifty_returns, bins=50, alpha=0.5, label="NIFTY Returns")
    ax3.set_title("Histogram of Daily Returns")
    ax3.set_xlabel("Daily Return")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    st.pyplot(fig3)

    # Show performance metrics side-by-side
    perf_metrics = {
        "Metric": ["Annualized Return", "Annualized Volatility", "Sharpe Ratio"],
        "Portfolio": [
            annualized_return(portfolio_returns),
            annualized_volatility(portfolio_returns),
            sharpe_ratio(portfolio_returns),
        ],
        "NIFTY 50": [
            annualized_return(nifty_returns),
            annualized_volatility(nifty_returns),
            sharpe_ratio(nifty_returns),
        ]
    }
    perf_df = pd.DataFrame(perf_metrics)

    # Fix: convert columns to numeric before formatting
    perf_df["Portfolio"] = pd.to_numeric(perf_df["Portfolio"], errors='coerce')
    perf_df["NIFTY 50"] = pd.to_numeric(perf_df["NIFTY 50"], errors='coerce')

    st.write("### Performance Metrics Comparison")
    st.dataframe(perf_df.style.format({
        "Portfolio": "{:.2%}",
        "NIFTY 50": "{:.2%}"
    }))

    # Concentration risk check
    st.write("### Concentration Risk Analysis")
    high_concentration = df[df['Weight'] > 0.10]
    if not high_concentration.empty:
        st.warning("High concentration detected in the following holdings (weight > 10%):")
        st.dataframe(high_concentration[['Ticker', 'Weight']].style.format({'Weight': '{:.2%}'}))
    else:
        st.success("Portfolio is well diversified.")

    # Recommendations
    st.write("### Recommendations")
    recs = []
    for _, row in df.iterrows():
        if row['Weight'] > 0.15:
            recs.append(f"Consider reducing position in {row['Ticker']} (weight {row['Weight']:.2%})")
        elif row['Weight'] < 0.01:
            recs.append(f"Position in {row['Ticker']} is very small (weight {row['Weight']:.2%}), consider if necessary")

    if recs:
        for r in recs:
            st.write("- " + r)
    else:
        st.write("Your portfolio weights look well balanced.")

if __name__ == "__main__":
    portfolio_analyzer()
