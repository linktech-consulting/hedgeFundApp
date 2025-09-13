# utils/stock_utils.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from getuseragent import UserAgent
from io import StringIO
import requests

@st.cache_resource
def get_useragent():
    return UserAgent().Random()

@st.cache_data(ttl=86400)
def load_equity_symbols():
    headers = {"User-Agent": get_useragent()}
    urlstock = 'https://www1.nseindia.com/content/equities/EQUITY_L.csv'
    req = requests.get(urlstock, headers=headers)
    data = StringIO(req.text)
    return pd.read_csv(data, encoding='ISO-8859-1', on_bad_lines='skip')

@st.cache_data
def load_recommendations(path='Recomm.csv'):
    return pd.read_csv(path)

def stock_analysis_dashboard():
    st.subheader("Stock Analysis Dashboard")
    symbols = pd.read_csv('EQUITY_L.csv')
    user_agent = get_useragent()

    if st.checkbox("Search Value Stock (Buffett Style)"):
        data = load_recommendations()

        st.markdown("### Warren Buffett Filters")
        min_eps = st.slider("Minimum EPS", 1.0, 100.0, 5.0)
        min_roe = st.slider("Minimum Return on Equity (ROE)", 0.0, 100.0, 15.0)
        max_pe = st.slider("Maximum PE Ratio", 0.0, 100.0, 20.0)
        min_margin = st.slider("Minimum Operating Profit Margin", 0.0, 100.0, 15.0)

        value = data[
            (data['EPS'] >= min_eps) &
            (data['Value Indicator'] == 1)  &
            (data['Timing'] == "STRONG BUY TIME")
            
        ]

        st.dataframe(value)

    if st.checkbox("Search Growth Stock (Peter Lynch Style)"):
        data = load_recommendations()

        st.markdown("### Peter Lynch Filters")
        min_eps = st.slider("Minimum EPS Growth", 0.0, 100.0, 10.0)
        min_sales_growth = st.slider("Minimum Sales Growth", 0.0, 100.0, 10.0)
        min_growth_ratio = st.slider("Minimum PEG Ratio", 0.0, 3.0, 1.0)

        growth = data[
            (data['EPS'] > 0) &
            (data['Growth Indicator'] == 1) &
            (data['Timing'] == "STRONG BUY TIME")
        ]

        st.dataframe(growth)

    # Remainder of stock research stays the same
    if st.checkbox("Research a Stock"):
        company = st.selectbox('Select Company', symbols['NAME OF COMPANY'])
        symbol = symbols.loc[symbols['NAME OF COMPANY'] == company, 'SYMBOL'].values[0] + '.NS'
        stock = yf.Ticker(symbol)
        

        info = stock.info
        hist = stock.history(period="1Y")

        st.markdown(f"### {info.get('longName', 'N/A')}")
        st.write("Sector:", info.get('sector', 'N/A'))
        st.write("Book Value:", info.get('bookValue'))
        st.write("Revenue Per Share:", info.get('revenuePerShare'))
        st.write("Current Price:", hist['Close'].iloc[-1])

        col1, col2 = st.columns(2)
        fig, ax = plt.subplots()
        margins = [info.get('grossMargins', 0), info.get('operatingMargins', 0), info.get('profitMargins', 0)]
        bars = ['Gross', 'Operating', 'Profit']
        ax.bar(bars, [m*100 for m in margins], color=['black', 'red', 'green'])
        ax.set_ylabel('%')
        col1.pyplot(fig)

        # Balance Sheet
        balance = stock.balance_sheet.T
        if 'Total Assets' in balance.columns and 'Total Liab' in balance.columns:
            balance['Equity'] = balance['Total Assets'] - balance['Total Liab']
            fig1 = px.bar(balance[['Total Assets', 'Total Liab', 'Equity']])
            st.markdown("#### Balance Sheet Equity Growth")
            st.plotly_chart(fig1, use_container_width=True)

        # Income Statements
        income = stock.financials.T
        cols = ['Operating Income', 'Net Income Applicable To Common Shares']
        if all(col in income.columns for col in cols):
            income_ps = income[cols] / float(info.get("sharesOutstanding", 1))
            fig2 = px.bar(income_ps)
            st.markdown("#### Income Per Share")
            st.plotly_chart(fig2, use_container_width=True)

        # Price Distribution Stats
        stats = hist['Close'].describe()
        col2.markdown("### Price Stats")
        col2.write(stats)
        st.markdown("#### Overbought/Oversold Zones")
        sd = stats['std']
        mean = stats['mean']
        st.write("Oversold:", mean - sd, mean - 2*sd)
        st.write("Overbought:", mean + sd, mean + 2*sd)

        if st.checkbox("View Price Chart"):
            interval = st.selectbox("Interval", ["1d", "1m", "5m", "15m", "30m"])
            period = st.number_input("Period (days)", 1, 365, 365)
            data = stock.history(interval=interval, period=f"{period}d")
            fig3 = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                                  low=data['Low'], close=data['Close'])])
            st.plotly_chart(fig3, use_container_width=True)

        if st.checkbox("Show Raw Financials"):
            st.subheader("Balance Sheet")
            st.dataframe(stock.balance_sheet)
            st.subheader("Cash Flow")
            st.dataframe(stock.cashflow)
            st.subheader("Income Statement")
            st.dataframe(stock.financials)

        if st.checkbox("Institutional Holdings"):
            if hasattr(stock, 'institutional_holders') and not stock.institutional_holders.empty:
                fig = px.bar(stock.institutional_holders, x='Holder', y='Value')
                st.plotly_chart(fig)
            else:
                st.write("No data available.")

        if st.checkbox("Earnings Report"):
            fig = px.bar(stock.earnings, barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Full Info JSON"):
            st.json(info)
