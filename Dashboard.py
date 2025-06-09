# app.py
import streamlit as st
from utils.auth import check_password
from utils.news_utils import news_dashboard
from utils.stock_utils import stock_analysis_dashboard
from utils.econ_utils import economic_data_dashboard
from utils.trader import trader_dashboard
from utils.etf_builder import etf_builder_dashboard
from utils.portfolio_analyzer import portfolio_analyzer



def main():
    st.set_page_config(layout="wide", page_title="LinkTech Financial Dashboard")

    if check_password():
        st.sidebar.title('LinkTech Fund Management Analytics Portal')

        if st.sidebar.checkbox('News Search Dashboard'):
            news_dashboard()

        if st.sidebar.checkbox('Stocks Data Analysis'):
            stock_analysis_dashboard()

        if st.sidebar.checkbox("Fed And World Bank Data Analysis"):
            economic_data_dashboard()

        if st.sidebar.checkbox('Trader Dashboard'):
            trader_dashboard()
        if st.sidebar.checkbox('ETF Builder'):
            etf_builder_dashboard()
        if st.sidebar.checkbox("Portfolio Analyzer"):
            portfolio_analyzer()

if __name__ == "__main__":
    main()
