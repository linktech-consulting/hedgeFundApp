# utils/econ_utils.py
import streamlit as st
import datetime
import plotly.express as px
from pandas_datareader import data as web
from pandas_datareader import wb
import pandas as pd

@st.cache_data(ttl=86400)
def get_fred_data(symbols, start, end):
    data_frames = []
    for symbol in symbols:
        df = web.DataReader(symbol, 'fred', start, end)
        df.columns = [symbol]
        data_frames.append(df)
    return data_frames

@st.cache_data(ttl=86400)
def search_world_bank_indicators(query):
    return wb.search(query)

@st.cache_data(ttl=86400)
def get_world_bank_data(country, indicator, start, end):
    return wb.download(indicator=indicator, country=country, start=start, end=end)

def economic_data_dashboard():
    st.subheader("Global Economic Indicators")

    if st.checkbox('FRED Data'):
        start_date = st.date_input("Start Date", datetime.date(2009, 7, 6))
        end_date = st.date_input("End Date", datetime.date(2022, 3, 3))

        fred_symbols = st.multiselect(
            "Choose FRED Symbols",
            ['GDP', 'DFF', 'CPILFESL', 'PWHEAMTUSDM', 'PSUNOUSDM'],
            default=['DFF', 'GDP']
        )

        data_frames = get_fred_data(fred_symbols, start_date, end_date)
        for df in data_frames:
            fig = px.line(df, x=df.index, y=df.columns[0], title=df.columns[0])
            st.plotly_chart(fig)

    if st.checkbox('World Bank Indicator'):
        search_term = st.text_input("Search World Bank Indicators", 'gdp.*capita.*const')
        matches = search_world_bank_indicators(search_term)
        st.write(matches)

        start_date = st.date_input("WB Start Date", datetime.date(2009, 7, 6))
        end_date = st.date_input("WB End Date", datetime.date(2022, 3, 3))
        country = st.text_input("Country Code (e.g., US)", 'US')
        indicator_code = st.text_input("Indicator Code (e.g., NY.GDP.PCAP.KD)", 'NY.GDP.PCAP.KD')

        if st.button("Load World Bank Data"):
            data = get_world_bank_data(country, indicator_code, start_date, end_date)
            st.write(data)
            fig = px.line(data, x=data.index.get_level_values(1), y=indicator_code,
                          title=f"{indicator_code} for {country}")
            st.plotly_chart(fig)
