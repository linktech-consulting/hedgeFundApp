# importing all the libraries that we require for Dashboard Visualisation

from logging import error
from newsapi import NewsApiClient
import streamlit as st
import pandas as pd
import numpy as np
from GoogleNews import GoogleNews
import sys
import yfinance as yf
import pandas_datareader.data as web
from pandas_datareader import wb
import datetime
import streamlit.components.v1 as components
from pivottablejs import pivot_ui
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure, show
import requests
from bs4 import BeautifulSoup
import advertools as adv
import os
from getuseragent import UserAgent
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from io import StringIO
import requests



path = os.path.dirname(__file__)







newsapi = NewsApiClient(api_key='b6c59bf93ef84bc28fcebe34b66ee639')
googlenews = GoogleNews()



def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    useragent = UserAgent()

    theuseragent = useragent.Random()


    headers = {
    "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "DNT": "1",
    "Origin": "https://www.premierleague.com",
    "Referer": "https://www.premierleague.com/players",
    "User-Agent": theuseragent,
    }
    queryParams = {
    "pageSize": 32,
    "compSeasons": 274,
    "altIds": True,
    "page": 0,
    "type": "player",
    "id": -1,
    "compSeasonId": 274
}


    st.sidebar.title('LinkTech Fund Management Analytics Portal')

    # News Dashboard Selection
    if st.sidebar.checkbox('News Search Dashboard'):

        value = st.selectbox(
            'Please Select the Search preference',
            ('Trends', 'Google Search','Newsapi'))

        if(value == 'Newsapi'):
            input = st.text_input(
                'Enter What you have to Search for', 'stock in news')
            try:
                top_headlines = newsapi.get_everything(q=str(input))
            except Exception as e:
                st.error(f"News API request failed: {e}")
                top_headlines = {"articles": []}

            for headline in top_headlines.get('articles', []):
                st.write("Title : " + headline.get('title', ''))
                st.write("Description: " + headline.get('url', ''))
        elif(value == 'Google Search'):
            input = st.text_input(
            'Enter What you have to Search for', 'stock in news')
            googlenews.search(str(input))
            for results in googlenews.results():
                st.write("Title : " + results['title'])
                st.write("Description: " + results['desc'])
                st.write("Link: " + results['link'])

        elif(value=='Trends'):
            df=[]
            links=['https://www.moneycontrol.com/news/business/economy/',
            'https://economictimes.indiatimes.com/news/economy',
            'https://economictimes.indiatimes.com/industry',
            'https://www.business-standard.com/',
            'https://www.livemint.com/',
            'https://www.ndtv.com/business',
            'https://www.bloomberg.com/markets/economics',
            'https://www.bloomberg.com/asia',
            'https://www.financialexpress.com/',
            'https://www.cnbc.com/world/?region=world',
            'https://www.cnbc.com/economy/',
            'https://www.reuters.com/news/archive/GCA-Commodities',
            'https://www.marketwatch.com/',
            'https://seekingalpha.com/market-news',
            'https://www.morningstar.com/stocks',

            ]

            for link in links:
                page = requests.get(url = link, headers = headers, params = queryParams)
                contents = page.content
                soup = BeautifulSoup(contents, 'html.parser')
                datas=soup.find_all('p')


                for data in soup.find_all("a"):
                    para=data.get_text()
                    if (len(para)>50):
                        df.append((str(data.get_text(strip=True)),link))
                dFinal=pd.DataFrame(df, columns=['News','Link'])

            if st.checkbox("See News Headlines"):

                AgGrid(dFinal,
                        data_return_mode='AS_INPUT',
                        update_mode='MODEL_CHANGED',
                        fit_columns_on_grid_load=False,
                        enable_enterprise_modules=True,
                        height=350,
                        width='100%',
                        reload_data=True
                        )

            value= adv.word_frequency(dFinal["News"],phrase_len=2, rm_words=adv.stopwords.keys())
            st.dataframe(value)
            if st.checkbox('Search Phrases in Headlines'):
                search=st.text_input('Enter the Phrase You Want to Search', 'Inflation')
                df=dFinal[dFinal['News'].str.contains(search,case=False)]
                st.dataframe(df['News'])


    if st.sidebar.checkbox('Stocks Data Analysis'):
        urlstock='https://www1.nseindia.com/content/equities/EQUITY_L.csv'
        
        req = requests.get(urlstock, headers=headers)
        data = StringIO(req.text)
        
        
       
        
        symbols=pd.read_csv(data )
        stocklist_IN=symbols


        st.write("Dashboard For Stock Analysis Using Python and Machine Learning")
        if st.checkbox("Search Value Stock"):
            st.write("Value Stocks")
            data = pd.read_csv(path+"/Recomm.csv")
            value = data[(data['EPS'] > 0) & (
            data['Value Indicator'] == 1) & (data['Sales_Growth'] > 0)
                        & (data['Operating_Profit_Growth'] > 0) & (data['PE Ratio'] > 0) &
                        (data['Timing']=="STRONG BUY TIME") ]

            st.dataframe(value)

        elif st.checkbox("Search Growth Stock"):
            st.write("Growth Stocks")
            data = pd.read_csv("Recomm.csv",storage_options = {'User-Agent': theuseragent})
            growth = data[(data['EPS'] > 0) & (
            data['Growth Indicator'] == 1) & (data['Sales_Growth'] > 0)
                    & (data['Operating_Profit_Growth'] > 0) & (data['PE Ratio'] > 0) &
                    (data['Timing']=="STRONG BUY TIME")]

            st.dataframe(growth)
        if st.checkbox("Research"):
            val = st.selectbox('Enter the Yahoo Finance Symbol', (stocklist_IN['NAME OF COMPANY']))
            symbol=stocklist_IN[val==stocklist_IN['NAME OF COMPANY']]['SYMBOL']

            nse_listed=list(symbol)[0]+'.NS'
            stock = yf.Ticker(nse_listed)
            summary = {'Company Name':stock.info["longName"],
                    'Sector': stock.info["sector"],
                    'Gross Margins':stock.info["grossMargins"]*100,
                     'Profit Margins':stock.info["profitMargins"]*100,
                     'Operating Margins':stock.info["operatingMargins"]*100,
                     'Return on Assets':stock.info["returnOnAssets"],
                     'Revenur per Share':stock.info["revenuePerShare"],
                     'Peg Ratio':stock.info["pegRatio"]
                }
            col1, col2 = st.columns(2)
            col2.markdown('### %s' %summary['Company Name'])
            col2.markdown('Company Summary: %s '%stock.info["longBusinessSummary"])
            col2.markdown('Book Value : %s' %stock.info["bookValue"])
            col2.markdown('Revenue Per Share : %s' %summary['Revenur per Share'])
            col2.markdown('Sector: %s' %summary['Sector'])
            hist = stock.history(period="1Y")
            col2.markdown('Current Price: %s' %hist['Close'].iloc[-1])

            fig, ax = plt.subplots()
            value = [summary['Gross Margins'],summary['Operating Margins'],summary['Profit Margins']]
            bars = ('Gross Margins', 'Operating Margins', 'Profit Margins')
            x_pos = np.arange(len(bars))


            barvalue=ax.bar(x_pos, value, color=['black', 'red', 'green'])
            plt.xticks(x_pos, bars)
            plt.figure(figsize=(2, 2))
            ax.set_xlabel(summary['Company Name'])
            for barvalue in ax.containers:
                ax.bar_label(barvalue)
                col1.pyplot(fig)

            df = stock.balance_sheet
            L=['Total Assets', 'Total Liab']
            data_1=df.loc[L]
            df_new=data_1.T
            df_new['Equity']=df_new['Total Assets']-df_new['Total Liab']
            fig_1 = px.bar(df_new,barmode='group')
            st.markdown('#### Balance Sheet Equity Growth Analysis')
            st.plotly_chart(fig_1,use_container_width=True)
            st.markdown("1. Remember the growth in equity is important for company to have profitability prospects")


            df2=stock.financials
            L2=['Net Income Applicable To Common Shares','Operating Income']
            data_2=df2.loc[L2]/float(stock.info["sharesOutstanding"])
            data_2_new=data_2.astype(float, errors = 'raise')
            st.markdown('#### Net Income and Operating Income Per Share')
            fig_2 = px.bar(data_2_new.T,barmode='group')
            st.plotly_chart(fig_2,use_container_width=True)
            st.markdown('1. The growth in income and smaller the difference between operating income and net income indicates less debt and strong growth prospects')



            st.markdown('### Company Financial Soundness')

            value =df2.T['Operating Income']/df_new['Total Assets']
            value=pd.to_numeric(value.rename("Operating Income wrt Total Assets"))
            st.markdown('1.The more the negative value the more the company will loose its assets and incure more liabilities ')
            st.markdown('2.The growth in this factor indicates the growth in efficiency of the companyr. ')
            fig_3=px.bar(value, barmode='group')
            st.plotly_chart(fig_3,use_container_width=True)

            col1.markdown('Time To Buy A Stock')
            stats=hist.describe()
            col1.markdown('#### Current Price: %s' %hist['Close'].iloc[-1])
            col1.write(stats['Close'])
            SD_1_0=stats['Close'][1]-stats['Close'][2] #68.2% times stock will be in this range
            SD_1_1=stats['Close'][1]+stats['Close'][2] #68.2% times stock will be in this range
            SD_2_0=stats['Close'][1]-2*stats['Close'][2] #95.4 % times stock will be in this range
            SD_2_1=stats['Close'][1]+2*stats['Close'][2] #95.4 % times stock will be in this range


            col1.markdown('###### Oversold Zones')
            col1.markdown('Oversold Price Resistance 1= %s' %SD_1_0)
            col1.markdown('Oversold Price Resistance 2= %s' %SD_2_0)

            col1.markdown('###### Overbought Zones')
            col1.markdown('Overbought Price Resistance 1= %s' %SD_1_1)
            col1.markdown('Overbought Price Resistance 1= %s' %SD_2_1)

            if col1.checkbox('Recent News'):
                input = col1.text_input('Enter the company you want to search for', summary['Company Name'])
                googlenews.search(str(input))
                for results in googlenews.results():
                    col1.markdown("### Title : " + results['title'])
                    col1.markdown("Description: " + results['desc'])
                    col1.markdown("Link: " + results['link'])
            if col2.checkbox('Indian Stock Research Useful Sites'):
                col2.markdown('[Moneycontrol Research](https://www.moneycontrol.com/)- Enter stock name and go to research section for analyst recommendation')
                col2.markdown('[Ticker Tape](https://www.tickertape.in/)- Enter stock name for Fundamental Research')
                col2.markdown('[Trendlyne](https://trendlyne.com/features/)- Good website for stock summary and overview')
                col2.markdown('[Trading View](https://in.tradingview.com/)- Good website for Technical Trends and Analyst Trends')

            try:


                if(stock.stock.institutional_holder.empty !=True):
                    st.markdown('Major Institution Holders')
                    fig_2 = px.bar(stock.institutional_holders.T,barmode='group',y=stock.institutional_holders['Holder'],
                    x=stock.institutional_holders['Value'])

                    st.plotly_chart(fig_2)
            except:
                st.write('No Institutional Holding Data Available')










            if st.checkbox("Stock Info"):
                st.write(stock.info)
            elif st.checkbox("Price Charts"):
                st.write("Charts Diplay Section")
                i = st.sidebar.selectbox(
                    "Interval in minutes", ("1d", "1m", "5m", "15m", "30m"))
                p = st.sidebar.number_input(
                    "How many days (1-365)", min_value=1, max_value=365, step=1, value=365)
                data = stock.history(interval=i, period=str(p) + "d")
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'])])

                st.plotly_chart(fig, use_container_width=True)
                if st.checkbox('Display Data in DataFrame'):
                    st.dataframe(data)

            elif st.checkbox("Balance Sheet"):
                df = stock.balance_sheet
                st.dataframe(df)
                st.write(df[df.index == 'Total Assets'])

            elif st.checkbox(" Cash Flow"):
                st.dataframe(stock.cashflow)
            elif st.checkbox("Financials"):
                st.dataframe(stock.financials)
            elif st.checkbox("Major Institutional Holders"):
                st.write(stock.institutional_holders)
            elif st.checkbox("Analyst Recommendation"):
                st.write(stock.recommendations)
            elif st.checkbox("Earnings Report"):
                fig = px.bar(stock.earnings, barmode='group')
                st.plotly_chart(fig, use_container_width=True)




    if st.sidebar.checkbox("Fed And World Bank Data Analysis"):
            if st.checkbox('FRED Data'):
                start = st.date_input("Enter From Date", datetime.date(2009, 7, 6))
                end = st.date_input("Enter To Date", datetime.date(2022, 3, 3))
                symbol =  st.multiselect(
                "Please Enter the Symbol of the Data you want form FRED Site.", ['GDP','DFF','CPILFESL','PWHEAMTUSDM'
                ,'PSUNOUSDM'],['DFF','GDP'])
                data_old=pd.DataFrame()
                for value in symbol:
                    data_new = web.DataReader(str(value), 'fred', start, end)
                    fig = px.line(data_new)
                    st.plotly_chart(fig)



            elif st.checkbox('World Bank Indicator'):
                search_input = st.text_input(
                'Enter What you Want to search on the world bank site', 'gdp.*capita.*const')
                matches = wb.search(str(search_input))
                st.write(matches)
                st.text('Download World Bank Data')
                star = st.date_input("Enter From Date", datetime.date(2009, 7, 6))
                en = st.date_input("Enter To pDate", datetime.date(2022, 3, 3))
                count = st.text_input('Enter the Country Symbol', 'US')
                symbol = st.text_input('Enter the indicator symbol', 'NY.GDP.PCAP.KD')

                data = wb.download(indicator=str(symbol),
                               country=str(count), start=star, end=en)
                st.write(data)
                fig = px.line(data, x=data.index.get_level_values(1), y=data[symbol])
                st.plotly_chart(fig)


    if st.sidebar.checkbox('ETF Strategy Builder Portal'):
        st.markdown('# Stay Tuned We are Building Something Interesting')
