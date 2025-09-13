# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:42:54 2022

@author: ameys
"""

import pandas as pd
import yfinance as yf
import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:42:54 2022

@author: ameys
"""

import pandas as pd
import yfinance as yf
import numpy as np
import requests
import time

nseEquityPath = "EQUITY_L.csv"
smeEquityPath = 'SME_EQUITY_L.csv'

nseEquity = pd.read_csv(nseEquityPath)
smeEquity = pd.read_csv(smeEquityPath)

# Making List of Stocks File to Collect Data From 
stock_list_nse = [x + ".NS" for x in nseEquity['SYMBOL']]

# Industry parameter Recomendation
industry_data = pd.read_csv("Recomm.csv")

# Filter numeric columns only for groupby aggregation
numeric_cols = ['PE Ratio', 'EPS', 'Sales_Growth', 'Operating_Profit_Growth']
numeric_cols = [col for col in numeric_cols if col in industry_data.columns]
industry_avg = industry_data.groupby("Industry")[numeric_cols].mean()

# Data Collection
final_df = pd.DataFrame()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

for st in stock_list_nse:
    try:
        stock = yf.Ticker(st)
        info = stock.info
        df = stock.financials

        # Retry alternative source if yfinance fails for info
        if not info or 'industry' not in info:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{st}?modules=defaultKeyStatistics,financialData,summaryProfile"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                json_data = response.json()
                info = json_data.get('quoteSummary', {}).get('result', [{}])[0].get('summaryProfile', {})

        if not info or 'industry' not in info:
            raise ValueError("Missing industry data even after retry")

        # Check required rows exist
        if 'Net Income' not in df.index or 'Operating Income' not in df.index:
            raise ValueError("Missing Net Income or Operating Income")

        # Sales Growth Average
        NI_0 = float(df.loc['Net Income'].iloc[3])
        NI_1 = float(df.loc['Net Income'].iloc[0])
        sales_growth = (((NI_1 / NI_0) - 1) / 4) * 100

        # Operating Income Growth Rate
        OI_0 = float(df.loc['Operating Income'].iloc[0])
        OI_1 = float(df.loc['Operating Income'].iloc[1])
        OI_2 = float(df.loc['Operating Income'].iloc[2])
        OI_3 = float(df.loc['Operating Income'].iloc[3])
        operating_growth = (((OI_0 / OI_3) - 1) / 4) * 100

        # EPS and PE
        net_income = info.get('netIncomeToCommon')
        shares = info.get('sharesOutstanding')
        price = info.get('currentPrice')

        if net_income and shares and price:
            eps = net_income / shares
            if info.get('forwardEps') and info['forwardEps'] < price / eps:
                pe = info['forwardEps']
            else:
                pe = price / eps
        else:
            eps = 0
            pe = 0

        # Beta
        beta = info.get('beta', 0)

        # Value Indicator: PE lower than industry avg
        industry_pe = industry_avg['PE Ratio'].get(info['industry'], 0)
        value_ind = 1 if pe < industry_pe else 0

        # Growth Indicator: Rising operating income & beta > 1
        growth_ind = int(OI_0 > OI_1 > OI_2 > OI_3 and beta > 1)

        # Buffett Filter
        passes_buffett = int(
            pe < 20 and
            info.get('priceToBook', 99) < 3 and
            info.get('returnOnEquity', 0) > 0.15 and
            info.get('debtToEquity', 999) < 50 and
            info.get('operatingMargins', 0) > 0.10 and
            info.get('bookValue', 0) > 0
        )

        # Lynch Filter
        passes_lynch = int(
            info.get('earningsGrowth', 0) > 0.15 and
            info.get('revenueGrowth', 0) > 0.1 and
            info.get('pegRatio', 99) < 1.5 and
            info.get('returnOnAssets', 0) > 0.08 and
            info.get('priceToSalesTrailing12Months', 99) < 2 and
            info.get('grossMargins', 0) > 0.4
        )

        # Append results
        row = {
            'Stock': st,
            'Sales_Growth': sales_growth,
            'Operating_Profit_Growth': operating_growth,
            'Growth Indicator': growth_ind,
            'Sector': info.get('sector'),
            'Industry': info.get('industry'),
            'PB Ratio': info.get('priceToBook'),
            'Industry PE': industry_pe,
            'Value Indicator': value_ind,
            'EPS': eps,
            'PE Ratio': pe,
            'D/E Ratio': info.get('debtToEquity'),
            'Beta': beta,
            'Buffett Filter': passes_buffett,
            'Lynch Filter': passes_lynch
        }

        row_df = pd.DataFrame([row])
        final_df = pd.concat([final_df, row_df], ignore_index=True)

        # Save after each stock to avoid data loss
        final_df.to_csv("Recomm.csv", index=False)
        time.sleep(1)  # avoid rate limit

    except Exception as e:
        print(f"Error processing {st}: {e}")

nseEquityPath = "EQUITY_L.csv"
smeEquityPath = 'SME_EQUITY_L.csv'

nseEquity = pd.read_csv(nseEquityPath)
smeEquity = pd.read_csv(smeEquityPath)

# Making List of Stocks File to Collect Data From 
stock_list_nse = [x + ".NS" for x in nseEquity['SYMBOL']]

# Industry parameter Recomendation
industry_data = pd.read_csv("Recomm.csv")

# Filter numeric columns only for groupby aggregation
numeric_cols = ['PE Ratio', 'EPS', 'Sales_Growth', 'Operating_Profit_Growth']
numeric_cols = [col for col in numeric_cols if col in industry_data.columns]
industry_avg = industry_data.groupby("Industry")[numeric_cols].mean()

# Data Collection
final_df = pd.DataFrame()

for st in stock_list_nse:
    try:
        stock = yf.Ticker(st)
        info = stock.info
        df = stock.financials

        # Check required rows exist
        if 'Net Income' not in df.index or 'Operating Income' not in df.index:
            raise ValueError("Missing Net Income or Operating Income")

        # Sales Growth Average
        NI_0 = float(df.loc['Net Income'].iloc[3])
        NI_1 = float(df.loc['Net Income'].iloc[0])
        sales_growth = (((NI_1 / NI_0) - 1) / 4) * 100

        # Operating Income Growth Rate
        OI_0 = float(df.loc['Operating Income'].iloc[0])
        OI_1 = float(df.loc['Operating Income'].iloc[1])
        OI_2 = float(df.loc['Operating Income'].iloc[2])
        OI_3 = float(df.loc['Operating Income'].iloc[3])
        operating_growth = (((OI_0 / OI_3) - 1) / 4) * 100

        # EPS and PE
        net_income = info.get('netIncomeToCommon')
        shares = info.get('sharesOutstanding')
        price = info.get('currentPrice')

        if net_income and shares and price:
            eps = net_income / shares
            if info.get('forwardEps') and info['forwardEps'] < price / eps:
                pe = info['forwardEps']
            else:
                pe = price / eps
        else:
            eps = 0
            pe = 0

        # Beta
        beta = info.get('beta', 0)

        # Value Indicator: PE lower than industry avg
        industry_pe = industry_avg['PE Ratio'].get(info['industry'], 0)
        value_ind = 1 if pe < industry_pe else 0

        # Growth Indicator: Rising operating income & beta > 1
        growth_ind = int(OI_0 > OI_1 > OI_2 > OI_3 and beta > 1)

        # Buffett Filter
        passes_buffett = int(
            pe < 20 and
            info.get('priceToBook', 99) < 3 and
            info.get('returnOnEquity', 0) > 0.15 and
            info.get('debtToEquity', 999) < 50 and
            info.get('operatingMargins', 0) > 0.10 and
            info.get('bookValue', 0) > 0
        )

        # Lynch Filter
        passes_lynch = int(
            info.get('earningsGrowth', 0) > 0.15 and
            info.get('revenueGrowth', 0) > 0.1 and
            info.get('pegRatio', 99) < 1.5 and
            info.get('returnOnAssets', 0) > 0.08 and
            info.get('priceToSalesTrailing12Months', 99) < 2 and
            info.get('grossMargins', 0) > 0.4
        )

        # Append results
        row = {
            'Stock': st,
            'Sales_Growth': sales_growth,
            'Operating_Profit_Growth': operating_growth,
            'Growth Indicator': growth_ind,
            'Sector': info.get('sector'),
            'Industry': info.get('industry'),
            'PB Ratio': info.get('priceToBook'),
            'Industry PE': industry_pe,
            'Value Indicator': value_ind,
            'EPS': eps,
            'PE Ratio': pe,
            'D/E Ratio': info.get('debtToEquity'),
            'Beta': beta,
            'Buffett Filter': passes_buffett,
            'Lynch Filter': passes_lynch
        }

        row_df = pd.DataFrame([row])
        final_df = pd.concat([final_df, row_df], ignore_index=True)

        # Save after each stock to avoid data loss
        final_df.to_csv("Recomm.csv", index=False)

    except Exception as e:
        print(f"Error processing {st}: {e}")
