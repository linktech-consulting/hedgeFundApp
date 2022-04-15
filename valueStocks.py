# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:42:54 2022

@author: ameys
"""

import pandas as pd
import yfinance as yf
import numpy as np



nseEquityPath ="EQUITY_L.csv"
smeEquityPath ='SME_EQUITY_L.csv'

nseEquity=pd.read_csv(nseEquityPath)
smeEquity=pd.read_csv(smeEquityPath)




# Making List of Stocks File to Collect Data From 
stock_list_nse=[]

for  x in nseEquity['SYMBOL']:
  stock_list_nse.append(x+".NS")
  


#Industry parameter Recomendation

data=pd.read_csv("Recomm.csv")
dfS=data.groupby("Industry").mean()


# Data Collection


data_F=pd.DataFrame()
for st in stock_list_nse:
  try: 
    stock = yf.Ticker(str(st))
    df=stock.financials
    # Sales Growth Average
    I_0=float(df[df.index== 'Net Income'].iloc[:,3])
    I_1=float(df[df.index== 'Net Income'].iloc[:,0])
    growth_Avg = (((I_1/I_0)-1)/4)*100 # 4yr average growth rate
    #Operating Income Growth Rate
    
    I_3=float(df[df.index== 'Operating Income'].iloc[:,3])
    I_0=float(df[df.index== 'Operating Income'].iloc[:,0])
    I_1=float(df[df.index== 'Operating Income'].iloc[:,1])
    I_2=float(df[df.index== 'Operating Income'].iloc[:,2])
    operatingGrowth_Avg = (((I_0/I_3)-1)/4)*100
    # Market Beta Value
    beta=stock.info['beta']
    if (stock.info['netIncomeToCommon']!= False and stock.info['sharesOutstanding']!= False and stock.info['currentPrice']!= False):
      net_income=stock.info['netIncomeToCommon']
      total_shares=stock.info['sharesOutstanding']
      eps=net_income/total_shares
      price=stock.info['currentPrice']
      if (type(stock.info['forwardEps'])!=type(None) and stock.info['forwardEps']<float(price/eps)):
        pe=stock.info['forwardEps']
      else:
        pe=float(price/eps)
    else:
      pe=0
      eps=0
    
    if (dfS[dfS.index.isin([stock.info['industry']])].empty == False):
      I_pe= dfS['PE Ratio'].iloc[dfS.index == stock.info['industry']][0]
      if (pe<I_pe):
        V_ind=1
      else:
        V_ind=0
    else:
      I_pe=0
      V_ind=0

  
    
    
    data=[{
      'Stock': st,
      'Sales_Growth':growth_Avg,
      'Operating_Profit_Growth':operatingGrowth_Avg,
      'Growth Indicator': np.where(I_0> I_1 and I_1>I_2 and I_2> I_3 and beta>1 , 1,0),
      'Sector':stock.info['sector'],
      'Industry':stock.info['industry'],
      'PB Ratio':stock.info['priceToBook'],
      'Industry PE': I_pe,
      'Value Indicator':V_ind,
      'EPS':eps,
      'PE Ratio':pe ,
      'D/E Ratio':stock.info['debtToEquity'],
      'Beta':beta
      
    }]
    data1=pd.DataFrame(data, columns = ['Stock', 'Sales_Growth','Operating_Profit_Growth','Sector','PB Ratio','EPS','PE Ratio','D/E Ratio','Growth Indicator','Beta',
                                        'Industry','Industry PE','Value Indicator'])
    data_F=data_F.append(data1)
    data_F.to_csv("Recomm.csv")
  except Exception as e:

    print(e)
    print(st)



