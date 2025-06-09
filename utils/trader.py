import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time

# --- Candlestick Pattern Functions ---
def is_hammer(row):
    body = abs(row['close'] - row['open'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    return lower_shadow > 2 * body and upper_shadow < body

def is_bullish_engulfing(prev, curr):
    return prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']

def is_bearish_engulfing(prev, curr):
    return prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']

def is_morning_star(data, i):
    if i < 2: return False
    return data.iloc[i-2]['close'] < data.iloc[i-2]['open'] and abs(data.iloc[i-1]['close'] - data.iloc[i-1]['open']) < 0.1 and data.iloc[i]['close'] > data.iloc[i]['open'] and data.iloc[i]['close'] > data.iloc[i-2]['open']

def is_evening_star(data, i):
    if i < 2: return False
    return data.iloc[i-2]['close'] > data.iloc[i-2]['open'] and abs(data.iloc[i-1]['close'] - data.iloc[i-1]['open']) < 0.1 and data.iloc[i]['close'] < data.iloc[i]['open'] and data.iloc[i]['close'] < data.iloc[i-2]['open']

def is_piercing_line(prev, curr):
    return prev['close'] < prev['open'] and curr['open'] < prev['low'] and curr['close'] > (prev['open'] + prev['close']) / 2

def is_dark_cloud_cover(prev, curr):
    return prev['close'] > prev['open'] and curr['open'] > prev['high'] and curr['close'] < (prev['open'] + prev['close']) / 2

def is_shooting_star(row):
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    return upper_shadow > 2 * body and lower_shadow < body

def is_breakout_with_volume(data, i):
    if i < 2 or pd.isna(data['volume_sma'].iloc[i]):
        return False
    curr = data.iloc[i]
    prev = data.iloc[i - 1]
    return curr['close'] > prev['high'] and curr['volume'] > 1.5 * data['volume_sma'].iloc[i]

# --- Technical Indicator & Pattern Analysis ---
def tech_indicator(data, token_id, higher_tf_ema=None):
    data = data.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    data["ema20"] = ta.trend.EMAIndicator(close=data["close"], window=20).ema_indicator()
    data["ema50"] = ta.trend.EMAIndicator(close=data["close"], window=50).ema_indicator()
    data["rsi"] = ta.momentum.RSIIndicator(close=data["close"], window=14).rsi()
    data["atr"] = ta.volatility.AverageTrueRange(high=data["high"], low=data["low"], close=data["close"], window=14).average_true_range()
    
    try:
        data["mfi"] = ta.volume.MFIIndicator(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=14).money_flow_index()
    except Exception:
        data["mfi"] = pd.Series([None] * len(data))

    data["volume_sma"] = data["volume"].rolling(window=20).mean()

    pattern_flags = ["hammer", "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star", "piercing_line", "dark_cloud_cover", "shooting_star"]
    for pf in pattern_flags:
        data[pf] = False
    data["volume_breakout"] = False

    for i in range(1, len(data)):
        curr = data.iloc[i]
        prev = data.iloc[i - 1]
        if is_hammer(curr): data.at[data.index[i], "hammer"] = True
        if is_bullish_engulfing(prev, curr): data.at[data.index[i], "bullish_engulfing"] = True
        if is_bearish_engulfing(prev, curr): data.at[data.index[i], "bearish_engulfing"] = True
        if is_morning_star(data, i): data.at[data.index[i], "morning_star"] = True
        if is_evening_star(data, i): data.at[data.index[i], "evening_star"] = True
        if is_piercing_line(prev, curr): data.at[data.index[i], "piercing_line"] = True
        if is_dark_cloud_cover(prev, curr): data.at[data.index[i], "dark_cloud_cover"] = True
        if is_shooting_star(curr): data.at[data.index[i], "shooting_star"] = True
        if is_breakout_with_volume(data, i): data.at[data.index[i], "volume_breakout"] = True

    latest = data.iloc[-1]
    close_price = latest["close"]
    atr = latest["atr"]
    mfi = latest.get("mfi", None)
    rsi = latest["rsi"]
    higher_tf_ok = True if higher_tf_ema is None else close_price > higher_tf_ema

    if higher_tf_ok and latest["ema20"] > latest["ema50"] and rsi > 55 and mfi and mfi > 55:
        trend = "Bullish"
        stop_loss = round(close_price - atr, 2)
        target = round(close_price + 2 * (close_price - stop_loss), 2)
        color = "green"
    elif latest["ema20"] < latest["ema50"] and rsi < 45 and mfi and mfi < 45:
        trend = "Bearish"
        stop_loss = round(close_price + atr, 2)
        target = round(close_price - 2 * (stop_loss - close_price), 2)
        color = "red"
    else:
        trend = "Sideways"
        stop_loss = round(close_price - atr, 2)
        target = round(close_price + atr, 2)
        color = "gray"

    return [token_id, trend, close_price, target, stop_loss]

# --- Symbols ---
op_symbols = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS",
    "BAJAJHLDNG.NS", "BAJAJHFL.NS", "BANKBARODA.NS", "BPCL.NS", "BRITANNIA.NS",
    "BOSCHLTD.NS", "CANBK.NS", "CGPOWER.NS", "CHOLAFIN.NS", "DABUR.NS",
    "DIVISLAB.NS", "DLF.NS", "DMART.NS", "GAIL.NS", "GODREJCP.NS",
    "HAVELLS.NS", "HAL.NS", "ICICIGI.NS", "ICICIPRULI.NS",
    "INDHOTEL.NS", "IOC.NS", "INDIGO.NS", "NAUKRI.NS", "IRFC.NS",
    "JINDALSTEL.NS", "JSWENERGY.NS", "LICI.NS", "LODHA.NS", "LTIM.NS",
    "PIDILITIND.NS", "PFC.NS", "PNB.NS", "RECLTD.NS", "MOTHERSON.NS",
    "SHREECEM.NS", "SIEMENS.NS", "TATAPOWER.NS", "TORNTPHARM.NS",
    "TVSMOTOR.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ZYDUSLIFE.NS",
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BHARTIARTL.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "NTPC.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "WIPRO.NS", "TECHM.NS",
    "INDUSINDBK.NS", "ONGC.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "DRREDDY.NS", "TITAN.NS", "GRASIM.NS", "HCLTECH.NS",
    "SBILIFE.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "CIPLA.NS", "BAJAJ-AUTO.NS",
    "TATACONSUM.NS", "UPL.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "APOLLOHOSP.NS",
    "SUNPHARMA.NS", "HINDALCO.NS", "M&M.NS", "SBICARD.NS"
]

index_name_map = {
    "NSE:NIFTY 50": "^NSEI",
    "NSE:NIFTY BANK": "^NSEBANK",
    "NSE:NIFTY PHARMA": "^NSEPHARMA",
    "NSE:NIFTY NEXT 50": "^NIFTY"
}

# --- Helper to normalize DataFrame columns ---
def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Map variants to standard names
    col_map = {}
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close', 'volume']:
            col_map[col] = col
        elif col.startswith('open'):
            col_map[col] = 'open'
        elif col.startswith('high'):
            col_map[col] = 'high'
        elif col.startswith('low'):
            col_map[col] = 'low'
        elif col.startswith('close'):
            col_map[col] = 'close'
        elif col.startswith('volume'):
            col_map[col] = 'volume'
    df = df.rename(columns=col_map)
    return df

# --- Data Fetch & Analyze ---
@st.cache_data(ttl=300)
def fetch_data(ticker, interval, period="7d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def analyze_symbol(ticker, interval):
    df = fetch_data(ticker, interval)
    if df.empty or len(df) < 50:
        return None

    df = normalize_columns(df)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data for {ticker} missing required columns: {required_cols}")
        return None

    res = tech_indicator(df, ticker)
    return res

# --- Streamlit UI ---
def trader_dashboard():
    st.title("📊 Trader Dashboard: Multi-Timeframe Technical Analysis")

    timeframe = st.selectbox("Select Timeframe", ["5m", "15m"], index=0)

    # Select all symbols by default to run analysis on all
    all_symbols = op_symbols + list(index_name_map.keys())
    symbols_to_check = st.multiselect(
        "Select symbols to analyze",
        options=all_symbols,
        default=all_symbols  # All symbols preselected
    )

    if not symbols_to_check:
        st.warning("Please select at least one symbol.")
        return

    st.write(f"Analyzing {len(symbols_to_check)} symbols at {timeframe} timeframe...")

    results = []
    progress_bar = st.progress(0)

    for idx, sym in enumerate(symbols_to_check):
        yf_sym = index_name_map.get(sym, sym)
        st.text(f"Fetching and analyzing {sym} ({yf_sym})")
        res = analyze_symbol(yf_sym, timeframe)
        if res:
            results.append(res)
        progress_bar.progress((idx + 1) / len(symbols_to_check))
        time.sleep(1)  # throttle API calls

    if not results:
        st.warning("No results to display. Data may be missing or insufficient.")
        return

    df_res = pd.DataFrame(results, columns=["Symbol", "Trend", "Close", "Target", "Stop Loss"])

    # --- This is the filter shown AFTER analysis is done ---
    trends = df_res["Trend"].unique().tolist()
    selected_trends = st.multiselect("Filter by Trend", options=trends, default=trends)

    filtered_df = df_res[df_res["Trend"].isin(selected_trends)]

    st.dataframe(filtered_df, use_container_width=True)
