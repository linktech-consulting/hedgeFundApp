# trade.py
import pandas as pd
import yfinance as yf
import ta
import time
import argparse

# --- Candlestick Pattern Functions (same as before) ---
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

def is_double_top_neckline(data, i):
    if i < 5: return False
    p1, p2, p3, p4, p5, p6 = list(data.iloc[i-5:i+1]["close"])
    neckline = min(p2, p4)
    return p1 < p2 > p3 and p3 < p4 > p5 and p6 < neckline and p3 < neckline

def is_double_bottom_neckline(data, i):
    if i < 5: return False
    p1, p2, p3, p4, p5, p6 = list(data.iloc[i-5:i+1]["close"])
    neckline = max(p2, p4)
    return p1 > p2 < p3 and p3 > p4 < p5 and p6 > neckline and p3 > neckline

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

    pattern_flags = ["hammer", "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star", "piercing_line", "dark_cloud_cover", "shooting_star", "double_top_neckline", "double_bottom_neckline"]
    for pf in pattern_flags:
        data[pf] = False
    data["volume_breakout"] = False

    for i in range(5, len(data)):
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
        if is_double_top_neckline(data, i): data.at[data.index[i], "double_top_neckline"] = True
        if is_double_bottom_neckline(data, i): data.at[data.index[i], "double_bottom_neckline"] = True

    latest = data.iloc[-1]
    close_price = latest["close"]
    atr = latest["atr"]
    mfi = latest.get("mfi", None)
    rsi = latest["rsi"]
    higher_tf_ok = True if higher_tf_ema is None else close_price > higher_tf_ema

    # Enhanced confirmation logic
    if (
        latest["double_bottom_neckline"] and
        latest["volume"] > 1.5 * data["volume_sma"].iloc[-1] and
        rsi > 55 and
        latest["ema20"] > latest["ema50"]
    ):
        trend = "A+ Bullish Setup"
        stop_loss = round(close_price - atr, 2)
        target = round(close_price + 2.5 * (close_price - stop_loss), 2)
    elif (
        latest["double_top_neckline"] and
        latest["volume"] > 1.5 * data["volume_sma"].iloc[-1] and
        rsi < 45 and
        latest["ema20"] < latest["ema50"]
    ):
        trend = "A+ Bearish Setup"
        stop_loss = round(close_price + atr, 2)
        target = round(close_price - 2.5 * (stop_loss - close_price), 2)
    elif latest["double_bottom_neckline"]:
        trend = "Strong Bullish"
        stop_loss = round(close_price - atr, 2)
        target = round(close_price + 2.5 * (close_price - stop_loss), 2)
    elif latest["double_top_neckline"]:
        trend = "Strong Bearish"
        stop_loss = round(close_price + atr, 2)
        target = round(close_price - 2.5 * (stop_loss - close_price), 2)
    elif higher_tf_ok and latest["ema20"] > latest["ema50"] and rsi > 55 and mfi and mfi > 55:
        trend = "Bullish"
        stop_loss = round(close_price - atr, 2)
        target = round(close_price + 2 * (close_price - stop_loss), 2)
    elif latest["ema20"] < latest["ema50"] and rsi < 45 and mfi and mfi < 45:
        trend = "Bearish"
        stop_loss = round(close_price + atr, 2)
        target = round(close_price - 2 * (stop_loss - close_price), 2)
    else:
        trend = "Sideways"
        stop_loss = round(close_price - atr, 2)
        target = round(close_price + atr, 2)

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

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

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

def fetch_data(ticker, interval, period="7d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def analyze_symbol(ticker, interval):
    df = fetch_data(ticker, interval)
    if df.empty or len(df) < 50:
        return None
    df = normalize_columns(df)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return None
    res = tech_indicator(df, ticker)
    return res

def run_bulk_analysis(symbols_to_check, timeframe):
    results = []
    for i, sym in enumerate(symbols_to_check):
        yf_sym = index_name_map.get(sym, sym)
        print(f"{i+1}/{len(symbols_to_check)}: Analyzing {sym} ({yf_sym}) ...", flush=True)
        res = analyze_symbol(yf_sym, timeframe)
        if res:
            results.append(res)
        time.sleep(0.5)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run technical analysis and save to CSV")
    parser.add_argument('--timeframe', default='5m', help='Timeframe for analysis (default: 5m)')
    parser.add_argument('--output', default='trade_analysis_results.csv', help='Output CSV file name')
    args = parser.parse_args()

    print(f"Running trade analysis for {len(op_symbols)} symbols at timeframe {args.timeframe}")
    results = run_bulk_analysis(op_symbols, args.timeframe)
    if not results:
        print("No results found. Exiting.")
    else:
        df = pd.DataFrame(results, columns=["Symbol", "Trend", "Close", "Target", "Stop Loss"])
        df.to_csv(args.output, index=False)
        print(f"Analysis complete. Results saved to: {args.output}")
'''
python trade.py --timeframe 5m --output results.csv
'''
