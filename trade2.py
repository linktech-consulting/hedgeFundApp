


# trade_modular.py
import pandas as pd
import yfinance as yf
import ta
import time
import argparse

def trend_strategy(df, token_id):
    # --- Robust column finder ---
    def find_col(name):
        for c in df.columns:
            if name in str(c).lower():
                return c
        raise Exception(f"Column for {name} not found.")
    close_col = find_col('close')
    high_col = find_col('high')
    low_col = find_col('low')
    open_col = find_col('open')
    vol_col = find_col('volume')

    close = df[close_col].squeeze()
    high = df[high_col].squeeze()
    low = df[low_col].squeeze()
    open_ = df[open_col].squeeze()
    volume = df[vol_col].squeeze()

    # --- Indicators ---
    df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["volume_sma20"] = volume.rolling(20).mean()

    # --- Volatility base ---
    recent_bars = df["atr"].dropna()[-30:] if len(df) > 30 else df["atr"].dropna()
    avg_atr = recent_bars.mean() if not recent_bars.empty else df["atr"].mean()

    # --- Output variables ---
    signal, trend, entry, target, stop, vol_tag = None, "None", None, None, None, None

    # Start from bar 30 for full indicator coverage
    for i in range(30, len(df)):
        entry_price = close.iloc[i]
        atr = df["atr"].iloc[i]
        uptrend = df["ema20"].iloc[i] > df["ema50"].iloc[i]
        downtrend = df["ema20"].iloc[i] < df["ema50"].iloc[i]

        # --- 1. Trend momentum breakout ---
        recent_high = high.iloc[i-20:i].max()
        prev_range = high.iloc[i-5:i].max() - low.iloc[i-5:i].min()
        atr_expansion = (high.iloc[i] - low.iloc[i]) > 0.5 * prev_range

        if (
            uptrend and
            close.iloc[i] > recent_high and
            df["rsi"].iloc[i] > 60 and
            volume.iloc[i] > 1.3 * df["volume_sma20"].iloc[i] and
            atr_expansion
        ):
            signal, trend, entry = "long", "Momentum Breakout (Up)", entry_price
            stop = entry_price - atr
            target = entry_price + 2 * atr

        # Downtrend breakdown
        recent_low = low.iloc[i-20:i].min()
        if (
            downtrend and
            close.iloc[i] < recent_low and
            df["rsi"].iloc[i] < 40 and
            volume.iloc[i] > 1.3 * df["volume_sma20"].iloc[i] and
            atr_expansion
        ):
            signal, trend, entry = "short", "Momentum Breakdown (Down)", entry_price
            stop = entry_price + atr
            target = entry_price - 2 * atr

        # --- 2. Pullback Entries ---
        # Uptrend pullback to EMA20 and resume
        if (
            uptrend and
            close.iloc[i-1] < df["ema20"].iloc[i-1] and
            close.iloc[i] > df["ema20"].iloc[i] and
            df["rsi"].iloc[i] > 55 and
            volume.iloc[i] > 1.1 * df["volume_sma20"].iloc[i]
        ):
            signal, trend, entry = "long", "Pullback Up", entry_price
            stop = entry_price - atr
            target = entry_price + 2 * atr

        # Downtrend pullback to EMA20 and resume
        if (
            downtrend and
            close.iloc[i-1] > df["ema20"].iloc[i-1] and
            close.iloc[i] < df["ema20"].iloc[i] and
            df["rsi"].iloc[i] < 45 and
            volume.iloc[i] > 1.1 * df["volume_sma20"].iloc[i]
        ):
            signal, trend, entry = "short", "Pullback Down", entry_price
            stop = entry_price + atr
            target = entry_price - 2 * atr

        # --- 3. Double Bottom (bullish reversal) + neckline break ---
        if i >= 7:
            p1, p2, p3, p4, p5, p6, p7 = close.iloc[i-7:i]
            neckline_db = max(p2, p4)
            if (
                p1 > p2 < p3 and
                p3 > p4 < p5 and
                p2 < p4 and
                p7 > neckline_db and
                uptrend and
                df["rsi"].iloc[i] > 50 and
                volume.iloc[i] > 1.15 * df["volume_sma20"].iloc[i]
            ):
                signal, trend, entry = "long", "Double Bottom+Neckline", entry_price
                stop = entry_price - atr
                target = entry_price + 2.5 * atr

            # --- 4. Double Top (bearish reversal) + neckline break ---
            neckline_dt = min(p2, p4)
            if (
                p1 < p2 > p3 and
                p3 < p4 > p5 and
                p2 > p4 and
                p7 < neckline_dt and
                downtrend and
                df["rsi"].iloc[i] < 50 and
                volume.iloc[i] > 1.15 * df["volume_sma20"].iloc[i]
            ):
                signal, trend, entry = "short", "Double Top+Neckline", entry_price
                stop = entry_price + atr
                target = entry_price - 2.5 * atr

        # If signal generated, also calculate volatility for this bar!
        if signal:
            vol_ratio = atr / avg_atr if avg_atr > 0 else 1
            if vol_ratio < 0.7:
                vol_tag = "Very Low"
            elif vol_ratio < 0.9:
                vol_tag = "Low"
            elif vol_ratio < 1.1:
                vol_tag = "Medium"
            elif vol_ratio < 1.4:
                vol_tag = "High"
            else:
                vol_tag = "Very High"

    # Return only the *last* signal found, or "No Signal"
    if signal:
        return [
            token_id, trend, round(entry, 2) if entry else None,
            round(target, 2) if target else None,
            round(stop, 2) if stop else None,
            vol_tag
        ]
    else:
        return [token_id, "No Signal", None, None, None, None]

# --- Helper functions & main() logic would remain identical to your provided bulk script ---
# You can copy-paste the rest (normalize_columns, fetch_data, analyze_symbol, run_bulk_analysis, __main__ etc.)


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
    "SUNPHARMA.NS", "HINDALCO.NS", "M&M.NS", "SBICARD.NS","^NSEI", "^NSEBANK"
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
    return trend_strategy(df, ticker)

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
        df = pd.DataFrame(results, columns=["Symbol", "Trend", "Close", "Target", "Stop Loss", "Volatility"])

        df.to_csv(args.output, index=False)
        print(f"Analysis complete. Results saved to: {args.output}")
'''
python trade2.py --timeframe 5m --output results2.csv
'''
