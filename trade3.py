# enhanced_trade_analysis.py

import pandas as pd
import yfinance as yf
import ta
import time
import argparse
from ollama import Client

# --- Scoring weights (tune these for your market style) ---
WEIGHTS = {
    "momentum": 3,
    "double": 2,
    "pullback": 2,
    "high_vol": 2,
    "rr_ok": 2,
    "session": 1,
}

# --- Trading Session Window in minutes (Indian market example) ---
SESSION_FILTER = {
    "enable": True,
    "start": "09:15",  # market open
    "end": "15:30",    # market close
    "high_vol_window": 60  # Only look at first/last 60min
}

def within_session_filter(ts):
    if not SESSION_FILTER["enable"]:
        return True
    t = pd.to_datetime(ts).time()
    s = pd.to_datetime(SESSION_FILTER["start"]).time()
    e = pd.to_datetime(SESSION_FILTER["end"]).time()
    first_window = (t >= s) and (t <= (pd.Timestamp.combine(pd.Timestamp.today(), s) + pd.Timedelta(minutes=SESSION_FILTER["high_vol_window"])).time())
    last_window = (t >= (pd.Timestamp.combine(pd.Timestamp.today(), e) - pd.Timedelta(minutes=SESSION_FILTER["high_vol_window"])).time()) and (t <= e)
    return first_window or last_window

def trend_strategy(df, token_id):
    opportunities = []

    def find_col(name):
        for c in df.columns:
            if name in str(c).lower():
                return c
        raise Exception(f"Column for {name} not found.")
    close_col, high_col, low_col, open_col, vol_col = (find_col(x) for x in ['close','high','low','open','volume'])
    close, high, low, open_, volume = (df[x].squeeze() for x in [close_col, high_col, low_col, open_col, vol_col])

    # Indicators
    df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["volume_sma20"] = volume.rolling(20).mean()
    recent_bars = df["atr"].dropna()[-30:] if len(df) > 30 else df["atr"].dropna()
    avg_atr = recent_bars.mean() if not recent_bars.empty else df["atr"].mean()

    for i in range(30, len(df)):
        ts = df.index[i]
        if SESSION_FILTER["enable"] and not within_session_filter(ts):
            continue  # skip signals outside high volume periods

        entry_price = close.iloc[i]
        atr = df["atr"].iloc[i]
        uptrend = df["ema20"].iloc[i] > df["ema50"].iloc[i]
        downtrend = df["ema20"].iloc[i] < df["ema50"].iloc[i]
        vol_tag = None
        score = 0
        trend = None
        pattern = []
        target = stop = None

        # Volatility regime
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1
        if vol_ratio < 0.7:
            continue  # Skip "Very Low" volatility

        # Momentum breakout up
        recent_high = high.iloc[i-20:i].max()
        prev_range = high.iloc[i-5:i].max() - low.iloc[i-5:i].min()
        atr_expansion = (high.iloc[i] - low.iloc[i]) > 0.5 * prev_range
        rr = None
        if (
            uptrend and
            close.iloc[i] > recent_high and
            df["rsi"].iloc[i] > 60 and
            volume.iloc[i] > 1.3 * df["volume_sma20"].iloc[i] and
            atr_expansion
        ):
            trend = "Momentum Breakout (Up)"
            pattern.append("MomentumUp")
            stop = entry_price - atr
            target = entry_price + 2 * atr
            rr = abs(target - entry_price) / abs(entry_price - stop)
            score += WEIGHTS["momentum"]
            if vol_ratio >= 1.1: score += WEIGHTS["high_vol"]

        # Momentum breakout down
        recent_low = low.iloc[i-20:i].min()
        if (
            downtrend and
            close.iloc[i] < recent_low and
            df["rsi"].iloc[i] < 40 and
            volume.iloc[i] > 1.3 * df["volume_sma20"].iloc[i] and
            atr_expansion
        ):
            trend = "Momentum Breakdown (Down)"
            pattern.append("MomentumDown")
            stop = entry_price + atr
            target = entry_price - 2 * atr
            rr = abs(target - entry_price) / abs(entry_price - stop)
            score += WEIGHTS["momentum"]
            if vol_ratio >= 1.1: score += WEIGHTS["high_vol"]

        # Pullback Up
        if (
            uptrend and
            close.iloc[i-1] < df["ema20"].iloc[i-1] and
            close.iloc[i] > df["ema20"].iloc[i] and
            df["rsi"].iloc[i] > 55 and
            volume.iloc[i] > 1.1 * df["volume_sma20"].iloc[i]
        ):
            trend = "Pullback Up"
            pattern.append("PullbackUp")
            stop = entry_price - atr
            target = entry_price + 2 * atr
            rr = abs(target - entry_price) / abs(entry_price - stop)
            score += WEIGHTS["pullback"]

        # Pullback Down
        if (
            downtrend and
            close.iloc[i-1] > df["ema20"].iloc[i-1] and
            close.iloc[i] < df["ema20"].iloc[i] and
            df["rsi"].iloc[i] < 45 and
            volume.iloc[i] > 1.1 * df["volume_sma20"].iloc[i]
        ):
            trend = "Pullback Down"
            pattern.append("PullbackDown")
            stop = entry_price + atr
            target = entry_price - 2 * atr
            rr = abs(target - entry_price) / abs(entry_price - stop)
            score += WEIGHTS["pullback"]

        # Double Bottom + Neckline
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
                trend = "Double Bottom+Neckline"
                pattern.append("DoubleBottom")
                stop = entry_price - atr
                target = entry_price + 2.5 * atr
                rr = abs(target - entry_price) / abs(entry_price - stop)
                score += WEIGHTS["double"]

            # Double Top + Neckline
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
                trend = "Double Top+Neckline"
                pattern.append("DoubleTop")
                stop = entry_price + atr
                target = entry_price - 2.5 * atr
                rr = abs(target - entry_price) / abs(entry_price - stop)
                score += WEIGHTS["double"]

        # --- Risk:Reward filter ---
        if trend and rr and rr < 1.5:
            continue  # Skip low R:R setups

        if trend:
            score += WEIGHTS["rr_ok"]
            if SESSION_FILTER["enable"]:
                score += WEIGHTS["session"]

            # Volatility tag
            if vol_ratio < 0.9: vol_tag = "Low"
            elif vol_ratio < 1.1: vol_tag = "Medium"
            elif vol_ratio < 1.4: vol_tag = "High"
            else: vol_tag = "Very High"

            opportunities.append({
                "token_id": token_id, "trend": trend, "entry": entry_price, "target": target, "stop": stop,
                "vol_tag": vol_tag, "score": score, "risk_reward": round(rr,2) if rr else None,
                "pattern": ",".join(pattern), "index": i, "signal_time": str(ts)
            })

    # Return top scoring opportunity if any
    if opportunities:
        best = sorted(opportunities, key=lambda x: (x["score"], x["risk_reward"]), reverse=True)[0]
        return [
            best["token_id"], best["trend"], round(best["entry"],2) if best["entry"] else None,
            round(best["target"],2) if best["target"] else None,
            round(best["stop"],2) if best["stop"] else None,
            best["vol_tag"], best["score"], best["risk_reward"], best["pattern"], best["signal_time"]
        ]
    else:
        return [token_id, "No Signal", None, None, None, None, None, None, None, None]

# --- Rest of the code: data helpers, analysis, LLM call, and main() ---

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
        if interval=='1d':
            df = yf.download(ticker, interval=interval, progress=False, auto_adjust=False)
        else:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def analyze_symbol(ticker, interval):
    df = fetch_data(ticker, interval)
    if df.empty or len(df) < 50:
        return None, None
    df = normalize_columns(df)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return None, None
    res = trend_strategy(df, ticker)
    return res, df

def get_llm_signal(signal_row, df, model='gemma3n:e2b'):
    # Only send the last 20 rows and the generated opportunity to LLM for summary
    if df is not None:
        df_sample = df.tail(20)[['open','high','low','close','volume']].to_string()
    else:
        df_sample = "No Data"
    prompt = (
        f"Opportunity: {signal_row}\n"
        f"Recent Data:\n{df_sample}\n"
        "Strictly reply:\n"
        "Action: <Strong Buy/Strong Sell/Neutral>\n"
        "Summary: <Very concise reason>\n"
        "Target: <number>\n"
        "Stop-Loss: <number>\n"
        "No extra explanation. No filler text."
    )
    client = Client(host='http://localhost:11434')
    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0}
        )
        verdict = response['response'].strip()
        return verdict
    except Exception as e:
        print("LLM error:", e)
        return "LLM Error"

def run_bulk_analysis(symbols_to_check, timeframe, min_score=7):
    results = []
    for i, sym in enumerate(symbols_to_check):
        print(f"{i+1}/{len(symbols_to_check)}: Analyzing {sym} ...", flush=True)
        res, df = analyze_symbol(sym, timeframe)
        # Only call LLM for strong signals
        if res and res[1] != "No Signal" and res[6] is not None and res[6] >= min_score:
            llm_call = get_llm_signal(res, df)
            res.append(llm_call)
            results.append(res)
        elif res and res[1] != "No Signal":
            # Still save signal, but note that LLM was not called
            res.append("LLM skipped (weak signal)")
            results.append(res)
        time.sleep(0.2)
    return results


# --- List of symbols (short for testing, expand for prod) ---
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enhanced technical analysis and save to CSV")
    parser.add_argument('--timeframe', default='5m', help='Timeframe for analysis (default: 5m)')
    parser.add_argument('--output', default='trade_analysis_enhanced.csv', help='Output CSV file name')
    args = parser.parse_args()

    print(f"Running trade analysis for {len(op_symbols)} symbols at timeframe {args.timeframe}")
    results = run_bulk_analysis(op_symbols, args.timeframe)
    if not results:
        print("No results found. Exiting.")
    else:
        df = pd.DataFrame(results, columns=[
            "Symbol", "Trend", "Entry", "Target", "Stop Loss", "Volatility", "Score", "RiskReward", "Pattern", "SignalTime", "LLM_Call"
        ])
        df.to_csv(args.output, index=False)
        print(f"Analysis complete. Results saved to: {args.output}")

'''
Run with:
python trade3.py --timeframe 5m --output results4.csv
'''

