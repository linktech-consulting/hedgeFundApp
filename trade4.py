import pandas as pd
import yfinance as yf
import ta
import time
import argparse
from ollama import Client
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Scoring weights ---
WEIGHTS = {
    "momentum": 3,
    "double": 2,
    "pullback": 2,
    "high_vol": 2,
    "rr_ok": 2,
    "session": 1,
    "confluence": 2,
    "trend_strength": 2,
}

# --- Trading Session Window in minutes (Indian market example) ---
SESSION_FILTER = {
    "enable": True,
    "start": "09:15",
    "end": "15:30",
    "high_vol_window": 60
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

def trend_strategy(df, token_id, benchmark_df=None):
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
    df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df["volume_sma20"] = volume.rolling(20).mean()
    recent_bars = df["atr"].dropna()[-30:] if len(df) > 30 else df["atr"].dropna()
    avg_atr = recent_bars.mean() if not recent_bars.empty else df["atr"].mean()

    for i in range(50, len(df)):
        ts = df.index[i]
        if SESSION_FILTER["enable"] and not within_session_filter(ts):
            continue

        entry_price = close.iloc[i]
        atr = df["atr"].iloc[i]
        adx = df["adx"].iloc[i]
        uptrend = df["ema20"].iloc[i] > df["ema50"].iloc[i]
        downtrend = df["ema20"].iloc[i] < df["ema50"].iloc[i]
        if adx < 25:
            continue

        # Market context check
        if benchmark_df is not None and len(benchmark_df) > i:
            bench_up = benchmark_df["ema20"].iloc[i] > benchmark_df["ema50"].iloc[i]
            if (uptrend and not bench_up) or (downtrend and bench_up):
                continue

        vol_tag = None
        score = 0
        trend = None
        pattern = []
        target = stop = None

        # Volatility regime
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1
        if vol_ratio < 0.8 or vol_ratio > 1.5:
            continue

        # Momentum breakout up
        recent_high = high.iloc[i-20:i].max()
        prev_range = high.iloc[i-5:i].max() - low.iloc[i-5:i].min()
        atr_expansion = (high.iloc[i] - low.iloc[i]) > 0.5 * prev_range
        vol_increasing = volume.iloc[i-5:i].is_monotonic_increasing
        rr = None
        if (
            uptrend and
            close.iloc[i] > recent_high and
            df["rsi"].iloc[i] > 65 and
            volume.iloc[i] > 1.5 * df["volume_sma20"].iloc[i] and
            atr_expansion and vol_increasing
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
            df["rsi"].iloc[i] < 35 and
            volume.iloc[i] > 1.5 * df["volume_sma20"].iloc[i] and
            atr_expansion and vol_increasing
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
            df["rsi"].iloc[i] > 60 and
            volume.iloc[i] > 1.2 * df["volume_sma20"].iloc[i]
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
            df["rsi"].iloc[i] < 40 and
            volume.iloc[i] > 1.2 * df["volume_sma20"].iloc[i]
        ):
            trend = "Pullback Down"
            pattern.append("PullbackDown")
            stop = entry_price + atr
            target = entry_price - 2 * atr
            rr = abs(target - entry_price) / abs(entry_price - stop)
            score += WEIGHTS["pullback"]

        # Double Bottom + Neckline (Dynamic lookback)
        lookback = min(30, max(10, i // 2))
        if i >= lookback:
            recent_closes = close.iloc[i-lookback:i]
            lows = recent_closes[recent_closes == recent_closes.rolling(5).min()]
            if len(lows) >= 2:
                low1, low2 = lows[-2:].values
                if abs(low1 - low2) / low1 < 0.02:
                    neckline_db = recent_closes[lows.index[-2]:].max()
                    if (
                        close.iloc[i] > neckline_db and
                        uptrend and
                        df["rsi"].iloc[i] > 55 and
                        volume.iloc[i] > 1.5 * df["volume_sma20"].iloc[i]
                    ):
                        trend = "Double Bottom+Neckline"
                        pattern.append("DoubleBottom")
                        stop = entry_price - atr
                        target = entry_price + 2.5 * atr
                        rr = abs(target - entry_price) / abs(entry_price - stop)
                        score += WEIGHTS["double"]

        # Double Top + Neckline (Dynamic lookback)
        if i >= lookback:
            recent_closes = close.iloc[i-lookback:i]
            highs = recent_closes[recent_closes == recent_closes.rolling(5).max()]
            if len(highs) >= 2:
                high1, high2 = highs[-2:].values
                if abs(high1 - high2) / high1 < 0.02:
                    neckline_dt = recent_closes[highs.index[-2]:].min()
                    if (
                        close.iloc[i] < neckline_dt and
                        downtrend and
                        df["rsi"].iloc[i] < 45 and
                        volume.iloc[i] > 1.5 * df["volume_sma20"].iloc[i]
                    ):
                        trend = "Double Top+Neckline"
                        pattern.append("DoubleTop")
                        stop = entry_price + atr
                        target = entry_price - 2.5 * atr
                        rr = abs(target - entry_price) / abs(entry_price - stop)
                        score += WEIGHTS["double"]

        # Confluence and Filters
        if len(pattern) < 2:
            continue
        if trend and rr and rr < 2.0:
            continue

        if trend:
            score += WEIGHTS["rr_ok"]
            if SESSION_FILTER["enable"]:
                score += WEIGHTS["session"]
            if len(pattern) >= 2:
                score += WEIGHTS["confluence"]
            if adx > 30:
                score += WEIGHTS["trend_strength"]

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

def fetch_data(ticker, interval, period="7d", retries=3):
    for attempt in range(retries):
        try:
            if interval == '1d':
                df = yf.download(ticker, interval=interval, progress=False, auto_adjust=True)
            else:
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logging.warning(f"Retry {attempt+1}/{retries} for {ticker}: {e}")
            time.sleep(1)
    logging.error(f"Failed to fetch data for {ticker}")
    return pd.DataFrame()

def analyze_symbol(args):
    sym, interval, benchmark_df, min_score, model = args
    logging.info(f"Analyzing {sym}...")
    df = fetch_data(sym, interval)
    if df.empty or len(df) < 50:
        return None
    df = normalize_columns(df)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return None
    res = trend_strategy(df, sym, benchmark_df)
    if res and res[1] != "No Signal" and res[6] is not None and res[6] >= min_score:
        llm_call = get_llm_signal(res, df, model)
        res.append(llm_call)
        return res
    elif res and res[1] != "No Signal":
        res.append("LLM skipped (weak signal)")
        return res
    return None

def get_llm_signal(signal_row, df, model='gemma3n:e2b'):
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
        return response['response'].strip()
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "LLM Error"

def run_bulk_analysis(symbols_to_check, timeframe, min_score=8, processes=4, model='gemma3n:e2b'):
    # Fetch benchmark once
    benchmark_df = fetch_data('^NSEI', timeframe)
    if not benchmark_df.empty:
        benchmark_df = normalize_columns(benchmark_df)
        # Compute EMAs for benchmark
        close = benchmark_df['close']
        benchmark_df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        benchmark_df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    results = []
    args_list = [(sym, timeframe, benchmark_df, min_score, model) for sym in symbols_to_check]

    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        for res in executor.map(analyze_symbol, args_list):
            if res:
                results.append(res)

    return results

# --- List of symbols ---
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
    parser.add_argument('--processes', type=int, default=4, help='Number of parallel processes (default: 4)')
    args = parser.parse_args()

    logging.info(f"Running trade analysis for {len(op_symbols)} symbols at timeframe {args.timeframe} with {args.processes} processes")
    results = run_bulk_analysis(op_symbols, args.timeframe, processes=args.processes)
    if not results:
        logging.info("No results found. Exiting.")
    else:
        df = pd.DataFrame(results, columns=[
            "Symbol", "Trend", "Entry", "Target", "Stop Loss", "Volatility", "Score", "RiskReward", "Pattern", "SignalTime", "LLM_Call"
        ])
        df.to_csv(args.output, index=False)
        logging.info(f"Analysis complete. Results saved to: {args.output}")

"python trade4.py --timeframe 5m --output results5.csv --processes 4"
