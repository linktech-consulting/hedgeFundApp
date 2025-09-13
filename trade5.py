# trade5.py

import argparse
import time
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import ta
from ollama import Client


# ===================== Config / Weights =====================
WEIGHTS = {
    "momentum": 3,
    "double": 2,
    "pullback": 2,
    "high_vol": 2,
    "rr_ok": 2,
    "session": 1,
}

IST = ZoneInfo("Asia/Kolkata")

SESSION_FILTER = {
    "enable": True,
    "start": "09:15",      # IST
    "end": "15:30",        # IST
    "high_vol_window": 60  # minutes
}

# ===================== Session Window =====================
def _parse_hhmm(s: str) -> dtime:
    return pd.to_datetime(s).time()

def within_session_filter(ts: pd.Timestamp, session_cfg: dict) -> bool:
    if not session_cfg.get("enable", True):
        return True
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo("UTC"))
    ts_ist = ts.tz_convert(IST)

    s_time = _parse_hhmm(session_cfg["start"])
    e_time = _parse_hhmm(session_cfg["end"])

    s_dt = datetime.combine(ts_ist.date(), s_time, tzinfo=IST)
    e_dt = datetime.combine(ts_ist.date(), e_time, tzinfo=IST)

    first_end = s_dt + pd.Timedelta(minutes=session_cfg["high_vol_window"])
    last_start = e_dt - pd.Timedelta(minutes=session_cfg["high_vol_window"])

    return (s_dt <= ts_ist <= first_end) or (last_start <= ts_ist <= e_dt)

# ===================== Columns =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, col)).lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    col_map = {}
    for col in df.columns:
        if col in ["open", "high", "low", "close", "volume"]:
            col_map[col] = col
        elif col.startswith("open"):
            col_map[col] = "open"
        elif col.startswith("high"):
            col_map[col] = "high"
        elif col.startswith("low"):
            col_map[col] = "low"
        elif col.startswith("close"):
            col_map[col] = "close"
        elif col.startswith("volume"):
            col_map[col] = "volume"
    return df.rename(columns=col_map)

# ===================== Fetchers =====================
def _default_period_for_interval(interval: str) -> str:
    intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
    if interval in intraday:
        return "7d"
    if interval in {"1h"}:
        return "30d"
    return "2y"

def fetch_one(ticker: str, interval: str, period: str | None) -> pd.DataFrame:
    try:
        if period is None:
            period = _default_period_for_interval(interval)
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_many_threaded(symbols: list[str], interval: str, period: str | None, max_workers: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, s, interval, period): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                out[sym] = fut.result()
            except Exception:
                out[sym] = pd.DataFrame()
    return out

def fetch_many_batched(symbols: list[str], interval: str, period: str | None, batch_size: int) -> dict[str, pd.DataFrame]:
    """
    Use yfinance multi-ticker download to reduce HTTP calls.
    We chunk symbols to avoid giant requests.
    """
    out: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
    if period is None:
        period = _default_period_for_interval(interval)
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i+batch_size]
        try:
            df_all = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                auto_adjust=False,
                threads=True,  # yfinance internal threading
            )
            # df_all may be MultiIndex columns when multiple tickers
            if isinstance(df_all.columns, pd.MultiIndex):
                for s in chunk:
                    if s in df_all.columns.get_level_values(0):
                        df_s = df_all[s].copy()
                        df_s.dropna(how="all", inplace=True)
                        out[s] = df_s
            else:
                # Single ticker fallback
                s = chunk[0]
                out[s] = df_all
        except Exception:
            # keep empties for this chunk
            pass
    return out

# ===================== Strategy / Signal =====================
@dataclass
class Signal:
    token_id: str
    trend: str
    entry: float
    target: float
    stop: float
    vol_tag: str
    score: int
    rr: float
    pattern: str
    signal_time: str
    atr: float
    vol_ratio: float
    candidate_count: int

def _build_signal(token_id, trend, entry, target, stop, vol_tag, score, rr, pattern, ts, atr, vol_ratio, cand_count) -> Signal:
    return Signal(
        token_id, trend,
        round(float(entry), 2),
        round(float(target), 2),
        round(float(stop), 2),
        vol_tag, int(score),
        round(float(rr), 2),
        pattern, str(ts),
        round(float(atr), 4),
        round(float(vol_ratio), 3),
        int(cand_count)
    )

def trend_strategy(
    df: pd.DataFrame,
    token_id: str,
    rr_min: float = 1.5,
    vol_ratio_min: float = 0.7,
    session_cfg: dict | None = None,
) -> Signal | None:

    session_cfg = session_cfg or SESSION_FILTER

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            return None

    close = df["close"].squeeze()
    high = df["high"].squeeze()
    low = df["low"].squeeze()
    volume = df["volume"].squeeze()

    try:
        df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        df["volume_sma20"] = volume.rolling(20).mean()
    except Exception:
        return None

    recent_bars = df["atr"].dropna().tail(30)
    avg_atr = (recent_bars.mean() if not recent_bars.empty else df["atr"].mean()) or 0.0

    best_overall: Signal | None = None

    for i in range(50, len(df)):
        ts = df.index[i]
        if session_cfg.get("enable", False) and not within_session_filter(ts, session_cfg):
            continue

        entry_price = float(close.iloc[i])
        atr = float(df["atr"].iloc[i]) if pd.notna(df["atr"].iloc[i]) else 0.0
        if atr <= 0 or avg_atr <= 0:
            continue

        uptrend = df["ema20"].iloc[i] > df["ema50"].iloc[i]
        downtrend = df["ema20"].iloc[i] < df["ema50"].iloc[i]

        vol_ratio = atr / avg_atr
        if vol_ratio < vol_ratio_min:
            continue

        if vol_ratio < 0.9:
            vol_tag = "Low"
        elif vol_ratio < 1.1:
            vol_tag = "Medium"
        elif vol_ratio < 1.4:
            vol_tag = "High"
        else:
            vol_tag = "Very High"

        candidates: list[Signal] = []

        recent_high = high.iloc[i-20:i].max()
        recent_low = low.iloc[i-20:i].min()
        prev_range = (high.iloc[i-5:i].max() - low.iloc[i-5:i].min()) or 0.0
        atr_expansion = (high.iloc[i] - low.iloc[i]) > 0.5 * prev_range

        # Momentum Up
        if (
            uptrend and close.iloc[i] > recent_high
            and df["rsi"].iloc[i] > 60
            and volume.iloc[i] > 1.3 * df["volume_sma20"].iloc[i]
            and atr_expansion
        ):
            stop = entry_price - atr
            target = entry_price + 2 * atr
            rr = abs(target - entry_price) / max(1e-9, abs(entry_price - stop))
            if rr >= rr_min:
                score = WEIGHTS["momentum"] + WEIGHTS["rr_ok"]
                if vol_ratio >= 1.1:
                    score += WEIGHTS["high_vol"]
                if session_cfg.get("enable", False):
                    score += WEIGHTS["session"]
                candidates.append(_build_signal(token_id, "Momentum Breakout (Up)", entry_price, target, stop,
                                               vol_tag, score, rr, "MomentumUp", ts, atr, vol_ratio, 0))

        # Momentum Down
        if (
            downtrend and close.iloc[i] < recent_low
            and df["rsi"].iloc[i] < 40
            and volume.iloc[i] > 1.3 * df["volume_sma20"].iloc[i]
            and atr_expansion
        ):
            stop = entry_price + atr
            target = entry_price - 2 * atr
            rr = abs(target - entry_price) / max(1e-9, abs(entry_price - stop))
            if rr >= rr_min:
                score = WEIGHTS["momentum"] + WEIGHTS["rr_ok"]
                if vol_ratio >= 1.1:
                    score += WEIGHTS["high_vol"]
                if session_cfg.get("enable", False):
                    score += WEIGHTS["session"]
                candidates.append(_build_signal(token_id, "Momentum Breakdown (Down)", entry_price, target, stop,
                                               vol_tag, score, rr, "MomentumDown", ts, atr, vol_ratio, 0))

        # Pullback Up
        if (
            uptrend
            and close.iloc[i-1] < df["ema20"].iloc[i-1]
            and close.iloc[i] > df["ema20"].iloc[i]
            and df["rsi"].iloc[i] > 55
            and volume.iloc[i] > 1.1 * df["volume_sma20"].iloc[i]
        ):
            stop = entry_price - atr
            target = entry_price + 2 * atr
            rr = abs(target - entry_price) / max(1e-9, abs(entry_price - stop))
            if rr >= rr_min:
                score = WEIGHTS["pullback"] + WEIGHTS["rr_ok"]
                if session_cfg.get("enable", False):
                    score += WEIGHTS["session"]
                candidates.append(_build_signal(token_id, "Pullback Up", entry_price, target, stop,
                                               vol_tag, score, rr, "PullbackUp", ts, atr, vol_ratio, 0))

        # Pullback Down
        if (
            downtrend
            and close.iloc[i-1] > df["ema20"].iloc[i-1]
            and close.iloc[i] < df["ema20"].iloc[i]
            and df["rsi"].iloc[i] < 45
            and volume.iloc[i] > 1.1 * df["volume_sma20"].iloc[i]
        ):
            stop = entry_price + atr
            target = entry_price - 2 * atr
            rr = abs(target - entry_price) / max(1e-9, abs(entry_price - stop))
            if rr >= rr_min:
                score = WEIGHTS["pullback"] + WEIGHTS["rr_ok"]
                if session_cfg.get("enable", False):
                    score += WEIGHTS["session"]
                candidates.append(_build_signal(token_id, "Pullback Down", entry_price, target, stop,
                                               vol_tag, score, rr, "PullbackDown", ts, atr, vol_ratio, 0))

        # Double patterns
        if i >= 7:
            p1, p2, p3, p4, p5, p6, p7 = list(close.iloc[i-7:i])
            tol = 0.5 * atr

            # Double Bottom
            neckline_db = max(p3, p5)
            cond_troughs_equal = abs(p2 - p4) <= tol and (p1 > p2 < p3) and (p3 > p4 < p5)
            if (
                cond_troughs_equal and uptrend and p7 > neckline_db
                and df["rsi"].iloc[i] > 50
                and volume.iloc[i] > 1.15 * df["volume_sma20"].iloc[i]
            ):
                stop = entry_price - atr
                target = entry_price + 2.5 * atr
                rr = abs(target - entry_price) / max(1e-9, abs(entry_price - stop))
                if rr >= rr_min:
                    score = WEIGHTS["double"] + WEIGHTS["rr_ok"]
                    if session_cfg.get("enable", False):
                        score += WEIGHTS["session"]
                    candidates.append(_build_signal(token_id, "Double Bottom+Neckline", entry_price, target, stop,
                                                   vol_tag, score, rr, "DoubleBottom", ts, atr, vol_ratio, 0))

            # Double Top
            neckline_dt = min(p3, p5)
            cond_peaks_equal = abs(p2 - p4) <= tol and (p1 < p2 > p3) and (p3 < p4 > p5)
            if (
                cond_peaks_equal and downtrend and p7 < neckline_dt
                and df["rsi"].iloc[i] < 50
                and volume.iloc[i] > 1.15 * df["volume_sma20"].iloc[i]
            ):
                stop = entry_price + atr
                target = entry_price - 2.5 * atr
                rr = abs(target - entry_price) / max(1e-9, abs(entry_price - stop))
                if rr >= rr_min:
                    score = WEIGHTS["double"] + WEIGHTS["rr_ok"]
                    if session_cfg.get("enable", False):
                        score += WEIGHTS["session"]
                    candidates.append(_build_signal(token_id, "Double Top+Neckline", entry_price, target, stop,
                                                   vol_tag, score, rr, "DoubleTop", ts, atr, vol_ratio, 0))

        if candidates:
            for c in candidates:
                c.candidate_count = len(candidates)
            bar_best = sorted(candidates, key=lambda x: (x.score, x.rr), reverse=True)[0]
            if (best_overall is None) or ((bar_best.score, bar_best.rr) > (best_overall.score, best_overall.rr)):
                best_overall = bar_best

    return best_overall

# ----------------- Analyze ONE symbol (for multiprocessing) -----------------
def analyze_one_symbol_payload(payload: dict) -> dict:
    """
    Payload keys:
      symbol, interval, rr_min, vol_ratio_min, session_cfg_dict, df_raw (DataFrame or None)
    Returns dict with: symbol, signal (dict or None), df_tail20 (string or None)
    """
    symbol = payload["symbol"]
    interval = payload["interval"]
    rr_min = payload["rr_min"]
    vol_ratio_min = payload["vol_ratio_min"]
    session_cfg = payload["session_cfg_dict"]
    df_raw = payload["df_raw"]

    if df_raw is None or df_raw.empty or len(df_raw) < 60:
        return {"symbol": symbol, "signal": None, "df_tail20": None}

    df = normalize_columns(df_raw.copy())
    sig = trend_strategy(df, symbol, rr_min=rr_min, vol_ratio_min=vol_ratio_min, session_cfg=session_cfg)
    if sig:
        try:
            tail20 = df.tail(20)[["open", "high", "low", "close", "volume"]].to_string()
        except Exception:
            tail20 = None
        return {"symbol": symbol, "signal": asdict(sig), "df_tail20": tail20}
    else:
        return {"symbol": symbol, "signal": None, "df_tail20": None}

# ===================== LLM =====================
def get_llm_signal(signal_row: Signal, df_tail20_str: str | None, model: str = "gemma3n:e2b") -> str:
    if df_tail20_str is None:
        df_tail20_str = "No Data"
    opp = {
        "Symbol": signal_row.token_id,
        "Trend": signal_row.trend,
        "Entry": signal_row.entry,
        "Target": signal_row.target,
        "Stop": signal_row.stop,
        "RR": signal_row.rr,
        "ATR": signal_row.atr,
        "VolRatio": signal_row.vol_ratio,
        "Pattern": signal_row.pattern,
        "SignalTime": signal_row.signal_time,
    }
    prompt = (
        f"Opportunity: {opp}\n"
        f"Recent Data:\n{df_tail20_str}\n"
        "Strictly reply:\n"
        "Action: <Strong Buy/Strong Sell/Neutral>\n"
        "Summary: <Very concise reason>\n"
        "Target: <number>\n"
        "Stop-Loss: <number>\n"
        "No extra explanation. No filler text."
    )
    client = Client(host="http://localhost:11434")

    try:
        response = client.generate(model=model, prompt=prompt, options={"temperature": 0})
        return (response.get("response") or "").strip()
    except Exception as e:
        print("LLM error:", e)
        return "LLM Error"

# ===================== Orchestrator =====================
def run_bulk_analysis_fast(
    symbols: list[str],
    timeframe: str,
    min_score: int,
    rr_min: float,
    vol_ratio_min: float,
    llm_model: str,
    throttle_sec: float,
    force_session: bool | None,
    max_workers_fetch: int,
    max_workers_analyze: int,
    batch_download: bool,
    batch_size: int,
    period: str | None,
) -> list[list]:

    daily_like = timeframe.lower() in {"1d", "1wk", "1mo"}
    session_cfg = SESSION_FILTER.copy()
    if force_session is not None:
        session_cfg["enable"] = bool(force_session)
    else:
        session_cfg["enable"] = session_cfg["enable"] and not daily_like

    # 1) Fetch data (batched or threaded)
    if batch_download:
        print(f"Fetching (batched) {len(symbols)} symbols, batch_size={batch_size} ...")
        raw_map = fetch_many_batched(symbols, timeframe, period, batch_size)
    else:
        print(f"Fetching (threaded) {len(symbols)} symbols with max_workers={max_workers_fetch} ...")
        raw_map = fetch_many_threaded(symbols, timeframe, period, max_workers_fetch)

    # 2) Analyze in parallel (process pool)
    print(f"Analyzing with {max_workers_analyze} processes ...")
    results_rows: list[list] = []
    payloads = []
    for s in symbols:
        payloads.append({
            "symbol": s,
            "interval": timeframe,
            "rr_min": rr_min,
            "vol_ratio_min": vol_ratio_min,
            "session_cfg_dict": session_cfg,
            "df_raw": raw_map.get(s, pd.DataFrame()),
        })

    with ProcessPoolExecutor(max_workers=max_workers_analyze) as pex:
        futs = [pex.submit(analyze_one_symbol_payload, pl) for pl in payloads]
        for fut in as_completed(futs):
            out = fut.result()
            symbol = out["symbol"]
            sig_dict = out["signal"]
            df_tail20 = out["df_tail20"]
            if sig_dict:
                sig = Signal(**sig_dict)
                # 3) LLM call only if score threshold met (on main process)
                if sig.score >= min_score:
                    llm_call = get_llm_signal(sig, df_tail20, model=llm_model)
                else:
                    llm_call = "LLM skipped (weak signal)"
                results_rows.append([
                    sig.token_id, sig.trend, sig.entry, sig.target, sig.stop,
                    sig.vol_tag, sig.score, sig.rr, sig.pattern, sig.signal_time,
                    llm_call, timeframe, sig.atr, sig.vol_ratio, sig.candidate_count
                ])
            if throttle_sec > 0:
                time.sleep(throttle_sec)

    return results_rows

# ===================== Universe =====================
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
    "SUNPHARMA.NS", "HINDALCO.NS", "M&M.NS", "SBICARD.NS", "^NSEI", "^NSEBANK"
]

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(description="Fast enhanced technical analysis (parallel fetch + analyze)")
    parser.add_argument("--timeframe", default="5m", help="1m,2m,5m,15m,30m,60m,90m,1h,1d,1wk,1mo")
    parser.add_argument("--output", default="trade_analysis_enhanced_fast.csv", help="Output CSV")
    parser.add_argument("--period", default=None, help="Override yfinance period (e.g., 7d, 30d, 2y)")
    parser.add_argument("--min_score", type=int, default=7, help="Min score to call LLM")
    parser.add_argument("--rr_min", type=float, default=1.5, help="Min Risk:Reward ratio")
    parser.add_argument("--vol_ratio_min", type=float, default=0.7, help="Min ATR/avgATR to consider")
    parser.add_argument("--throttle", type=float, default=0.0, help="Sleep between processed results (sec)")
    parser.add_argument("--llm_model", default="gemma3n:e2b", help="Ollama model")
    parser.add_argument("--force_session", choices=["on", "off"], default=None,
                        help="Force session filter ON/OFF (default: auto off for daily-like)")
    parser.add_argument("--max_workers_fetch", type=int, default=10, help="Threads for fetching")
    parser.add_argument("--max_workers_analyze", type=int, default=8, help="Processes for analysis")
    parser.add_argument("--batch_download", action="store_true", help="Use yfinance multi-ticker batching")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for multi-ticker download")

    args = parser.parse_args()

    force_session_bool = None
    if args.force_session is not None:
        force_session_bool = (args.force_session == "on")

    print(f"Universe: {len(op_symbols)} symbols | TF={args.timeframe} | batch={args.batch_download} | "
          f"fetch_workers={args.max_workers_fetch} | analyze_procs={args.max_workers_analyze}")

    rows = run_bulk_analysis_fast(
        symbols=op_symbols,
        timeframe=args.timeframe,
        min_score=args.min_score,
        rr_min=args.rr_min,
        vol_ratio_min=args.vol_ratio_min,
        llm_model=args.llm_model,
        throttle_sec=args.throttle,
        force_session=force_session_bool,
        max_workers_fetch=args.max_workers_fetch,
        max_workers_analyze=args.max_workers_analyze,
        batch_download=args.batch_download,
        batch_size=args.batch_size,
        period=args.period,
    )

    if not rows:
        print("No results found. Exiting.")
        return

    df_out = pd.DataFrame(
        rows,
        columns=[
            "Symbol", "Trend", "Entry", "Target", "Stop Loss", "Volatility",
            "Score", "RiskReward", "Pattern", "SignalTime", "LLM_Call",
            "Timeframe", "ATR", "VolRatio", "CandidateCount"
        ],
    )
    df_out.to_csv(args.output, index=False)
    print(f"Done. Saved: {args.output}")

if __name__ == "__main__":
    main()

"""
Examples:
# Fast threaded fetch + process pool analyze (recommended start):
python trade5.py --timeframe 5m --max_workers_fetch 10 --max_workers_analyze 8

# Batched download (fewer HTTP calls) + parallel analyze:
python trade5.py --timeframe 5m --batch_download --batch_size 25 --max_workers_analyze 8

# Daily TF (auto session off), gentler fetch:
python trade5.py --timeframe 1d --max_workers_fetch 6 --max_workers_analyze 6
"""
