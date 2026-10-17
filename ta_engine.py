"""
Windows-Safe Technical Analysis Engine (TA-Lib + Pure Vectorized NumPy/Pandas Fallback).
Computes institutional swing structure, Rising 44-SMA setup, Bollinger Bands, Momentum (RSI, MACD, ADX, ATR),
Volume Confirmation (RVOL, OBV, Rolling 20-bar VWAP), Benchmark Relative Strength (SPY / ^NSEI),
20-bar Support/Resistance, 7 Japanese Candlestick Patterns, and Earnings Risk (<7 days).
"""

import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# Attempt TA-Lib import; silently fallback to native vectorized implementation on Windows if unavailable
try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False


def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and cast OHLCV columns to float64."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else str(c) for c in out.columns]
    # Remove duplicate columns if any
    out = out.loc[:, ~out.columns.duplicated()]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].fillna(0.0).astype("float64")
    else:
        out["Volume"] = 0.0
    return out


import time
import urllib.parse
import requests


_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
)


def _fetch_yahoo_v8_direct(
    ticker: str,
    interval: str = "1d",
    period: str = "1y",
    prepost: bool = False,
) -> pd.DataFrame:
    """
    Direct HTTP fallback against Yahoo Finance v8 chart JSON API (query1 & query2).
    Immune to yfinance curl_cffi thread/crumb cache glitches inside Streamlit reruns.
    """
    encoded_ticker = urllib.parse.quote(ticker.strip())
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "true" if prepost else "false",
        "events": "div,splits",
    }
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{encoded_ticker}"
        try:
            resp = _HTTP_SESSION.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            result_list = (payload.get("chart") or {}).get("result")
            if not result_list:
                continue
            res = result_list[0]
            timestamps = res.get("timestamp") or []
            indicators = res.get("indicators") or {}
            quote_list = indicators.get("quote") or []
            if not timestamps or not quote_list:
                continue
            q = quote_list[0]
            tz_name = (res.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
            dt_idx = pd.to_datetime(timestamps, unit="s", utc=True)
            try:
                dt_idx = dt_idx.tz_convert(tz_name)
            except Exception:
                pass

            df = pd.DataFrame(
                {
                    "Open": q.get("open", [np.nan] * len(timestamps)),
                    "High": q.get("high", [np.nan] * len(timestamps)),
                    "Low": q.get("low", [np.nan] * len(timestamps)),
                    "Close": q.get("close", [np.nan] * len(timestamps)),
                    "Volume": q.get("volume", [0.0] * len(timestamps)),
                },
                index=dt_idx,
            )
            df.index.name = "Date"
            # Apply split/dividend adjustment ratio on daily/weekly bars if adjclose is present
            adj_list = indicators.get("adjclose") or []
            if adj_list and interval in ("1d", "1wk") and "adjclose" in adj_list[0]:
                adj_close = pd.Series(adj_list[0]["adjclose"], index=dt_idx, dtype="float64")
                raw_close = pd.to_numeric(df["Close"], errors="coerce")
                ratio = (adj_close / raw_close.replace(0.0, np.nan)).fillna(1.0)
                for col in ("Open", "High", "Low", "Close"):
                    df[col] = pd.to_numeric(df[col], errors="coerce") * ratio

            df = flatten_ohlcv(df)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _fetch_single_symbol_ohlcv(
    ticker: str,
    interval: str = "1d",
    period: str = "1y",
    prepost: bool = False,
) -> pd.DataFrame:
    """Attempt v8 direct API first, then yf.Ticker.history, then yf.download."""
    # Layer 1: Direct Yahoo v8 JSON endpoint (fast, thread-safe in Streamlit)
    df = _fetch_yahoo_v8_direct(ticker, interval=interval, period=period, prepost=prepost)
    if not df.empty:
        return df

    # Layer 2: yf.Ticker(ticker).history
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period=period, interval=interval, prepost=prepost, auto_adjust=True)
        df = flatten_ohlcv(df)
        if not df.empty:
            return df
    except Exception:
        pass

    # Layer 3: yf.download
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            prepost=prepost,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        df = flatten_ohlcv(df)
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()


def fetch_ohlcv(
    ticker: str,
    interval: str = "1d",
    period: str = "1y",
    prepost: bool = False,
) -> pd.DataFrame:
    """
    Safely download and flatten OHLCV data from Yahoo Finance with multi-host retry
    and automatic NSE (.NS) <-> BSE (.BO) exchange fallback for Indian tickers.
    """
    if not ticker or not str(ticker).strip():
        return pd.DataFrame()

    clean_t = str(ticker).strip().upper()
    candidates = [clean_t]
    if clean_t.endswith(".NS"):
        candidates.append(clean_t[:-3] + ".BO")
    elif clean_t.endswith(".BO"):
        candidates.append(clean_t[:-3] + ".NS")

    for attempt in range(2):
        for sym in candidates:
            df = _fetch_single_symbol_ohlcv(sym, interval=interval, period=period, prepost=prepost)
            if not df.empty:
                return df
        if attempt == 0:
            time.sleep(0.25)

    return pd.DataFrame()



def _native_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _native_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _native_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.bfill()


def _native_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)

    atr = _native_atr(high, low, close, period=period).replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm_s.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm_s.ewm(alpha=1.0 / period, adjust=False).mean() / atr)

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / denom)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(20.0), plus_di.fillna(20.0), minus_di.fillna(20.0)


def _detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect 7 institutional Japanese Candlestick Patterns using TA-Lib if installed,
    or exact vectorized OHLC math fallback.
    Patterns: Hammer, Bullish Engulfing, Bearish Engulfing, Morning Star, Evening Star, Shooting Star, Doji.
    """
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    if HAS_TALIB:
        op, hi, lo, cl = o.values, h.values, l.values, c.values
        df["CDL_HAMMER"] = (talib.CDLHAMMER(op, hi, lo, cl) > 0).astype(int)
        df["CDL_BULL_ENGULFING"] = (talib.CDLENGULFING(op, hi, lo, cl) > 0).astype(int)
        df["CDL_BEAR_ENGULFING"] = (talib.CDLENGULFING(op, hi, lo, cl) < 0).astype(int)
        df["CDL_MORNING_STAR"] = (talib.CDLMORNINGSTAR(op, hi, lo, cl) > 0).astype(int)
        df["CDL_EVENING_STAR"] = (talib.CDLEVENINGSTAR(op, hi, lo, cl) < 0).astype(int)
        df["CDL_SHOOTING_STAR"] = (talib.CDLSHOOTINGSTAR(op, hi, lo, cl) < 0).astype(int)
        df["CDL_DOJI"] = (talib.CDLDOJI(op, hi, lo, cl) != 0).astype(int)
    else:
        body = (c - o).abs()
        candle_range = (h - l).replace(0.0, np.nan)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        avg_body = body.rolling(10, min_periods=1).mean().replace(0.0, 1e-6)

        # 1. Doji: very small body relative to total range
        doji = (body <= 0.12 * candle_range).fillna(False)

        # 2. Hammer: lower shadow >= 2x body, tiny upper shadow, occurring near lows
        hammer = (
            (lower_shadow >= 2.0 * np.maximum(body, 0.05 * candle_range))
            & (upper_shadow <= 0.35 * np.maximum(body, 0.1 * candle_range))
            & (body > 0)
        ).fillna(False)

        # 3. Shooting Star: upper shadow >= 2x body, tiny lower shadow
        shooting_star = (
            (upper_shadow >= 2.0 * np.maximum(body, 0.05 * candle_range))
            & (lower_shadow <= 0.35 * np.maximum(body, 0.1 * candle_range))
            & (body > 0)
        ).fillna(False)

        # 4. Bullish Engulfing: prev red candle, current green candle engulfs prev body
        prev_o = o.shift(1)
        prev_c = c.shift(1)
        prev_body = body.shift(1)
        bull_engulfing = (
            (prev_c < prev_o)
            & (c > o)
            & (c >= prev_o)
            & (o <= prev_c)
            & (body > prev_body)
        ).fillna(False)

        # 5. Bearish Engulfing: prev green candle, current red candle engulfs prev body
        bear_engulfing = (
            (prev_c > prev_o)
            & (c < o)
            & (o >= prev_c)
            & (c <= prev_o)
            & (body > prev_body)
        ).fillna(False)

        # 6. Morning Star (3-bar bullish reversal)
        o2, c2, body2 = o.shift(2), c.shift(2), body.shift(2)
        morning_star = (
            (c2 < o2)
            & (body2 > 0.6 * avg_body)
            & (prev_body < 0.45 * body2)
            & (c > o)
            & (c > (o2 + c2) / 2.0)
        ).fillna(False)

        # 7. Evening Star (3-bar bearish reversal)
        evening_star = (
            (c2 > o2)
            & (body2 > 0.6 * avg_body)
            & (prev_body < 0.45 * body2)
            & (c < o)
            & (c < (o2 + c2) / 2.0)
        ).fillna(False)

        df["CDL_HAMMER"] = hammer.astype(int)
        df["CDL_BULL_ENGULFING"] = bull_engulfing.astype(int)
        df["CDL_BEAR_ENGULFING"] = bear_engulfing.astype(int)
        df["CDL_MORNING_STAR"] = morning_star.astype(int)
        df["CDL_EVENING_STAR"] = evening_star.astype(int)
        df["CDL_SHOOTING_STAR"] = shooting_star.astype(int)
        df["CDL_DOJI"] = doji.astype(int)

    # Create human-readable pattern summary per row
    pattern_labels = []
    pattern_biases = []
    for idx in df.index:
        row_pats = []
        bias = 0
        if df.loc[idx, "CDL_BULL_ENGULFING"]:
            row_pats.append("Bullish Engulfing")
            bias += 1
        if df.loc[idx, "CDL_MORNING_STAR"]:
            row_pats.append("Morning Star")
            bias += 1
        if df.loc[idx, "CDL_HAMMER"]:
            row_pats.append("Hammer")
            bias += 1
        if df.loc[idx, "CDL_BEAR_ENGULFING"]:
            row_pats.append("Bearish Engulfing")
            bias -= 1
        if df.loc[idx, "CDL_EVENING_STAR"]:
            row_pats.append("Evening Star")
            bias -= 1
        if df.loc[idx, "CDL_SHOOTING_STAR"]:
            row_pats.append("Shooting Star")
            bias -= 1
        if df.loc[idx, "CDL_DOJI"] and not row_pats:
            row_pats.append("Doji")
        pattern_labels.append(", ".join(row_pats) if row_pats else "")
        pattern_biases.append(bias)

    df["CANDLE_PATTERN"] = pattern_labels
    df["CANDLE_BIAS"] = pattern_biases
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all institutional technical indicators on an OHLCV DataFrame.
    Automatically uses TA-Lib if installed or pure vectorized NumPy/Pandas fallback.
    """
    out = flatten_ohlcv(df)
    if out.empty:
        return out

    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    open_p = out["Open"]
    volume = out["Volume"]

    if HAS_TALIB and len(out) >= 30:
        c_arr = close.values
        h_arr = high.values
        l_arr = low.values
        v_arr = volume.values

        out["SMA_20"] = pd.Series(talib.SMA(c_arr, timeperiod=20), index=out.index)
        out["SMA_44"] = pd.Series(talib.SMA(c_arr, timeperiod=44), index=out.index)
        out["EMA_50"] = pd.Series(talib.EMA(c_arr, timeperiod=50), index=out.index)
        out["SMA_200"] = pd.Series(talib.SMA(c_arr, timeperiod=200), index=out.index)

        upper, middle, lower = talib.BBANDS(c_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        out["BB_UPPER"] = pd.Series(upper, index=out.index)
        out["BB_MIDDLE"] = pd.Series(middle, index=out.index)
        out["BB_LOWER"] = pd.Series(lower, index=out.index)

        out["RSI_14"] = pd.Series(talib.RSI(c_arr, timeperiod=14), index=out.index).fillna(50.0)
        macd, macdsignal, macdhist = talib.MACD(c_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        out["MACD"] = pd.Series(macd, index=out.index).fillna(0.0)
        out["MACD_SIGNAL"] = pd.Series(macdsignal, index=out.index).fillna(0.0)
        out["MACD_HIST"] = pd.Series(macdhist, index=out.index).fillna(0.0)

        out["ATR_14"] = pd.Series(talib.ATR(h_arr, l_arr, c_arr, timeperiod=14), index=out.index).bfill()
        out["ADX_14"] = pd.Series(talib.ADX(h_arr, l_arr, c_arr, timeperiod=14), index=out.index).fillna(20.0)
        out["PLUS_DI"] = pd.Series(talib.PLUS_DI(h_arr, l_arr, c_arr, timeperiod=14), index=out.index).fillna(20.0)
        out["MINUS_DI"] = pd.Series(talib.MINUS_DI(h_arr, l_arr, c_arr, timeperiod=14), index=out.index).fillna(20.0)
        out["OBV"] = pd.Series(talib.OBV(c_arr, v_arr), index=out.index).fillna(0.0)
    else:
        # Pure Vectorized NumPy/Pandas Fallback
        out["SMA_20"] = close.rolling(20, min_periods=1).mean()
        out["SMA_44"] = close.rolling(44, min_periods=1).mean()
        out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        out["SMA_200"] = close.rolling(200, min_periods=1).mean()

        std_20 = close.rolling(20, min_periods=1).std().fillna(0.0)
        out["BB_MIDDLE"] = out["SMA_20"]
        out["BB_UPPER"] = out["BB_MIDDLE"] + 2.0 * std_20
        out["BB_LOWER"] = out["BB_MIDDLE"] - 2.0 * std_20

        out["RSI_14"] = _native_rsi(close, 14)
        macd, macdsignal, macdhist = _native_macd(close, 12, 26, 9)
        out["MACD"] = macd.fillna(0.0)
        out["MACD_SIGNAL"] = macdsignal.fillna(0.0)
        out["MACD_HIST"] = macdhist.fillna(0.0)

        out["ATR_14"] = _native_atr(high, low, close, 14)
        adx, plus_di, minus_di = _native_adx(high, low, close, 14)
        out["ADX_14"] = adx
        out["PLUS_DI"] = plus_di
        out["MINUS_DI"] = minus_di

        direction = np.sign(close.diff().fillna(0.0))
        out["OBV"] = (direction * volume).cumsum()

    # Fill initial NaNs in Moving Averages gracefully for shorter series
    out["SMA_20"] = out["SMA_20"].fillna(close.expanding().mean())
    out["SMA_44"] = out["SMA_44"].fillna(close.expanding().mean())
    out["EMA_50"] = out["EMA_50"].fillna(close.ewm(span=50, adjust=False).mean())
    out["SMA_200"] = out["SMA_200"].fillna(close.expanding().mean())
    out["BB_MIDDLE"] = out["BB_MIDDLE"].fillna(out["SMA_20"])
    out["BB_UPPER"] = out["BB_UPPER"].fillna(close * 1.02)
    out["BB_LOWER"] = out["BB_LOWER"].fillna(close * 0.98)

    # Bollinger Bandwidth & Squeeze Detection
    bb_mid_safe = out["BB_MIDDLE"].replace(0.0, np.nan)
    out["BB_WIDTH"] = ((out["BB_UPPER"] - out["BB_LOWER"]) / bb_mid_safe * 100.0).fillna(0.0)
    bw_20th = out["BB_WIDTH"].rolling(50, min_periods=10).quantile(0.20).bfill()
    out["BB_SQUEEZE"] = (out["BB_WIDTH"] <= bw_20th).astype(int)

    # Volume Confirmation & Institutional Flow: RVOL, OBV 10-bar slope, Rolling 20-bar VWAP
    out["VOL_SMA_20"] = volume.rolling(20, min_periods=1).mean().replace(0.0, np.nan)
    out["RVOL"] = (volume / out["VOL_SMA_20"]).fillna(1.0)
    out["OBV_SLOPE_10"] = out["OBV"].diff(10).fillna(0.0)

    typical_price = (high + low + close) / 3.0
    tp_vol_20 = (typical_price * volume).rolling(20, min_periods=1).sum()
    vol_sum_20 = volume.rolling(20, min_periods=1).sum().replace(0.0, np.nan)
    out["VWAP"] = (tp_vol_20 / vol_sum_20).fillna(close)

    # 20-bar Support & Resistance
    out["RESISTANCE_20"] = high.rolling(20, min_periods=1).max()
    out["SUPPORT_20"] = low.rolling(20, min_periods=1).min()

    # Rising 44-SMA Swing Setup Detector (Siddharth Bhanushali style institutional pullback bounce)
    sma44_rising_series = out["SMA_44"] > out["SMA_44"].shift(5)
    low_near_sma44 = (
        ((low - out["SMA_44"]).abs() / out["SMA_44"].replace(0.0, np.nan)) <= 0.015
    )
    green_candle = close > open_p
    close_above_sma44 = close >= out["SMA_44"]
    out["SMA44_RISING"] = sma44_rising_series.fillna(False).astype(int)
    out["SMA44_BOUNCE"] = (
        sma44_rising_series & low_near_sma44 & close_above_sma44 & green_candle
    ).fillna(False).astype(int)

    # Detect Japanese Candlestick Patterns
    out = _detect_candlestick_patterns(out)
    return out


def compute_benchmark_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    interval: str = "1d",
    period: str = "1y",
    bench_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Calculate 20-bar Relative Strength vs. Benchmark (SPY for US, ^NSEI for India)
    and check whether the Benchmark is in a bullish regime (above its SMA_20 and EMA_50).
    """
    result = {
        "benchmark_ticker": benchmark_ticker,
        "stock_return_20": 0.0,
        "bench_return_20": 0.0,
        "rs_spread_20": 0.0,
        "bench_above_sma20": True,
        "bench_above_ema50": True,
        "bench_regime": "Bullish Regime",
        "outperforming": True,
    }
    if stock_df is None or len(stock_df) < 2:
        return result

    lookback = min(20, len(stock_df) - 1)
    stock_ret = (
        (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-1 - lookback]) - 1.0
    ) * 100.0
    result["stock_return_20"] = float(stock_ret)

    try:
        if bench_df is None or bench_df.empty:
            bench_df = fetch_ohlcv(benchmark_ticker, interval=interval, period=period, prepost=False)
        if bench_df is not None and len(bench_df) >= 2:
            b_lookback = min(20, len(bench_df) - 1)
            bench_ret = (
                (bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-1 - b_lookback]) - 1.0
            ) * 100.0
            b_sma20 = bench_df["Close"].rolling(20, min_periods=1).mean().iloc[-1]
            b_ema50 = bench_df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            b_close = bench_df["Close"].iloc[-1]

            above_20 = bool(b_close >= b_sma20)
            above_50 = bool(b_close >= b_ema50)
            rs_spread = float(stock_ret - bench_ret)

            if above_20 and above_50:
                regime = "Bullish Regime (Above SMA20 & EMA50)"
            elif above_50:
                regime = "Consolidating (Below SMA20, Above EMA50)"
            else:
                regime = "Risk-Off / Bearish (Below EMA50)"

            result.update(
                {
                    "bench_return_20": float(bench_ret),
                    "rs_spread_20": rs_spread,
                    "bench_above_sma20": above_20,
                    "bench_above_ema50": above_50,
                    "bench_regime": regime,
                    "outperforming": rs_spread >= 0.0,
                }
            )
    except Exception:
        pass

    return result


def check_upcoming_earnings(ticker: str) -> Dict[str, Any]:
    """
    Safely check yf.Ticker(ticker).calendar for any earnings date within the next 7 calendar days.
    Returns dict with earnings_warning (bool), earnings_date (str or None), and days_until (int or None).
    """
    info_out = {
        "earnings_warning": False,
        "earnings_date": None,
        "days_until": None,
        "badge_text": "No Imminent Earnings (<7d)",
    }
    if not ticker or ticker.startswith("^"):
        return info_out

    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return info_out

        dates_to_check: List[datetime.date] = []
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("Earnings High") or []
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, datetime.datetime):
                        dates_to_check.append(item.date())
                    elif isinstance(item, datetime.date):
                        dates_to_check.append(item)
            elif isinstance(raw, datetime.datetime):
                dates_to_check.append(raw.date())
            elif isinstance(raw, datetime.date):
                dates_to_check.append(raw)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            for val in cal.values.flatten():
                if isinstance(val, datetime.datetime):
                    dates_to_check.append(val.date())
                elif isinstance(val, datetime.date):
                    dates_to_check.append(val)
                elif isinstance(val, pd.Timestamp):
                    dates_to_check.append(val.date())

        today = datetime.date.today()
        upcoming = [d for d in dates_to_check if (d - today).days >= 0]
        if upcoming:
            nearest = min(upcoming, key=lambda d: (d - today).days)
            days_away = (nearest - today).days
            info_out["earnings_date"] = nearest.strftime("%Y-%m-%d")
            info_out["days_until"] = int(days_away)
            if 0 <= days_away <= 7:
                info_out["earnings_warning"] = True
                info_out["badge_text"] = f"⚠️ EARNINGS WARNING: {nearest.strftime('%b %d')} ({days_away}d away)"
            else:
                info_out["badge_text"] = f"Next Earnings: {nearest.strftime('%b %d')} ({days_away}d)"
    except Exception:
        pass

    return info_out


def evaluate_technical_confluence(
    df: pd.DataFrame,
    rs_data: Dict[str, Any],
    earnings_data: Dict[str, Any],
    currency_symbol: str = "$",
) -> Dict[str, Any]:
    """
    Evaluate comprehensive Technical Confluence Score (0-100), Rising 44-SMA status,
    and build the structured Technical Signal Breakdown table for Tab 1 & Tab 3.
    """
    if df is None or df.empty:
        return {
            "ta_score": 50.0,
            "sma44_status": "N/A",
            "sma44_rising": False,
            "sma44_bounce": False,
            "rvol": 1.0,
            "atr": 0.0,
            "current_price": 0.0,
            "pct_change": 0.0,
            "support_20": 0.0,
            "resistance_20": 0.0,
            "active_patterns": "None",
            "breakdown_rows": [],
        }

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    price = float(last["Close"])
    open_p = float(last["Open"])
    low_p = float(last["Low"])
    pct_change = float(((price - float(prev["Close"])) / float(prev["Close"])) * 100.0) if float(prev["Close"]) else 0.0

    sma20 = float(last["SMA_20"])
    sma44 = float(last["SMA_44"])
    ema50 = float(last["EMA_50"])
    sma200 = float(last["SMA_200"])

    # Rising 44-SMA check over last 5 bars (SMA_44.iloc[-1] > SMA_44.iloc[-6])
    idx_6 = -6 if len(df) >= 6 else 0
    sma44_prev5 = float(df["SMA_44"].iloc[idx_6])
    sma44_rising = bool(sma44 > sma44_prev5)
    sma44_slope_pct = ((sma44 - sma44_prev5) / sma44_prev5 * 100.0) if sma44_prev5 else 0.0

    low_within_1_5pct = abs(low_p - sma44) / sma44 <= 0.015 if sma44 else False
    green_candle = price > open_p
    sma44_bounce = bool(sma44_rising and low_within_1_5pct and (price >= sma44) and green_candle)
    # Also check if a bounce triggered on the immediate prior bar and is following through
    recent_bounce = bool(sma44_bounce or (len(df) >= 2 and bool(df["SMA44_BOUNCE"].iloc[-2]) and price >= sma44))

    if sma44_bounce:
        sma44_status = "🔥 RISING + A+ BOUNCE"
    elif recent_bounce:
        sma44_status = "✅ RISING + FOLLOW-THROUGH"
    elif sma44_rising and price >= sma44:
        sma44_status = "📈 Rising (Price Above)"
    elif sma44_rising and price < sma44:
        sma44_status = "⚠️ Rising (Price Below)"
    elif abs(sma44_slope_pct) < 0.15:
        sma44_status = "➖ Flat / Sideways"
    else:
        sma44_status = "📉 Falling (Downtrend)"

    rsi = float(last["RSI_14"])
    macd_val = float(last["MACD"])
    macd_sig = float(last["MACD_SIGNAL"])
    macd_hist = float(last["MACD_HIST"])
    adx = float(last["ADX_14"])
    plus_di = float(last["PLUS_DI"])
    minus_di = float(last["MINUS_DI"])
    atr = float(last["ATR_14"])
    rvol = float(last["RVOL"])
    obv_slope = float(last["OBV_SLOPE_10"])
    vwap = float(last["VWAP"])
    bb_squeeze = bool(last["BB_SQUEEZE"])
    bb_width = float(last["BB_WIDTH"])
    sup20 = float(last["SUPPORT_20"])
    res20 = float(last["RESISTANCE_20"])

    # Recent candlestick patterns (last 3 bars)
    recent_pats = []
    for i in range(max(0, len(df) - 3), len(df)):
        p_str = str(df["CANDLE_PATTERN"].iloc[i]).strip()
        if p_str:
            bar_tag = "Latest" if i == len(df) - 1 else f"-{len(df) - 1 - i}b"
            recent_pats.append(f"{p_str} ({bar_tag})")
    active_patterns_str = " | ".join(recent_pats) if recent_pats else "None Detected"
    last_candle_bias = int(df["CANDLE_BIAS"].iloc[-3:].sum())

    # Calculate Technical Confluence Score (0 - 100)
    score = 50.0

    # 1. Trend Alignment (Price vs SMA 20 / 44 / 50 / 200) -> up to +/- 18 pts
    ma_bull_count = sum([price > sma20, price > sma44, price > ema50, price > sma200])
    if ma_bull_count == 4:
        score += 16.0
        trend_status = "Bullish"
        trend_desc = f"Full Institutional Bull Stack (Price > SMA20/44/EMA50/SMA200)"
    elif ma_bull_count == 3:
        score += 9.0
        trend_status = "Bullish"
        trend_desc = f"Constructive Uptrend ({ma_bull_count}/4 Key MAs Accomplished)"
    elif ma_bull_count == 2:
        trend_status = "Neutral"
        trend_desc = f"Mixed / Transitional Structure (2/4 MAs Above)"
    elif ma_bull_count == 1:
        score -= 9.0
        trend_status = "Bearish"
        trend_desc = f"Weak Structure (Price Below 3 of 4 Key Moving Averages)"
    else:
        score -= 16.0
        trend_status = "Bearish"
        trend_desc = f"Full Bear Breakdown (Price Below SMA20/44/EMA50/SMA200)"

    # 2. Rising 44-SMA Swing Setup -> up to +12 / -6 pts
    if sma44_bounce:
        score += 12.0
        sma44_sig = "Bullish"
        sma44_desc = f"A+ Pullback Bounce off Rising 44-SMA ({currency_symbol}{sma44:,.2f}, 5b slope {sma44_slope_pct:+.2f}%)"
    elif sma44_rising and price >= sma44:
        score += 6.0
        sma44_sig = "Bullish"
        sma44_desc = f"44-SMA Rising ({sma44_slope_pct:+.2f}%) & Price Holding Above ({currency_symbol}{sma44:,.2f})"
    elif sma44_rising:
        score -= 2.0
        sma44_sig = "Neutral"
        sma44_desc = f"44-SMA Rising ({sma44_slope_pct:+.2f}%) but Price Trading Below ({currency_symbol}{sma44:,.2f})"
    else:
        score -= 6.0
        sma44_sig = "Bearish"
        sma44_desc = f"44-SMA Flat/Falling ({sma44_slope_pct:+.2f}% over 5 bars)"

    # 3. RSI(14) Momentum -> up to +/- 7 pts
    if 52.0 <= rsi <= 68.0:
        score += 7.0
        rsi_sig = "Bullish"
        rsi_desc = f"Healthy Bullish Momentum Zone (RSI {rsi:.1f})"
    elif 30.0 <= rsi < 42.0 and price >= sma200:
        score += 4.0
        rsi_sig = "Bullish"
        rsi_desc = f"Oversold Pullback Within Primary Uptrend (RSI {rsi:.1f})"
    elif rsi > 72.0:
        score -= 3.0
        rsi_sig = "Neutral"
        rsi_desc = f"Overbought Extension Warning (RSI {rsi:.1f} > 70)"
    elif rsi < 30.0:
        score -= 4.0
        rsi_sig = "Bearish"
        rsi_desc = f"Extreme Oversold / Heavy Selling Pressure (RSI {rsi:.1f})"
    elif 42.0 <= rsi < 52.0:
        rsi_sig = "Neutral"
        rsi_desc = f"Balanced / Neutral Equilibrium (RSI {rsi:.1f})"
    else:
        score -= 5.0
        rsi_sig = "Bearish"
        rsi_desc = f"Bearish Momentum Regime (RSI {rsi:.1f})"

    # 4. MACD (12, 26, 9) -> up to +/- 7 pts
    if macd_val > macd_sig and macd_hist > 0:
        score += 7.0
        macd_status = "Bullish"
        macd_desc = f"Bullish Crossover & Positive Histogram ({macd_hist:+.3f})"
    elif macd_hist > float(prev["MACD_HIST"]):
        score += 2.0
        macd_status = "Neutral"
        macd_desc = f"Histogram Recovering / Inflecting Higher ({macd_hist:+.3f})"
    else:
        score -= 7.0
        macd_status = "Bearish"
        macd_desc = f"Bearish Momentum Crossover ({macd_hist:+.3f})"

    # 5. ADX(14) Trend Strength -> up to +/- 5 pts
    if adx >= 25.0 and plus_di > minus_di:
        score += 5.0
        adx_status = "Bullish"
        adx_desc = f"Strong Institutional Trend (ADX {adx:.1f} > 25, +DI {plus_di:.1f} > -DI {minus_di:.1f})"
    elif adx >= 25.0 and minus_di > plus_di:
        score -= 5.0
        adx_status = "Bearish"
        adx_desc = f"Strong Bearish Trend (ADX {adx:.1f} > 25, -DI {minus_di:.1f} > +DI {plus_di:.1f})"
    else:
        adx_status = "Neutral"
        adx_desc = f"Range-Bound / Low Trend Intensity (ADX {adx:.1f} < 25)"

    # 6. Volume Confirmation (RVOL & OBV Slope) -> up to +/- 8 pts
    if rvol >= 1.5 and price >= open_p and obv_slope >= 0:
        score += 8.0
        vol_status = "Bullish"
        vol_desc = f"High Institutional Accumulation (RVOL {rvol:.2f}x >= 1.5x, OBV Rising)"
    elif obv_slope > 0 and rvol >= 1.0:
        score += 4.0
        vol_status = "Bullish"
        vol_desc = f"Steady Accumulation (RVOL {rvol:.2f}x, Positive 10-bar OBV Slope)"
    elif rvol >= 1.5 and price < open_p:
        score -= 8.0
        vol_status = "Bearish"
        vol_desc = f"High-Volume Institutional Distribution (RVOL {rvol:.2f}x on Red Candle)"
    elif obv_slope < 0:
        score -= 4.0
        vol_status = "Bearish"
        vol_desc = f"Negative 10-Bar OBV Slope (Distribution, RVOL {rvol:.2f}x)"
    else:
        vol_status = "Neutral"
        vol_desc = f"Average Volume Participation (RVOL {rvol:.2f}x)"

    # 7. Benchmark Relative Strength (vs SPY or ^NSEI) -> up to +/- 6 pts
    rs_spread = float(rs_data.get("rs_spread_20", 0.0))
    b_ticker = rs_data.get("benchmark_ticker", "SPY")
    if rs_spread >= 2.0 and rs_data.get("bench_above_ema50", True):
        score += 6.0
        rs_status = "Bullish"
        rs_desc = f"Market Leader: Outperforming {b_ticker} by {rs_spread:+.2f}% over 20 bars"
    elif rs_spread >= 0.0:
        score += 3.0
        rs_status = "Bullish"
        rs_desc = f"Outperforming {b_ticker} ({rs_spread:+.2f}% 20-bar alpha)"
    elif rs_spread <= -3.0:
        score -= 6.0
        rs_status = "Bearish"
        rs_desc = f"Laggard: Underperforming {b_ticker} by {rs_spread:+.2f}% over 20 bars"
    else:
        rs_status = "Neutral"
        rs_desc = f"In-Line / Slight Lag vs {b_ticker} ({rs_spread:+.2f}%)"

    # 8. Candlestick Patterns -> up to +/- 5 pts
    if last_candle_bias > 0:
        score += 5.0
        pat_status = "Bullish"
    elif last_candle_bias < 0:
        score -= 5.0
        pat_status = "Bearish"
    else:
        pat_status = "Neutral"

    ta_score = float(np.clip(score, 0.0, 100.0))

    breakdown_rows = [
        {
            "Signal / Factor": "Trend Alignment (SMA 20/44/50/200)",
            "Current Reading": f"Price {currency_symbol}{price:,.2f} | SMA20 {currency_symbol}{sma20:,.2f} | SMA200 {currency_symbol}{sma200:,.2f}",
            "Status": trend_status,
            "Institutional Commentary": trend_desc,
        },
        {
            "Signal / Factor": "Rising 44-SMA Swing Setup",
            "Current Reading": f"SMA44 {currency_symbol}{sma44:,.2f} ({sma44_status})",
            "Status": sma44_sig,
            "Institutional Commentary": sma44_desc,
        },
        {
            "Signal / Factor": "RSI (14) & Bollinger Bands",
            "Current Reading": f"RSI {rsi:.1f} | BB Width {bb_width:.1f}% {'(SQUEEZE)' if bb_squeeze else ''}",
            "Status": rsi_sig,
            "Institutional Commentary": rsi_desc,
        },
        {
            "Signal / Factor": "MACD (12, 26, 9) Momentum",
            "Current Reading": f"MACD {macd_val:.2f} | Signal {macd_sig:.2f} | Hist {macd_hist:+.2f}",
            "Status": macd_status,
            "Institutional Commentary": macd_desc,
        },
        {
            "Signal / Factor": "ADX (14) Trend Strength",
            "Current Reading": f"ADX {adx:.1f} (+DI {plus_di:.1f} / -DI {minus_di:.1f})",
            "Status": adx_status,
            "Institutional Commentary": adx_desc,
        },
        {
            "Signal / Factor": "RVOL / OBV / VWAP Flow",
            "Current Reading": f"RVOL {rvol:.2f}x | VWAP {currency_symbol}{vwap:,.2f} | OBV 10b {'Rising' if obv_slope >= 0 else 'Falling'}",
            "Status": vol_status,
            "Institutional Commentary": vol_desc,
        },
        {
            "Signal / Factor": f"Relative Strength vs {b_ticker}",
            "Current Reading": f"{rs_spread:+.2f}% vs {b_ticker} (Stock {rs_data.get('stock_return_20', 0.0):+.1f}% vs Bench {rs_data.get('bench_return_20', 0.0):+.1f}%)",
            "Status": rs_status,
            "Institutional Commentary": f"{rs_desc} | {rs_data.get('bench_regime', '')}",
        },
        {
            "Signal / Factor": "Candlestick Patterns & S/R",
            "Current Reading": f"Patterns: {active_patterns_str}",
            "Status": pat_status,
            "Institutional Commentary": f"20-Bar Support: {currency_symbol}{sup20:,.2f} | 20-Bar Resistance: {currency_symbol}{res20:,.2f}",
        },
    ]

    return {
        "ta_score": round(ta_score, 1),
        "sma44_status": sma44_status,
        "sma44_rising": sma44_rising,
        "sma44_bounce": sma44_bounce or recent_bounce,
        "sma44_val": sma44,
        "sma20_val": sma20,
        "ema50_val": ema50,
        "sma200_val": sma200,
        "rsi_14": rsi,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "adx_14": adx,
        "atr": atr,
        "rvol": rvol,
        "vwap": vwap,
        "obv_rising": obv_slope >= 0,
        "bb_squeeze": bb_squeeze,
        "current_price": price,
        "pct_change": pct_change,
        "support_20": sup20,
        "resistance_20": res20,
        "active_patterns": active_patterns_str,
        "candle_bias": last_candle_bias,
        "rs_spread_20": rs_spread,
        "earnings_warning": earnings_data.get("earnings_warning", False),
        "earnings_badge": earnings_data.get("badge_text", "No Imminent Earnings"),
        "breakdown_rows": breakdown_rows,
    }


def run_self_test() -> bool:
    """
    Automated self-test verifying that all required indicators compute cleanly
    without error on both 'NVDA' (US) and 'RELIANCE.NS' (India).
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    global HAS_TALIB
    required_cols = [
        "SMA_20", "SMA_44", "EMA_50", "SMA_200",
        "BB_UPPER", "BB_MIDDLE", "BB_LOWER",
        "RSI_14", "MACD", "MACD_SIGNAL", "MACD_HIST",
        "ATR_14", "ADX_14", "RVOL", "OBV", "VWAP",
        "SUPPORT_20", "RESISTANCE_20", "SMA44_RISING", "SMA44_BOUNCE",
        "CDL_HAMMER", "CDL_BULL_ENGULFING", "CDL_BEAR_ENGULFING",
        "CDL_MORNING_STAR", "CDL_EVENING_STAR", "CDL_SHOOTING_STAR", "CDL_DOJI",
    ]
    orig_talib = HAS_TALIB
    print(f"[TA Engine Self-Test] TA-Lib C-Extension Active: {HAS_TALIB} (Fallback Ready: True)")
    for test_ticker, bench, curr in [("NVDA", "SPY", "$"), ("RELIANCE.NS", "^NSEI", "₹")]:
        df = fetch_ohlcv(test_ticker, interval="1d", period="1y")
        if df.empty:
            raise RuntimeError(f"Failed to download test OHLCV for {test_ticker}")

        for mode_name, flag in [("TA-Lib C-Ext", orig_talib), ("Native Vectorized Fallback", False)]:
            HAS_TALIB = flag
            df_ind = compute_indicators(df)
            missing = [c for c in required_cols if c not in df_ind.columns]
            if missing:
                raise AssertionError(f"Missing indicator columns for {test_ticker} ({mode_name}): {missing}")
            if df_ind[required_cols].iloc[-1].isna().any():
                na_cols = df_ind[required_cols].columns[df_ind[required_cols].iloc[-1].isna()].tolist()
                raise AssertionError(f"NaN in final row for {test_ticker} ({mode_name}): {na_cols}")

        HAS_TALIB = orig_talib
        df_ind = compute_indicators(df)
        rs = compute_benchmark_relative_strength(df_ind, benchmark_ticker=bench, interval="1d", period="1y")
        earn = check_upcoming_earnings(test_ticker)
        summary = evaluate_technical_confluence(df_ind, rs, earn, currency_symbol=curr)
        print(
            f"  [PASS] {test_ticker}: Price={curr}{summary['current_price']:,.2f} | "
            f"TA Score={summary['ta_score']}/100 | SMA44={summary['sma44_status']} | "
            f"RVOL={summary['rvol']:.2f}x | RS vs {bench}={summary['rs_spread_20']:+.2f}% | "
            f"ATR={curr}{summary['atr']:.2f}"
        )
    print("[TA Engine Self-Test] ALL CHECKS PASSED SUCCESSFULLY!")
    return True


if __name__ == "__main__":
    run_self_test()
