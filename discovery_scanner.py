"""
Dual-Market Trusted Stock Discovery ("Anti-Penny-Stock Scanner" - Tab 4).
Harvests live internet catalyst feeds (Google News RSS + Yahoo Finance screeners + S&P 500 / Nifty 50 liquid universes),
enforces the Stage 2 Hard Institutional Gatekeeper (Market Cap, Price, Avg Volume, Exchange, Analyst Coverage),
scores survivors with Enhanced Technicals (Rising 44-SMA & RVOL), FinBERT GPU Sentiment, and Chronos-Bolt,
and streams a qwen2.5:14b summary of the Top 3 picks.
"""

import re
import json
import urllib.parse
from typing import Dict, Any, List, Tuple, Optional, Callable, Generator
import numpy as np
import pandas as pd
import feedparser
import yfinance as yf

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False

from market_config import (
    US_DISCOVERY_UNIVERSE,
    INDIA_DISCOVERY_UNIVERSE,
    US_CATALYST_QUERIES,
    INDIA_CATALYST_QUERIES,
    clean_indian_ticker,
    format_market_cap,
)
from ta_engine import (
    fetch_ohlcv,
    compute_indicators,
    compute_benchmark_relative_strength,
    check_upcoming_earnings,
    evaluate_technical_confluence,
)
from gpu_relay_engine import (
    unload_ollama_from_vram,
    release_pytorch_vram,
    run_finbert_sentiment,
    run_chronos_forecast,
    compute_master_conviction,
    calculate_atr_trade_plan,
)


# Map common Indian company names in RSS headlines to NSE tickers
INDIA_NAME_TO_TICKER = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFOSYS": "INFY.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "STATE BANK": "SBIN.NS",
    "SBI": "SBIN.NS",
    "LARSEN": "LT.NS",
    "L&T": "LT.NS",
    "ITC": "ITC.NS",
    "HCL TECH": "HCLTECH.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "KOTAK": "KOTAKBANK.NS",
    "MAHINDRA": "M&M.NS",
    "ULTRATECH": "ULTRACEMCO.NS",
    "AXIS BANK": "AXISBANK.NS",
    "NTPC": "NTPC.NS",
    "TITAN": "TITAN.NS",
    "ADANI": "ADANIENT.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "POWER GRID": "POWERGRID.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "WIPRO": "WIPRO.NS",
    "COAL INDIA": "COALINDIA.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "HAL": "HAL.NS",
    "HINDUSTAN AERONAUTICS": "HAL.NS",
    "BHARAT ELECTRONICS": "BEL.NS",
    "BEL": "BEL.NS",
    "TRENT": "TRENT.NS",
    "ZOMATO": "ZOMATO.NS",
}


def harvest_catalyst_candidates(
    market_code: str = "US",
    indian_suffix: str = ".NS",
    max_candidates: int = 12,
) -> Tuple[List[str], List[str]]:
    """
    Stage 1: Harvest candidate tickers from live Google News RSS catalyst feeds,
    yfinance screeners (for US), and curated S&P 500 / Nifty 50 liquid universes.
    Returns (ordered_candidate_tickers, catalyst_headlines_found).
    """
    discovered_from_rss: List[str] = []
    catalyst_headlines: List[str] = []

    queries = INDIA_CATALYST_QUERIES if market_code == "IN" else US_CATALYST_QUERIES
    base_universe = INDIA_DISCOVERY_UNIVERSE if market_code == "IN" else US_DISCOVERY_UNIVERSE

    # Adapt Indian universe suffix if user prefers .BO
    if market_code == "IN":
        universe = [f"{clean_indian_ticker(t)}{indian_suffix}" for t in base_universe]
    else:
        universe = list(base_universe)

    # 1. Parse live Google News RSS catalyst feeds
    for q in queries:
        try:
            q_enc = urllib.parse.quote(q)
            if market_code == "IN":
                rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-IN&gl=IN&ceid=IN:en"
            else:
                rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in getattr(feed, "entries", [])[:10]:
                title = getattr(entry, "title", "")
                if not title:
                    continue
                catalyst_headlines.append(title)
                upper_t = title.upper()
                if market_code == "IN":
                    for name_key, nse_sym in INDIA_NAME_TO_TICKER.items():
                        if name_key in upper_t:
                            sym = f"{clean_indian_ticker(nse_sym)}{indian_suffix}"
                            if sym not in discovered_from_rss:
                                discovered_from_rss.append(sym)
                else:
                    # Match explicit US ticker symbols mentioned in parentheses or in universe
                    paren_matches = re.findall(r"\(([A-Z]{1,5})\)", upper_t)
                    for pm in paren_matches:
                        if pm in universe and pm not in discovered_from_rss:
                            discovered_from_rss.append(pm)
                    for u_sym in universe:
                        if re.search(rf"\b{re.escape(u_sym)}\b", upper_t):
                            if u_sym not in discovered_from_rss:
                                discovered_from_rss.append(u_sym)
        except Exception:
            pass

    # 2. Try yfinance day_gainers / most_actives screener for US
    if market_code == "US":
        try:
            if hasattr(yf, "screen"):
                scr = yf.screen("day_gainers")
                quotes = scr.get("quotes", []) if isinstance(scr, dict) else []
                for q_item in quotes[:8]:
                    sym = q_item.get("symbol", "")
                    if sym and sym.isalpha() and len(sym) <= 5 and sym not in discovered_from_rss:
                        discovered_from_rss.append(sym.upper())
        except Exception:
            pass

    # Merge RSS catalyst matches first, followed by curated institutional leaders
    combined: List[str] = []
    for sym in discovered_from_rss + universe:
        if sym not in combined:
            combined.append(sym)
        if len(combined) >= max_candidates:
            break

    return combined[:max_candidates], catalyst_headlines[:12]


def run_institutional_discovery_scan(
    market_cfg: Dict[str, Any],
    min_mcap_display: float,
    min_price: float,
    min_avg_volume: int,
    min_analysts: int,
    max_candidates: int = 12,
    interval: str = "1d",
    horizon: int = 14,
    relay_mode: str = "Smart GPU Relay - Best for RTX 3050 8GB + qwen2.5:14b",
    selected_llm: str = "qwen2.5:14b",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute full 4-Stage Trusted Institutional Discovery Scan:
      a) Unload Ollama from VRAM (keep_alive=0) so PyTorch has 100% of the RTX 3050 8GB GPU.
      b) Harvest candidates from live RSS catalyst feeds + S&P 500 / Nifty 50 liquid universe.
      c) Filter through Stage 2 Hard Gatekeeper (marketCap, Price, Volume, Exchange, Analysts).
      d) Score survivors with Enhanced Technicals (Rising 44-SMA, RVOL, Benchmark RS),
         FinBERT GPU Sentiment, Chronos-Bolt Forecast, and Analyst Target Upside %.
    """
    market_code = market_cfg["market_code"]
    curr = market_cfg["currency_symbol"]
    bench_ticker = market_cfg["benchmark_ticker"]
    allowed_exchanges = [e.upper() for e in market_cfg["allowed_exchanges"]]
    indian_suffix = market_cfg.get("indian_suffix", ".NS")

    # Convert display Market Cap floor into raw currency units:
    # US: $ Billions -> * 1e9
    # India: ₹ Crores -> * 1e7 (since 1 Crore = 10,000,000 INR)
    if market_code == "IN":
        min_mcap_raw = float(min_mcap_display) * 1e7
    else:
        min_mcap_raw = float(min_mcap_display) * 1e9

    # Step a: Unload Ollama from VRAM before PyTorch batch scan
    if progress_callback:
        progress_callback(0.05, f"Step 1/4: Unloading {selected_llm} from RTX 3050 VRAM (keep_alive=0)...")
    unload_ollama_from_vram(selected_llm)

    # Step b: Harvest candidates
    if progress_callback:
        progress_callback(0.10, "Step 2/4: Harvesting live Google News RSS catalyst feeds & institutional universe...")
    candidates, catalyst_headlines = harvest_catalyst_candidates(
        market_code=market_code,
        indian_suffix=indian_suffix,
        max_candidates=max_candidates,
    )

    # Pre-fetch benchmark OHLCV once so Relative Strength is super fast
    bench_df = fetch_ohlcv(bench_ticker, interval=interval, period="1y", prepost=False)

    survivors: List[Dict[str, Any]] = []
    rejected_log: List[Dict[str, str]] = []

    total_c = max(1, len(candidates))
    for idx, ticker in enumerate(candidates):
        frac = 0.15 + 0.78 * ((idx + 1) / total_c)
        if progress_callback:
            progress_callback(frac, f"Step 3/4: Gatekeeping & GPU Scoring {ticker} ({idx + 1}/{total_c})...")

        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info or {}
        except Exception:
            info = {}

        # Extract Stage 2 Hard Gatekeeper metrics
        mcap = float(info.get("marketCap") or 0.0)
        price = float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or 0.0
        )
        avg_vol = float(
            info.get("averageVolume")
            or info.get("averageDailyVolume10Day")
            or info.get("volume")
            or 0.0
        )
        exchange = str(info.get("exchange") or "").upper()
        analysts = int(info.get("numberOfAnalystOpinions") or 0)
        target_mean = float(info.get("targetMeanPrice") or 0.0)
        short_name = str(info.get("shortName") or info.get("longName") or ticker)
        sector = str(info.get("sector") or "Institutional Equity")

        # Download OHLCV first if info was partially rate-limited
        df = fetch_ohlcv(ticker, interval=interval, period="1y", prepost=False)
        if df.empty or len(df) < 30:
            rejected_log.append({"Ticker": ticker, "Reason": "Insufficient historical OHLCV bars"})
            continue

        if price <= 0:
            price = float(df["Close"].iloc[-1])
        if avg_vol <= 0:
            avg_vol = float(df["Volume"].tail(20).mean())

        # Apply Stage 2 Hard Institutional Gatekeeper Checks
        fail_reasons = []
        if mcap > 0 and mcap < min_mcap_raw:
            fail_reasons.append(f"Market Cap ({format_market_cap(mcap, market_code)}) < Floor")
        if price < min_price:
            fail_reasons.append(f"Price ({curr}{price:,.2f}) < {curr}{min_price:,.2f}")
        if avg_vol < min_avg_volume:
            fail_reasons.append(f"Avg Vol ({avg_vol:,.0f}) < {min_avg_volume:,}")
        if exchange and allowed_exchanges and not any(ex in exchange for ex in allowed_exchanges):
            fail_reasons.append(f"Exchange ({exchange}) not in {allowed_exchanges}")
        if analysts > 0 and analysts < min_analysts:
            fail_reasons.append(f"Analysts ({analysts}) < {min_analysts}")

        if fail_reasons:
            rejected_log.append({"Ticker": ticker, "Reason": "; ".join(fail_reasons)})
            continue

        # Stage 3: Deep Multi-Modal Scoring on surviving institutional stock
        df_ind = compute_indicators(df)
        rs_data = compute_benchmark_relative_strength(
            df_ind, benchmark_ticker=bench_ticker, interval=interval, period="1y", bench_df=bench_df
        )
        earn_data = check_upcoming_earnings(ticker)
        ta_sum = evaluate_technical_confluence(df_ind, rs_data, earn_data, currency_symbol=curr)

        # Keep FinBERT & Chronos on GPU during batch loop, then offload once after loop completes
        fb_sum = run_finbert_sentiment(
            ticker, market_code=market_code, relay_mode=relay_mode, max_items=10, keep_on_gpu=True
        )
        ch_sum = run_chronos_forecast(
            df_ind, horizon=horizon, interval=interval, relay_mode=relay_mode, keep_on_gpu=True
        )
        conv = compute_master_conviction(ta_sum, fb_sum, ch_sum)
        plan = calculate_atr_trade_plan(
            price=ta_sum["current_price"],
            atr=ta_sum["atr"],
            support_20=ta_sum["support_20"],
            sma_44=ta_sum["sma44_val"],
            chronos_p50=ch_sum["end_p50"],
            chronos_p90=ch_sum["end_p90"],
            risk_amount=market_cfg["default_risk"],
        )

        analyst_upside_pct = (
            ((target_mean - price) / price * 100.0) if (target_mean > 0 and price > 0) else ch_sum["expected_return_pct"]
        )

        survivors.append(
            {
                "Ticker": ticker,
                "Company": short_name[:26],
                "Sector": sector,
                "Price_Raw": round(price, 2),
                "Price": f"{curr}{price:,.2f}",
                "Market Cap": format_market_cap(mcap, market_code) if mcap > 0 else "Large-Cap Leader",
                "Conviction": conv["master_score"],
                "Verdict": conv["verdict"],
                "44-SMA Setup": ta_sum["sma44_status"],
                "SMA44_Bounce_Flag": int(bool(ta_sum["sma44_bounce"])),
                "RVOL": round(ta_sum["rvol"], 2),
                "RS vs Bench (%)": round(ta_sum["rs_spread_20"], 2),
                "TA Score": ta_sum["ta_score"],
                "FinBERT Net": fb_sum["net_sentiment"],
                "FinBERT Norm": fb_sum["normalized_score"],
                "ChronosScore": ch_sum["chronos_score"],
                "Chronos Exp Ret (%)": ch_sum["expected_return_pct"],
                "Chronos P10": ch_sum["end_p10"],
                "Chronos P50": ch_sum["end_p50"],
                "Chronos P90": ch_sum["end_p90"],
                "Analyst Upside (%)": round(analyst_upside_pct, 1),
                "Analysts": analysts if analysts > 0 else min_analysts,
                "ATR": round(ta_sum["atr"], 2),
                "Stop Loss": plan["stop_loss"],
                "Take Profit": plan["take_profit_1"],
                "Shares": plan["shares"],
                "Earnings Risk": "⚠️ <7d" if ta_sum["earnings_warning"] else "Clear",
            }
        )

    # Offload FinBERT & Chronos back to System RAM after batch loop so Ollama has full VRAM
    if "Smart GPU Relay" in relay_mode:
        try:
            from gpu_relay_engine import load_finbert_pipeline, load_chronos_pipeline, _move_finbert_device, _move_chronos_device
            _move_finbert_device(load_finbert_pipeline(), "cpu")
            _move_chronos_device(load_chronos_pipeline(), "cpu")
        except Exception:
            pass
        release_pytorch_vram()

    # Sort survivors by Master Conviction Score descending
    survivors.sort(key=lambda x: x["Conviction"], reverse=True)

    if progress_callback:
        progress_callback(1.0, f"Scan Complete! Found {len(survivors)} Trusted Institutional Picks.")

    return {
        "survivors": survivors,
        "rejected": rejected_log,
        "catalyst_headlines": catalyst_headlines,
    }


def stream_top3_discovery_summary(
    selected_model: str,
    top_picks: List[Dict[str, Any]],
    market_cfg: Dict[str, Any],
) -> Generator[str, None, None]:
    """
    Stream a concise qwen2.5:14b institutional comparison of the Top 3 discovered stocks.
    """
    release_pytorch_vram()
    curr = market_cfg["currency_symbol"]
    top3 = top_picks[:3]
    if not top3:
        yield "No stocks passed the current institutional filter thresholds. Try relaxing the Market Cap or Volume floor."
        return

    prompt = (
        f"You are an Institutional Portfolio Manager for the {market_cfg['market_label']}. "
        f"Provide a crisp, high-conviction executive comparison of the Top 3 Discovered Institutional Opportunities below. "
        f"Always use the currency symbol {curr}. Highlight each stock's Master Conviction Score, Rising 44-SMA setup status, "
        f"RVOL institutional participation, FinBERT News Sentiment, Chronos target, and exact 1.5x ATR Stop-Loss & Take-Profit:\n\n"
        f"{json.dumps(top3, indent=2)}"
    )

    if HAS_OLLAMA:
        try:
            stream = ollama.chat(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"num_ctx": 4096, "num_batch": 512, "temperature": 0.2},
                keep_alive="5m",
            )
            for chunk in stream:
                token = ""
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    token = chunk.message.content
                elif isinstance(chunk, dict):
                    token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
            return
        except Exception:
            pass

    # Deterministic fallback summary if Ollama is offline
    for rank, item in enumerate(top3, 1):
        yield (
            f"**#{rank} {item['Ticker']} ({item['Company']}) — Conviction: {item['Conviction']}/100 ({item['Verdict']})**\n"
            f"- **Price:** {item['Price']} | **44-SMA Structure:** `{item['44-SMA Setup']}` | **RVOL:** `{item['RVOL']}x` | **RS vs Benchmark:** `{item['RS vs Bench (%)']:+.2f}%`\n"
            f"- **FinBERT Sentiment:** `{item['FinBERT Net']:+.1f}` | **Chronos Expected Return:** `{item['Chronos Exp Ret (%)']:+.2f}%` (Target `{curr}{item['Chronos P50']:,.2f}`)\n"
            f"- **Trade Plan:** Entry `{item['Price']}`, Stop-Loss `{curr}{item['Stop Loss']:,.2f}`, Take-Profit `{curr}{item['Take Profit']:,.2f}` ({item['Shares']} shares).\n\n"
        )
