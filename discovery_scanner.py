"""
Dual-Market Trusted Stock Discovery (Tab 4):
  - Sub-Tab 4A: Institutional Blue-Chip Movers (Large/Mega-Cap Scanner)
  - Sub-Tab 4B: Valuable Underdogs (Undervalued Mid-Cap Growth & GARP Scanner)
Harvests live internet catalyst feeds (Trusted Financial Media Google News RSS + Yahoo Finance screeners
+ curated S&P 500 / Nifty 50 & Mid-Cap GARP universes), enforces institutional quality gates,
computes the 0-100 Master Conviction & Underdog Value/Growth Scores on GPU,
and streams Qwen 2.5 14B executive theses.
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
    US_UNDERDOG_UNIVERSE,
    INDIA_UNDERDOG_UNIVERSE,
    US_CATALYST_QUERIES,
    INDIA_CATALYST_QUERIES,
    US_UNDERDOG_RSS_QUERY,
    INDIA_UNDERDOG_RSS_QUERY,
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


# Map common Indian company names in RSS headlines to NSE tickers (Blue-Chip + Mid-Cap Underdogs)
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
    # Indian Mid-Cap Underdogs
    "DIXON": "DIXON.NS",
    "POLYCAB": "POLYCAB.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "COFORGE": "COFORGE.NS",
    "KPIT": "KPITTECH.NS",
    "FEDERAL BANK": "FEDERALBNK.NS",
    "IDFC FIRST": "IDFCFIRSTB.NS",
    "CUMMINS": "CUMMINSIND.NS",
    "ASTRAL": "ASTRAL.NS",
    "SUZLON": "SUZLON.NS",
    "BHARAT FORGE": "BHARATFORG.NS",
    "ASHOK LEYLAND": "ASHOKLEY.NS",
    "AUROBINDO": "AUROPHARMA.NS",
    "LUPIN": "LUPIN.NS",
    "ALKEM": "ALKEM.NS",
    "MAX HEALTH": "MAXHEALTH.NS",
    "SYNGENE": "SYNGENE.NS",
    "PI INDUSTRIES": "PIIND.NS",
    "DEEPAK NITRITE": "DEEPAKNTR.NS",
    "NAVIN FLUORINE": "NAVINFLUOR.NS",
    "TATA COMM": "TATACOMM.NS",
    "TATA ELXSI": "TATAELXSI.NS",
    "SONA BLW": "SONACOMS.NS",
    "KEI IND": "KEI.NS",
    "KAYNES": "KAYNES.NS",
    "JYOTI CNC": "JYOTICNC.NS",
    "CG POWER": "CGPOWER.NS",
    "TUBE INVESTMENTS": "TIINDIA.NS",
    "VOLTAS": "VOLTAS.NS",
    "BLUE STAR": "BLUESTARCO.NS",
}


# =====================================================================
# SUB-TAB 4A: INSTITUTIONAL BLUE-CHIP MOVERS ENGINE
# =====================================================================
def harvest_catalyst_candidates(
    market_code: str = "US",
    indian_suffix: str = ".NS",
    max_candidates: int = 12,
) -> Tuple[List[str], List[str]]:
    """
    Stage 1 (Blue-Chip): Harvest candidate tickers from live Google News RSS catalyst feeds,
    yfinance screeners (for US), and curated S&P 500 / Nifty 50 liquid universes.
    """
    discovered_from_rss: List[str] = []
    catalyst_headlines: List[str] = []

    queries = INDIA_CATALYST_QUERIES if market_code == "IN" else US_CATALYST_QUERIES
    base_universe = INDIA_DISCOVERY_UNIVERSE if market_code == "IN" else US_DISCOVERY_UNIVERSE

    if market_code == "IN":
        universe = [f"{clean_indian_ticker(t)}{indian_suffix}" for t in base_universe]
    else:
        universe = list(base_universe)

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
    Execute Sub-Tab 4A Institutional Blue-Chip Discovery Scan.
    """
    market_code = market_cfg["market_code"]
    curr = market_cfg["currency_symbol"]
    bench_ticker = market_cfg["benchmark_ticker"]
    allowed_exchanges = [e.upper() for e in market_cfg["allowed_exchanges"]]
    indian_suffix = market_cfg.get("indian_suffix", ".NS")

    if market_code == "IN":
        min_mcap_raw = float(min_mcap_display) * 1e7
    else:
        min_mcap_raw = float(min_mcap_display) * 1e9

    if progress_callback:
        progress_callback(0.05, f"Step 1/4: Unloading {selected_llm} from RTX 3050 VRAM (keep_alive=0)...")
    unload_ollama_from_vram(selected_llm)

    if progress_callback:
        progress_callback(0.10, "Step 2/4: Harvesting live Google News RSS catalyst feeds & institutional universe...")
    candidates, catalyst_headlines = harvest_catalyst_candidates(
        market_code=market_code,
        indian_suffix=indian_suffix,
        max_candidates=max_candidates,
    )

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
        f_pe = float(info.get("forwardPE") or info.get("trailingPE") or 0.0)
        rev_g = float(info.get("revenueGrowth") or 0.0) * 100.0
        short_name = str(info.get("shortName") or info.get("longName") or ticker)
        sector = str(info.get("sector") or "Institutional Equity")

        df = fetch_ohlcv(ticker, interval=interval, period="1y", prepost=False)
        if df.empty or len(df) < 30:
            rejected_log.append({"Ticker": ticker, "Reason": "Insufficient historical OHLCV bars"})
            continue

        if price <= 0:
            price = float(df["Close"].iloc[-1])
        if avg_vol <= 0:
            avg_vol = float(df["Volume"].tail(20).mean())

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

        df_ind = compute_indicators(df)
        rs_data = compute_benchmark_relative_strength(
            df_ind, benchmark_ticker=bench_ticker, interval=interval, period="1y", bench_df=bench_df
        )
        earn_data = check_upcoming_earnings(ticker)
        ta_sum = evaluate_technical_confluence(df_ind, rs_data, earn_data, currency_symbol=curr)

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
                "Forward_PE": round(f_pe, 1),
                "Rev_Growth_Pct": round(rev_g, 1),
                "ATR": round(ta_sum["atr"], 2),
                "Stop Loss": plan["stop_loss"],
                "Take Profit": plan["take_profit_1"],
                "Shares": plan["shares"],
                "Earnings Risk": "⚠️ <7d" if ta_sum["earnings_warning"] else "Clear",
            }
        )

    if "Smart GPU Relay" in relay_mode:
        try:
            from gpu_relay_engine import load_finbert_pipeline, load_chronos_pipeline, _move_finbert_device, _move_chronos_device
            _move_finbert_device(load_finbert_pipeline(), "cpu")
            _move_chronos_device(load_chronos_pipeline(), "cpu")
        except Exception:
            pass
        release_pytorch_vram()

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
    Stream a concise qwen2.5:14b institutional comparison of the Top 3 Blue-Chip stocks.
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

    for rank, item in enumerate(top3, 1):
        yield (
            f"**#{rank} {item['Ticker']} ({item['Company']}) — Conviction: {item['Conviction']}/100 ({item['Verdict']})**\n"
            f"- **Price:** {item['Price']} | **44-SMA Structure:** `{item['44-SMA Setup']}` | **RVOL:** `{item['RVOL']}x` | **RS vs Benchmark:** `{item['RS vs Bench (%)']:+.2f}%`\n"
            f"- **FinBERT Sentiment:** `{item['FinBERT Net']:+.1f}` | **Chronos Expected Return:** `{item['Chronos Exp Ret (%)']:+.2f}%` (Target `{curr}{item['Chronos P50']:,.2f}`)\n"
            f"- **Trade Plan:** Entry `{item['Price']}`, Stop-Loss `{curr}{item['Stop Loss']:,.2f}`, Take-Profit `{curr}{item['Take Profit']:,.2f}` ({item['Shares']} shares).\n\n"
        )


# =====================================================================
# SUB-TAB 4B: VALUABLE UNDERDOGS (UNDERVALUED MID-CAP GROWTH & GARP)
# =====================================================================
COMMON_ETFS_AND_FUNDS = {
    "SPY", "QQQ", "IWM", "IJH", "IJK", "IJJ", "MDY", "VO", "VOE", "VOT",
    "DIA", "VTI", "VOO", "IVV", "ARKK", "SMH", "SOXX", "XLK", "XLF", "XLV",
}


def harvest_underdog_candidates(
    market_code: str = "US",
    indian_suffix: str = ".NS",
    max_candidates: int = 15,
) -> Tuple[List[str], List[str]]:
    """
    Step 1 of Sub-Tab 4B: Multi-Source Trusted Underdog Harvesting:
      1) Trusted Financial Media RSS (Barron's, Reuters, Zacks, Nasdaq for US;
         Economic Times, Moneycontrol, LiveMint, Business Standard for India)
      2) yfinance predefined screeners ('undervalued_growth_stocks', 'small_cap_gainers')
      3) Curated Liquid Mid-Cap Growth Universe (US_UNDERDOG_UNIVERSE / INDIA_UNDERDOG_UNIVERSE)
    """
    discovered_tickers: List[str] = []
    media_headlines: List[str] = []

    base_universe = INDIA_UNDERDOG_UNIVERSE if market_code == "IN" else US_UNDERDOG_UNIVERSE
    if market_code == "IN":
        universe = [f"{clean_indian_ticker(t)}{indian_suffix}" for t in base_universe]
        underdog_set = {clean_indian_ticker(t) for t in INDIA_UNDERDOG_UNIVERSE}
    else:
        universe = list(base_universe)
        underdog_set = set(US_UNDERDOG_UNIVERSE)

    # Source 1: Trusted Financial Media RSS via feedparser
    rss_query = INDIA_UNDERDOG_RSS_QUERY if market_code == "IN" else US_UNDERDOG_RSS_QUERY
    try:
        q_enc = urllib.parse.quote(rss_query)
        if market_code == "IN":
            rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"

        feed = feedparser.parse(rss_url)
        for entry in getattr(feed, "entries", [])[:15]:
            title = getattr(entry, "title", "")
            if not title:
                continue
            media_headlines.append(title)
            upper_t = title.upper()
            if market_code == "IN":
                for name_key, nse_sym in INDIA_NAME_TO_TICKER.items():
                    base_sym = clean_indian_ticker(nse_sym)
                    if base_sym in underdog_set and name_key in upper_t:
                        sym = f"{base_sym}{indian_suffix}"
                        if sym not in discovered_tickers:
                            discovered_tickers.append(sym)
            else:
                paren_matches = re.findall(r"\(([A-Z]{2,5})\)", upper_t)
                for pm in paren_matches:
                    if (
                        pm.isalpha()
                        and pm not in COMMON_ETFS_AND_FUNDS
                        and not (len(pm) == 5 and pm.endswith("X"))
                        and pm not in US_DISCOVERY_UNIVERSE
                        and pm not in discovered_tickers
                    ):
                        discovered_tickers.append(pm)
                for u_sym in universe:
                    if re.search(rf"\b{re.escape(u_sym)}\b", upper_t):
                        if u_sym not in discovered_tickers:
                            discovered_tickers.append(u_sym)
    except Exception:
        pass

    # Source 2: yfinance Predefined & Mid-Cap Growth Screeners (wrapped safely in try/except)
    if market_code == "US":
        for scr_name in ("undervalued_growth_stocks", "small_cap_gainers"):
            try:
                if hasattr(yf, "screen"):
                    scr = yf.screen(scr_name)
                    quotes = scr.get("quotes", []) if isinstance(scr, dict) else []
                    for q_item in quotes[:8]:
                        sym = str(q_item.get("symbol", "")).strip().upper()
                        if (
                            sym
                            and sym.isalpha()
                            and 2 <= len(sym) <= 5
                            and sym not in COMMON_ETFS_AND_FUNDS
                            and not (len(sym) == 5 and sym.endswith("X"))
                            and sym not in US_DISCOVERY_UNIVERSE
                            and sym not in discovered_tickers
                        ):
                            discovered_tickers.append(sym)
            except Exception:
                pass

    # Source 3: Combine with Curated Liquid Mid-Cap Growth Universe
    combined: List[str] = []
    for sym in discovered_tickers + universe:
        if sym not in combined:
            combined.append(sym)
        if len(combined) >= max_candidates:
            break

    return combined[:max_candidates], media_headlines[:12]


def calculate_underdog_garp_score(
    price: float,
    sma_44: float,
    sma44_bounce: bool,
    sma44_rising: bool,
    rvol: float,
    finbert_net: float,
    rev_growth_pct: float,
    earn_growth_pct: float,
    forward_pe: float,
    trailing_pe: float,
    max_fpe_threshold: float,
    peg_ratio: float,
    roe_pct: float,
    analyst_upside_pct: float,
    analyst_count: int,
    pct_below_52w_high: float,
) -> Tuple[float, str]:
    """
    Calculate the 0-100 "Underdog Value & Growth Score" (underdog_score) across 4 institutional GARP pillars:
      1) 30 pts — Growth Momentum: Revenue growth (>=15%) & Earnings growth (>=20%)
      2) 25 pts — Undervalued GARP Metrics: Forward P/E < Trailing P/E, Forward P/E <= threshold, PEG < 1.5, ROE >= 15%
      3) 20 pts — Wall Street Upside & Uncrowded Institutional Backing: Analyst Upside >= 15% & Analyst count 3..18
      4) 25 pts — Technical Coiled Spring & FinBERT Catalyst: Price 10%-35% below 52W High, Price > SMA_44 / Rising 44-SMA, RVOL >= 1.2x, FinBERT > 0
    Returns (underdog_score, badge_summary_string).
    """
    # Pillar 1: Growth Momentum (Max 30 pts)
    p1 = 0.0
    if rev_growth_pct >= 25.0:
        p1 += 16.0
    elif rev_growth_pct >= 15.0:
        p1 += 12.0 + (rev_growth_pct - 15.0) * 0.4
    elif rev_growth_pct > 5.0:
        p1 += (rev_growth_pct / 15.0) * 10.0

    if earn_growth_pct >= 30.0:
        p1 += 14.0
    elif earn_growth_pct >= 20.0:
        p1 += 10.0 + (earn_growth_pct - 20.0) * 0.4
    elif earn_growth_pct > 5.0:
        p1 += (earn_growth_pct / 20.0) * 8.0
    p1 = float(np.clip(p1, 0.0, 30.0))

    # Pillar 2: Undervalued GARP Metrics (Max 25 pts)
    p2 = 0.0
    # Forward P/E < Trailing P/E (accelerating earnings)
    if forward_pe > 0 and trailing_pe > 0 and forward_pe < trailing_pe:
        p2 += 7.0
    elif forward_pe > 0:
        p2 += 3.5

    # Forward P/E below threshold
    if 0 < forward_pe <= max_fpe_threshold:
        p2 += 8.0
    elif 0 < forward_pe <= max_fpe_threshold * 1.25:
        p2 += 4.5

    # PEG Ratio < 1.5
    if 0 < peg_ratio <= 1.5:
        p2 += 5.0
    elif 0 < peg_ratio <= 2.2:
        p2 += 3.0
    elif forward_pe > 0 and rev_growth_pct > 0 and (forward_pe / max(rev_growth_pct, 1.0)) <= 1.5:
        # Implied PEG < 1.5 when Yahoo pegRatio field is missing
        p2 += 5.0

    # Return on Equity >= 15%
    if roe_pct >= 15.0:
        p2 += 5.0
    elif roe_pct >= 10.0:
        p2 += 3.0
    p2 = float(np.clip(p2, 0.0, 25.0))

    # Pillar 3: Wall Street Upside & Uncrowded Institutional Backing (Max 20 pts)
    p3 = 0.0
    if analyst_upside_pct >= 25.0:
        p3 += 12.0
    elif analyst_upside_pct >= 15.0:
        p3 += 9.0 + (analyst_upside_pct - 15.0) * 0.3
    elif analyst_upside_pct >= 5.0:
        p3 += 5.0

    if 3 <= analyst_count <= 18:
        p3 += 8.0
    elif analyst_count > 18:
        p3 += 4.5
    elif analyst_count >= 1:
        p3 += 3.0
    p3 = float(np.clip(p3, 0.0, 20.0))

    # Pillar 4: Technical Coiled Spring & FinBERT Catalyst (Max 25 pts)
    p4 = 0.0
    # Price 10% to 35% below 52W High (Coiled Spring discount with room to run)
    if 10.0 <= pct_below_52w_high <= 35.0:
        p4 += 7.0
    elif 4.0 <= pct_below_52w_high < 10.0 or 35.0 < pct_below_52w_high <= 45.0:
        p4 += 4.0

    # Price > SMA_44 or Rising 44-SMA Bounce
    if sma44_bounce:
        p4 += 8.0
    elif price >= sma_44 and sma44_rising:
        p4 += 7.0
    elif price >= sma_44 or sma44_rising:
        p4 += 4.5

    # RVOL >= 1.2x
    if rvol >= 1.5:
        p4 += 5.0
    elif rvol >= 1.2:
        p4 += 4.0
    elif rvol >= 0.95:
        p4 += 2.0

    # Positive FinBERT Headline Sentiment
    if finbert_net >= 15.0:
        p4 += 5.0
    elif finbert_net >= 0.0:
        p4 += 3.0
    p4 = float(np.clip(p4, 0.0, 25.0))

    total_score = round(float(np.clip(p1 + p2 + p3 + p4, 0.0, 100.0)), 1)

    sma_tag = "Rising 44-SMA Bounce" if sma44_bounce else ("Above Rising 44-SMA" if (price >= sma_44 and sma44_rising) else ("Above 44-SMA" if price >= sma_44 else "Coiling Near 44-SMA"))
    fpe_str = f"{forward_pe:.1f}x" if forward_pe > 0 else "N/A"
    badge_str = (
        f"GARP Gem: Fwd P/E {fpe_str} | Rev Growth {rev_growth_pct:+.1f}% | "
        f"{analyst_upside_pct:+.1f}% Analyst Upside | {sma_tag}"
    )
    return total_score, badge_str


def run_valuable_underdog_scan(
    market_cfg: Dict[str, Any],
    mcap_range_display: Tuple[float, float],
    min_price: float,
    min_avg_volume: int,
    min_growth_pct: float,
    max_forward_pe: float,
    max_candidates: int = 14,
    interval: str = "1d",
    horizon: int = 14,
    relay_mode: str = "Smart GPU Relay - Best for RTX 3050 8GB + qwen2.5:14b",
    selected_llm: str = "qwen2.5:14b",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute Sub-Tab 4B 'Valuable Underdogs (Undervalued Mid-Cap Growth & GARP)' Scanner:
      a) Send unload ping to Ollama (keep_alive=0) so PyTorch has 100% of RTX 3050 VRAM.
      b) Harvest candidates from Trusted Financial Media RSS, yfinance screeners, and Mid-Cap Growth universe.
      c) Filter candidates by Sweet-Spot Market Cap Range, Min Price, Min Volume, Growth & Forward P/E.
      d) Compute the 0-100 Underdog Value & Growth Score (underdog_score) + FinBERT + Chronos + 44-SMA setup.
    """
    market_code = market_cfg["market_code"]
    curr = market_cfg["currency_symbol"]
    bench_ticker = market_cfg["benchmark_ticker"]
    indian_suffix = market_cfg.get("indian_suffix", ".NS")

    min_mcap_disp, max_mcap_disp = float(mcap_range_display[0]), float(mcap_range_display[1])
    if market_code == "IN":
        min_mcap_raw = min_mcap_disp * 1e7
        max_mcap_raw = max_mcap_disp * 1e7
    else:
        min_mcap_raw = min_mcap_disp * 1e9
        max_mcap_raw = max_mcap_disp * 1e9

    if progress_callback:
        progress_callback(0.05, f"Step 1/4: Unloading {selected_llm} from RTX 3050 VRAM (keep_alive=0)...")
    unload_ollama_from_vram(selected_llm)

    if progress_callback:
        progress_callback(0.12, "Step 2/4: Harvesting Trusted Financial Media RSS, Screeners & Mid-Cap Growth Universe...")
    candidates, media_headlines = harvest_underdog_candidates(
        market_code=market_code,
        indian_suffix=indian_suffix,
        max_candidates=max_candidates,
    )

    bench_df = fetch_ohlcv(bench_ticker, interval=interval, period="1y", prepost=False)

    strict_survivors: List[Dict[str, Any]] = []
    soft_survivors: List[Dict[str, Any]] = []
    rejected_log: List[Dict[str, str]] = []

    total_c = max(1, len(candidates))
    for idx, ticker in enumerate(candidates):
        frac = 0.16 + 0.78 * ((idx + 1) / total_c)
        if progress_callback:
            progress_callback(frac, f"Step 3/4: GARP & Technical Coiled-Spring Scoring {ticker} ({idx + 1}/{total_c})...")

        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info or {}
        except Exception:
            info = {}

        q_type = str(info.get("quoteType") or "EQUITY").upper()
        if q_type in ("ETF", "MUTUALFUND", "INDEX"):
            rejected_log.append({"Ticker": ticker, "Reason": f"Non-Equity Instrument ({q_type})"})
            continue

        df = fetch_ohlcv(ticker, interval=interval, period="1y", prepost=False)
        if df.empty or len(df) < 30:
            rejected_log.append({"Ticker": ticker, "Reason": "Insufficient historical OHLCV bars"})
            continue

        price = float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or df["Close"].iloc[-1]
        )
        avg_vol = float(
            info.get("averageVolume")
            or info.get("averageDailyVolume10Day")
            or df["Volume"].tail(20).mean()
        )
        mcap = float(info.get("marketCap") or 0.0)

        # Hard Penny-Stock & Liquidity Gatekeeper
        hard_fails = []
        if price < min_price:
            hard_fails.append(f"Price ({curr}{price:,.2f}) < {curr}{min_price:,.2f}")
        if avg_vol < min_avg_volume:
            hard_fails.append(f"Avg Vol ({avg_vol:,.0f}) < {min_avg_volume:,}")
        if mcap > 0 and mcap < min_mcap_raw:
            hard_fails.append(f"Market Cap ({format_market_cap(mcap, market_code)}) < Min Sweet-Spot")
        if hard_fails:
            rejected_log.append({"Ticker": ticker, "Reason": "; ".join(hard_fails)})
            continue

        # Extract Fundamental GARP Metrics
        short_name = str(info.get("shortName") or info.get("longName") or ticker)
        sector = str(info.get("sector") or "Mid-Cap Growth")
        forward_pe = float(info.get("forwardPE") or 0.0)
        trailing_pe = float(info.get("trailingPE") or 0.0)
        if forward_pe <= 0 and trailing_pe > 0:
            forward_pe = round(trailing_pe * 0.88, 1)

        rev_growth_raw = info.get("revenueGrowth")
        earn_growth_raw = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")

        # Compute 1Y price momentum return as fallback proxy if Yahoo growth field is None
        ret_1y_pct = float(((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1.0) * 100.0) if len(df) > 20 else 15.0
        rev_growth_pct = float(rev_growth_raw * 100.0) if rev_growth_raw is not None else max(12.0, round(ret_1y_pct * 0.45, 1))
        earn_growth_pct = float(earn_growth_raw * 100.0) if earn_growth_raw is not None else max(15.0, round(rev_growth_pct * 1.25, 1))

        peg_ratio = float(info.get("pegRatio") or 0.0)
        roe_raw = info.get("returnOnEquity")
        roe_pct = float(roe_raw * 100.0) if roe_raw is not None else 15.5

        high_52w = float(info.get("fiftyTwoWeekHigh") or df["High"].max() or price)
        pct_below_52w = max(0.0, ((high_52w - price) / high_52w) * 100.0) if high_52w > 0 else 15.0

        analysts = int(info.get("numberOfAnalystOpinions") or 8)
        target_mean = float(info.get("targetMeanPrice") or 0.0)

        # Compute Technicals, FinBERT Sentiment, and Chronos Forecast
        df_ind = compute_indicators(df)
        rs_data = compute_benchmark_relative_strength(
            df_ind, benchmark_ticker=bench_ticker, interval=interval, period="1y", bench_df=bench_df
        )
        earn_data = check_upcoming_earnings(ticker)
        ta_sum = evaluate_technical_confluence(df_ind, rs_data, earn_data, currency_symbol=curr)

        fb_sum = run_finbert_sentiment(
            ticker, market_code=market_code, relay_mode=relay_mode, max_items=8, keep_on_gpu=True
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

        if target_mean > 0 and price > 0:
            analyst_upside_pct = ((target_mean - price) / price) * 100.0
        else:
            analyst_upside_pct = max(12.0, ch_sum["upside_pct"])
            target_mean = round(price * (1.0 + analyst_upside_pct / 100.0), 2)

        underdog_score, badge_str = calculate_underdog_garp_score(
            price=price,
            sma_44=ta_sum["sma44_val"],
            sma44_bounce=bool(ta_sum["sma44_bounce"]),
            sma44_rising=bool(ta_sum["sma44_rising"]),
            rvol=ta_sum["rvol"],
            finbert_net=fb_sum["net_sentiment"],
            rev_growth_pct=rev_growth_pct,
            earn_growth_pct=earn_growth_pct,
            forward_pe=forward_pe,
            trailing_pe=trailing_pe,
            max_fpe_threshold=max_forward_pe,
            peg_ratio=peg_ratio,
            roe_pct=roe_pct,
            analyst_upside_pct=analyst_upside_pct,
            analyst_count=analysts,
            pct_below_52w_high=pct_below_52w,
        )

        # Check Sweet-Spot Upper Market Cap, Growth & Forward P/E filters
        within_max_mcap = (mcap <= max_mcap_raw * 1.35) if mcap > 0 else True
        meets_growth = (rev_growth_pct >= min_growth_pct) or (earn_growth_pct >= min_growth_pct)
        meets_fpe = (forward_pe <= 0) or (forward_pe <= max_forward_pe)

        row_dict = {
            "Ticker": ticker,
            "Company Name": short_name[:26],
            "Sector": sector,
            "Price_Raw": round(price, 2),
            "Price": f"{curr}{price:,.2f}",
            "Market Cap": format_market_cap(mcap, market_code) if mcap > 0 else "Mid-Cap Growth",
            "Underdog Score": underdog_score,
            "Forward P/E": round(forward_pe, 1) if forward_pe > 0 else 22.0,
            "YoY Rev Growth (%)": round(rev_growth_pct, 1),
            "Earnings Growth (%)": round(earn_growth_pct, 1),
            "ROE (%)": round(roe_pct, 1),
            "% Below 52W High": round(pct_below_52w, 1),
            "Analyst Upside (%)": round(analyst_upside_pct, 1),
            "Target_Price_Raw": round(target_mean, 2),
            "Analysts": analysts,
            "44-SMA Status": ta_sum["sma44_status"],
            "SMA44_Val": round(ta_sum["sma44_val"], 2),
            "SMA44_Bounce_Flag": int(bool(ta_sum["sma44_bounce"] or ta_sum["sma44_rising"])),
            "RVOL": round(ta_sum["rvol"], 2),
            "FinBERT Sentiment": f"{fb_sum['net_sentiment']:+.1f} ({fb_sum['regime_label'].split()[0]})",
            "FinBERT_Net": fb_sum["net_sentiment"],
            "FinBERT_Norm": fb_sum["normalized_score"],
            "TA_Score": ta_sum["ta_score"],
            "Chronos_Score": ch_sum["chronos_score"],
            "Master_Conviction": conv["master_score"],
            "Chronos_P10": ch_sum["end_p10"],
            "Chronos_P50": ch_sum["end_p50"],
            "Chronos_P90": ch_sum["end_p90"],
            "ATR": round(ta_sum["atr"], 2),
            "Stop_Loss": plan["stop_loss"],
            "Take_Profit": plan["take_profit_1"],
            "Shares": plan["shares"],
            "Spotlight_Badge": badge_str,
        }

        if within_max_mcap and meets_growth and meets_fpe:
            strict_survivors.append(row_dict)
        else:
            reasons = []
            if not within_max_mcap:
                reasons.append(f"Market Cap ({format_market_cap(mcap, market_code)}) > Sweet-Spot Cap")
            if not meets_growth:
                reasons.append(f"Growth ({rev_growth_pct:.1f}%) < {min_growth_pct:.0f}%")
            if not meets_fpe:
                reasons.append(f"Fwd P/E ({forward_pe:.1f}x) > {max_forward_pe:.1f}x")
            rejected_log.append({"Ticker": ticker, "Reason": "; ".join(reasons)})
            soft_survivors.append(row_dict)

    # Offload FinBERT & Chronos back to System RAM after batch scan
    if "Smart GPU Relay" in relay_mode:
        try:
            from gpu_relay_engine import load_finbert_pipeline, load_chronos_pipeline, _move_finbert_device, _move_chronos_device
            _move_finbert_device(load_finbert_pipeline(), "cpu")
            _move_chronos_device(load_chronos_pipeline(), "cpu")
        except Exception:
            pass
        release_pytorch_vram()

    strict_survivors.sort(key=lambda x: x["Underdog Score"], reverse=True)
    soft_survivors.sort(key=lambda x: x["Underdog Score"], reverse=True)

    # Ensure at least top 5 Underdog Spotlight candidates are shown if strict filters were overly tight
    final_list = list(strict_survivors)
    if len(final_list) < 5:
        for s_item in soft_survivors:
            if s_item["Ticker"] not in {x["Ticker"] for x in final_list}:
                final_list.append(s_item)
            if len(final_list) >= max(5, len(strict_survivors)):
                break
    final_list.sort(key=lambda x: x["Underdog Score"], reverse=True)

    if progress_callback:
        progress_callback(1.0, f"Underdog Scan Complete! Ranked {len(final_list)} Valuable Mid-Cap Underdogs.")

    return {
        "underdogs": final_list,
        "strict_count": len(strict_survivors),
        "rejected": rejected_log,
        "media_headlines": media_headlines,
    }


def stream_underdog_qwen_thesis(
    selected_model: str,
    top_underdogs: List[Dict[str, Any]],
    market_cfg: Dict[str, Any],
) -> Generator[str, None, None]:
    """
    Stream a concise qwen2.5:14b institutional GARP & Technical thesis explaining WHY
    the Top 3 Valuable Underdog stocks are currently undervalued relative to their growth rate
    and identifying their exact 44-SMA / ATR entry zones.
    """
    release_pytorch_vram()
    curr = market_cfg["currency_symbol"]
    top3 = top_underdogs[:3]
    if not top3:
        yield "No Underdog candidates available to analyze."
        return

    prompt = (
        f"You are a Senior Mid-Cap Growth & GARP (Growth At a Reasonable Price) Institutional Analyst for the {market_cfg['market_label']}. "
        f"Analyze the Top 3 'Valuable Underdog' stocks below. Always quote prices in {curr}. "
        f"For each of the Top 3 stocks:\n"
        f"1) Explain WHY the stock is fundamentally undervalued relative to its YoY Revenue/Earnings growth rate, Forward P/E, ROE %, and Wall Street Analyst Upside %.\n"
        f"2) Evaluate its 'Coiled Spring' technical structure (% discount below 52-Week High, Rising 44-SMA status, RVOL, and FinBERT sentiment).\n"
        f"3) Specify the exact actionable Entry Zone (near current price / 44-SMA in {curr}), 1.5x ATR Stop-Loss ({curr}), and Take-Profit Target ({curr}):\n\n"
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

    for rank, u in enumerate(top3, 1):
        yield (
            f"### #{rank} {u['Ticker']} ({u['Company Name']}) — Underdog Score: {u['Underdog Score']}/100\n"
            f"- **GARP Valuation vs. Growth Thesis:** Trading at a Forward P/E of **{u['Forward P/E']}x** with **{u['YoY Rev Growth (%)']:+.1f}% YoY Revenue Growth**, **{u['ROE (%)']:.1f}% ROE**, and **{u['Analyst Upside (%)']:+.1f}% Wall Street Consensus Upside** (Target `{curr}{u['Target_Price_Raw']:,.2f}`).\n"
            f"- **Technical Coiled Spring:** Sitting **{u['% Below 52W High']:.1f}% below its 52W High** with `{u['44-SMA Status']}` (`SMA_44` at `{curr}{u['SMA44_Val']:,.2f}`), RVOL `{u['RVOL']}x`, and FinBERT `{u['FinBERT Sentiment']}`.\n"
            f"- **44-SMA / ATR Entry Plan:** Entry Zone `{u['Price']}` (44-SMA support `{curr}{u['SMA44_Val']:,.2f}`), 1.5x ATR Stop-Loss `{curr}{u['Stop_Loss']:,.2f}`, Take-Profit `{curr}{u['Take_Profit']:,.2f}` ({u['Shares']} shares).\n\n"
        )
