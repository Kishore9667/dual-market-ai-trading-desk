"""
Hardware Tuning, Smart GPU Relay Pipeline (RTX 3050 8GB VRAM + 32GB System RAM),
FinBERT Institutional News Sentiment (ProsusAI/finbert),
Chronos-Bolt-Small Probabilistic Forecasting (amazon/chronos-bolt-small),
Master Conviction Scoring, and Ollama (qwen2.5:14b) Streaming Analyst & Chat Engine.
"""

import os
import gc
import json
import datetime
import urllib.parse
from typing import Dict, Any, List, Optional, Generator, Tuple

# Optimize RTX 3050 VRAM usage for Ollama KV cache & Flash Attention at startup
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import psutil
import feedparser
import yfinance as yf
import streamlit as st

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False

from market_config import build_google_news_rss_url, clean_indian_ticker

RELAY_MODES = [
    "Smart GPU Relay - Best for RTX 3050 8GB + qwen2.5:14b",
    "Keep PyTorch on GPU (Fastest Repeat Scans)",
    "CPU Only Mode (Preserve 100% VRAM for Ollama)",
]


# =====================================================================
# 1. HARDWARE TELEMETRY & OLLAMA CONNECTION
# =====================================================================
def get_hardware_telemetry() -> Dict[str, Any]:
    """
    Query live CUDA GPU (RTX 3050 8GB) VRAM via torch.cuda.mem_get_info()
    and 32GB System RAM pool via psutil.virtual_memory().
    """
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)
    ram_pct = ram.percent

    cuda_available = bool(HAS_TORCH and torch.cuda.is_available())
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            free_gb = free_b / (1024 ** 3)
            total_gb = total_b / (1024 ** 3)
            used_gb = total_gb - free_gb
            vram_pct = (used_gb / total_gb * 100.0) if total_gb > 0 else 0.0
        except Exception:
            gpu_name = "CUDA GPU Detected"
            free_gb, total_gb, used_gb, vram_pct = 0.0, 8.0, 0.0, 0.0
    else:
        gpu_name = "CPU Mode (CUDA Unavailable)"
        free_gb, total_gb, used_gb, vram_pct = 0.0, 0.0, 0.0, 0.0

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_used_gb": round(used_gb, 2),
        "vram_free_gb": round(free_gb, 2),
        "vram_total_gb": round(total_gb, 2),
        "vram_percent": round(vram_pct, 1),
        "ram_used_gb": round(ram_used_gb, 1),
        "ram_total_gb": round(ram_total_gb, 1),
        "ram_percent": round(ram_pct, 1),
    }


def get_ollama_models() -> Tuple[bool, List[str], int]:
    """
    Connect to Ollama at http://localhost:11434 and fetch locally installed models.
    Supports both object responses ([m.model for m in ollama.list().models])
    and dict responses ([m.get('name') or m.get('model') for m in ollama.list().get('models', [])]).
    Returns (is_online: bool, models_list: List[str], default_index: int).
    """
    if not HAS_OLLAMA:
        return False, ["qwen2.5:14b"], 0

    try:
        resp = ollama.list()
        models: List[str] = []
        if hasattr(resp, "models"):
            models = [ getattr(m, "model", None) or getattr(m, "name", "") for m in resp.models ]
        elif isinstance(resp, dict):
            models = [
                (m.get("name") or m.get("model") or "")
                for m in resp.get("models", [])
                if isinstance(m, dict)
            ]
        models = [m for m in models if m]
        if not models:
            return True, ["qwen2.5:14b"], 0

        default_idx = 0
        for i, name in enumerate(models):
            if name.lower().startswith("qwen2.5:14b"):
                default_idx = i
                break
        return True, models, default_idx
    except Exception:
        return False, ["qwen2.5:14b"], 0


def unload_ollama_from_vram(model_name: str = "qwen2.5:14b") -> bool:
    """
    Step 3 of Smart GPU Relay: Send an unload ping (keep_alive=0) to Ollama
    before batch-running FinBERT/Chronos on candidate stocks so PyTorch and Ollama never collide in VRAM.
    """
    if not HAS_OLLAMA or not model_name:
        return False
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        return True
    except Exception:
        return False


def release_pytorch_vram() -> None:
    """Run garbage collection and clear PyTorch CUDA cache so RTX 3050 VRAM is freed for Ollama."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# =====================================================================
# 2. CACHED FINBERT & CHRONOS-BOLT-SMALL MODELS + SMART GPU RELAY
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    """
    Load and cache ProsusAI/finbert text-classification pipeline.
    Initializes on GPU (device=0) if CUDA is available, and supports dynamic CPU/GPU relay.
    """
    from transformers import pipeline

    device_id = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
        device=device_id,
    )
    return pipe


@st.cache_resource(show_spinner=False)
def load_chronos_pipeline():
    """
    Load and cache amazon/chronos-bolt-small using BaseChronosPipeline.
    """
    from chronos import BaseChronosPipeline

    device_map = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if (HAS_TORCH and torch.cuda.is_available()) else torch.float32
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map=device_map,
        torch_dtype=dtype,
    )
    return pipe


def _move_finbert_device(pipe, target_device: str) -> None:
    """Safely move HuggingFace pipeline between 'cuda:0' and 'cpu'."""
    if not HAS_TORCH or pipe is None:
        return
    try:
        dev = torch.device(target_device)
        pipe.model.to(dev)
        pipe.device = dev
    except Exception:
        pass


def _move_chronos_device(pipe, target_device: str) -> None:
    """Safely move Chronos-Bolt model between 'cuda:0' and 'cpu'."""
    if not HAS_TORCH or pipe is None:
        return
    try:
        dev = torch.device(target_device)
        pipe.model.to(dev)
    except Exception:
        pass


# =====================================================================
# 3. DUAL-SOURCE NEWS HARVESTER & FINBERT SENTIMENT ENGINE (TAB 2)
# =====================================================================
def fetch_stock_news(ticker: str, market_code: str = "US", max_items: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch up to 20 latest deduplicated news headlines combining:
    1) yf.Ticker(ticker).news (handling both nested 'content' dicts and flat dicts)
    2) Localized Google News RSS (gl=US for US stocks, gl=IN with .NS/.BO stripped for Indian stocks via feedparser)
    """
    articles: List[Dict[str, Any]] = []
    seen_titles = set()

    def _add_article(title: str, link: str, publisher: str, pub_time: str, source_tag: str):
        if not title:
            return
        clean_title = title.strip()
        norm_key = "".join(ch.lower() for ch in clean_title if ch.isalnum())[:80]
        if not norm_key or norm_key in seen_titles:
            return
        seen_titles.add(norm_key)
        articles.append(
            {
                "title": clean_title,
                "link": link or f"https://www.google.com/search?q={urllib.parse.quote(clean_title)}",
                "publisher": publisher or source_tag,
                "published": pub_time or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "source": source_tag,
            }
        )

    # Source 1: Yahoo Finance Ticker.news (supports nested 'content' and flat dicts)
    try:
        raw_yf_news = yf.Ticker(ticker).news or []
        for item in raw_yf_news:
            if not isinstance(item, dict):
                continue
            content = item.get("content") if isinstance(item.get("content"), dict) else item

            title = content.get("title") or item.get("title") or ""
            # Extract URL
            link = ""
            if isinstance(content.get("canonicalUrl"), dict):
                link = content["canonicalUrl"].get("url", "")
            elif isinstance(content.get("clickThroughUrl"), dict):
                link = content["clickThroughUrl"].get("url", "")
            else:
                link = content.get("link") or item.get("link") or ""

            # Extract Publisher
            publisher = ""
            if isinstance(content.get("provider"), dict):
                publisher = content["provider"].get("displayName", "Yahoo Finance")
            else:
                publisher = content.get("publisher") or item.get("publisher") or "Yahoo Finance"

            # Extract Timestamp
            pub_time = ""
            pub_date_raw = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
            if isinstance(pub_date_raw, (int, float)):
                pub_time = datetime.datetime.fromtimestamp(pub_date_raw).strftime("%Y-%m-%d %H:%M")
            elif isinstance(pub_date_raw, str):
                pub_time = pub_date_raw[:16].replace("T", " ")

            _add_article(title, link, publisher, pub_time, "Yahoo Finance")
            if len(articles) >= max_items:
                break
    except Exception:
        pass

    # Source 2: Localized Google News RSS via feedparser
    if len(articles) < max_items:
        try:
            rss_url = build_google_news_rss_url(ticker, market_code)
            feed = feedparser.parse(rss_url)
            for entry in getattr(feed, "entries", []):
                raw_title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                publisher = "Google News RSS"
                if hasattr(entry, "source") and isinstance(entry.source, dict):
                    publisher = entry.source.get("title", publisher)
                elif " - " in raw_title:
                    parts = raw_title.rsplit(" - ", 1)
                    raw_title, publisher = parts[0], parts[1]

                pub_parsed = getattr(entry, "published_parsed", None)
                if pub_parsed:
                    dt = datetime.datetime(*pub_parsed[:6])
                    pub_time = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    pub_time = str(getattr(entry, "published", ""))[:16]

                _add_article(raw_title, link, publisher, pub_time, f"Google News ({market_code})")
                if len(articles) >= max_items:
                    break
        except Exception:
            pass

    return articles[:max_items]


def run_finbert_sentiment(
    ticker: str,
    market_code: str = "US",
    relay_mode: str = RELAY_MODES[0],
    max_items: int = 20,
    keep_on_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Fetch up to 20 deduplicated headlines and score them in batch on GPU using ProsusAI/finbert.
    When Smart GPU Relay is active (and keep_on_gpu is False), moves FinBERT back to System RAM ('cpu')
    and calls gc.collect() + torch.cuda.empty_cache() immediately after scoring.
    """
    articles = fetch_stock_news(ticker, market_code=market_code, max_items=max_items)
    if not articles:
        return {
            "net_sentiment": 0.0,
            "normalized_score": 50.0,
            "regime_label": "NEUTRAL (No Recent Headlines)",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "articles": [],
        }

    titles = [a["title"] for a in articles]
    use_cuda = bool(HAS_TORCH and torch.cuda.is_available() and "CPU Only" not in relay_mode)
    smart_relay = "Smart GPU Relay" in relay_mode and not keep_on_gpu

    scored_articles: List[Dict[str, Any]] = []
    net_scores: List[float] = []
    weights: List[float] = []
    bull_c, bear_c, neut_c = 0, 0, 0

    try:
        pipe = load_finbert_pipeline()
        if use_cuda:
            _move_finbert_device(pipe, "cuda:0")
        else:
            _move_finbert_device(pipe, "cpu")

        batch_results = pipe(titles, batch_size=16, truncation=True, max_length=128)

        if smart_relay and use_cuda:
            _move_finbert_device(pipe, "cpu")
            release_pytorch_vram()

        for art, res_list in zip(articles, batch_results):
            probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            if isinstance(res_list, dict):
                res_list = [res_list]
            for item in res_list:
                lbl = str(item.get("label", "")).lower()
                sc = float(item.get("score", 0.0))
                if lbl in probs:
                    probs[lbl] = sc

            p_pos = probs["positive"]
            p_neg = probs["negative"]
            p_neu = probs["neutral"]

            # Net article score in [-100, +100]
            art_net = (p_pos - p_neg) * 100.0
            top_label = max(probs, key=probs.get)
            top_conf = probs[top_label]

            if top_label == "positive" or art_net >= 15.0:
                badge = "🟢 Bullish"
                bull_c += 1
            elif top_label == "negative" or art_net <= -15.0:
                badge = "🔴 Bearish"
                bear_c += 1
            else:
                badge = "⚪ Neutral"
                neut_c += 1

            # Weight articles with stronger directional conviction higher
            w = max(0.25, p_pos + p_neg)
            net_scores.append(art_net)
            weights.append(w)

            scored_articles.append(
                {
                    "Headline": art["title"],
                    "Sentiment": badge,
                    "Net Score": round(art_net, 1),
                    "Confidence": f"{top_conf * 100:.1f}%",
                    "Publisher": art["publisher"],
                    "Published": art["published"],
                    "Source": art["source"],
                    "Link": art["link"],
                }
            )
    except Exception:
        # Lexical fallback if model load fails
        pos_words = {"beat", "surge", "upgrade", "buy", "bull", "record", "growth", "profit", "outperform", "rally", "gain", "strong"}
        neg_words = {"miss", "drop", "downgrade", "sell", "bear", "loss", "cut", "lawsuit", "crash", "weak", "warn", "plunge", "fall"}
        for art in articles:
            words = set(art["title"].lower().split())
            p_hits = len(words & pos_words)
            n_hits = len(words & neg_words)
            art_net = float(np.clip((p_hits - n_hits) * 35.0, -100.0, 100.0))
            if art_net > 10:
                badge = "🟢 Bullish"
                bull_c += 1
            elif art_net < -10:
                badge = "🔴 Bearish"
                bear_c += 1
            else:
                badge = "⚪ Neutral"
                neut_c += 1
            net_scores.append(art_net)
            weights.append(1.0)
            scored_articles.append(
                {
                    "Headline": art["title"],
                    "Sentiment": badge,
                    "Net Score": round(art_net, 1),
                    "Confidence": "75.0%",
                    "Publisher": art["publisher"],
                    "Published": art["published"],
                    "Source": art["source"],
                    "Link": art["link"],
                }
            )

    net_sentiment = float(np.average(net_scores, weights=weights)) if net_scores else 0.0
    net_sentiment = float(np.clip(net_sentiment, -100.0, 100.0))
    normalized_score = float(np.clip((net_sentiment + 100.0) / 2.0, 0.0, 100.0))

    if net_sentiment >= 35.0:
        regime = "STRONG BULLISH CATALYST FLOW"
    elif net_sentiment >= 12.0:
        regime = "MODERATELY BULLISH NEWS FLOW"
    elif net_sentiment <= -35.0:
        regime = "EXTREME BEARISH NEWS HEADWINDS"
    elif net_sentiment <= -12.0:
        regime = "MODERATELY BEARISH NEWS FLOW"
    else:
        regime = "NEUTRAL / BALANCED MEDIA FLOW"

    return {
        "net_sentiment": round(net_sentiment, 1),
        "normalized_score": round(normalized_score, 1),
        "regime_label": regime,
        "bullish_count": bull_c,
        "bearish_count": bear_c,
        "neutral_count": neut_c,
        "articles": scored_articles,
    }


# =====================================================================
# 4. CHRONOS-BOLT-SMALL PROBABILISTIC FORECASTING (TAB 1 & TAB 3)
# =====================================================================
def _build_future_index(last_ts: Any, horizon: int, interval: str) -> List[Any]:
    """Generate future timestamps corresponding to the chart interval."""
    try:
        base_dt = pd.to_datetime(last_ts)
    except Exception:
        base_dt = pd.Timestamp.now()

    if interval == "15m":
        delta = pd.Timedelta(minutes=15)
    elif interval == "1h":
        delta = pd.Timedelta(hours=1)
    elif interval == "1wk":
        delta = pd.Timedelta(weeks=1)
    else:
        delta = pd.Timedelta(days=1)

    future_dates = []
    curr = base_dt
    for _ in range(horizon):
        curr = curr + delta
        # Skip weekends on daily bars
        if interval == "1d":
            while curr.weekday() >= 5:
                curr += pd.Timedelta(days=1)
        future_dates.append(curr)
    return future_dates


def run_chronos_forecast(
    df: pd.DataFrame,
    horizon: int = 14,
    interval: str = "1d",
    relay_mode: str = RELAY_MODES[0],
    keep_on_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Run amazon/chronos-bolt-small on historical Close series to generate
    10th, 50th (median), and 90th percentile forecasts for the next N bars.
    Immediately offloads back to CPU and clears CUDA cache if Smart GPU Relay is active.
    Includes graceful statistical Monte Carlo ATR fallback if offline.
    """
    if df is None or df.empty:
        return {
            "future_dates": [],
            "p10": [],
            "p50": [],
            "p90": [],
            "end_p10": 0.0,
            "end_p50": 0.0,
            "end_p90": 0.0,
            "expected_return_pct": 0.0,
            "upside_pct": 0.0,
            "downside_pct": 0.0,
            "rr_skew": 1.0,
            "chronos_score": 50.0,
            "engine_used": "None",
        }

    close_series = df["Close"].dropna().astype("float32").values
    current_price = float(close_series[-1])
    future_dates = _build_future_index(df.index[-1], horizon, interval)

    use_cuda = bool(HAS_TORCH and torch.cuda.is_available() and "CPU Only" not in relay_mode)
    smart_relay = "Smart GPU Relay" in relay_mode and not keep_on_gpu

    p10_arr, p50_arr, p90_arr = None, None, None
    engine_used = "amazon/chronos-bolt-small (GPU)" if use_cuda else "amazon/chronos-bolt-small (CPU)"

    try:
        pipe = load_chronos_pipeline()
        if use_cuda:
            _move_chronos_device(pipe, "cuda:0")
        else:
            _move_chronos_device(pipe, "cpu")

        context_tensor = torch.tensor(close_series[-512:], dtype=torch.float32)
        quantiles, _ = pipe.predict_quantiles(
            context_tensor,
            prediction_length=int(horizon),
            quantile_levels=[0.1, 0.5, 0.9],
        )
        # quantiles shape: [batch=1, prediction_length, 3]
        q_np = quantiles[0].detach().cpu().numpy()
        p10_arr = q_np[:, 0].astype(float).tolist()
        p50_arr = q_np[:, 1].astype(float).tolist()
        p90_arr = q_np[:, 2].astype(float).tolist()

        if smart_relay and use_cuda:
            _move_chronos_device(pipe, "cpu")
            release_pytorch_vram()
    except Exception:
        # Graceful Statistical Monte Carlo ATR Projection Fallback
        engine_used = "Monte Carlo ATR Projection (Fallback)"
        returns = pd.Series(close_series).pct_change().dropna()
        mu = float(returns.tail(60).mean()) if len(returns) > 5 else 0.0005
        sigma = float(returns.tail(60).std()) if len(returns) > 5 else 0.015
        steps = np.arange(1, horizon + 1, dtype=float)
        drift = current_price * np.exp(mu * steps)
        spread = current_price * sigma * np.sqrt(steps) * 1.2816
        p50_arr = drift.tolist()
        p10_arr = np.maximum(0.01, drift - spread).tolist()
        p90_arr = (drift + spread).tolist()

    end_p10 = float(p10_arr[-1])
    end_p50 = float(p50_arr[-1])
    end_p90 = float(p90_arr[-1])

    exp_ret_pct = ((end_p50 - current_price) / current_price * 100.0) if current_price else 0.0
    upside_pct = ((end_p90 - current_price) / current_price * 100.0) if current_price else 0.0
    downside_pct = ((end_p10 - current_price) / current_price * 100.0) if current_price else 0.0

    upside_mag = max(0.0001, end_p90 - current_price)
    downside_mag = max(0.0001, current_price - end_p10)
    rr_skew = float(np.clip(upside_mag / downside_mag, 0.1, 10.0))

    # Convert Chronos Expected Return + Cone Skew into a 0-100 score
    # +5% median return -> +25 pts; rr_skew > 1.0 -> bonus pts
    ret_component = np.clip(exp_ret_pct * 4.5, -35.0, 35.0)
    skew_component = np.clip((rr_skew - 1.0) * 12.0, -15.0, 15.0)
    chronos_score = float(np.clip(50.0 + ret_component + skew_component, 0.0, 100.0))

    return {
        "future_dates": future_dates,
        "p10": p10_arr,
        "p50": p50_arr,
        "p90": p90_arr,
        "end_p10": round(end_p10, 2),
        "end_p50": round(end_p50, 2),
        "end_p90": round(end_p90, 2),
        "expected_return_pct": round(exp_ret_pct, 2),
        "upside_pct": round(upside_pct, 2),
        "downside_pct": round(downside_pct, 2),
        "rr_skew": round(rr_skew, 2),
        "chronos_score": round(chronos_score, 1),
        "engine_used": engine_used,
    }


# =====================================================================
# 5. MASTER CONVICTION SCORE & TRAP DETECTOR (TAB 3)
# =====================================================================
def compute_master_conviction(
    ta_summary: Dict[str, Any],
    finbert_summary: Dict[str, Any],
    chronos_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute Master Conviction Score (0 to 100) and Traffic-Light Verdict Badge combining:
      - 45% Enhanced Technical Confluence (0-100)
      - 30% FinBERT Institutional News Sentiment (normalized 0-100)
      - 25% Chronos-Bolt Expected Return & Risk/Reward Skew (0-100)
    Plus Automatic Trap Detection:
      - Flag 'TRAP WARNING' if Technicals look oversold/bullish but FinBERT News Sentiment < -25,
        or if Earnings are within 7 days.
    """
    ta_score = float(ta_summary.get("ta_score", 50.0))
    fb_norm = float(finbert_summary.get("normalized_score", 50.0))
    fb_net = float(finbert_summary.get("net_sentiment", 0.0))
    ch_score = float(chronos_summary.get("chronos_score", 50.0))

    raw_master = 0.45 * ta_score + 0.30 * fb_norm + 0.25 * ch_score

    # Check Automatic Trap Detection
    rsi_val = float(ta_summary.get("rsi_14", 50.0))
    earnings_warn = bool(ta_summary.get("earnings_warning", False))
    tech_looks_bullish_or_oversold = (ta_score >= 55.0) or (rsi_val <= 36.0) or bool(ta_summary.get("sma44_bounce", False))

    trap_reasons: List[str] = []
    if tech_looks_bullish_or_oversold and fb_net < -25.0:
        trap_reasons.append(
            f"Bullish/Oversold Technicals (TA {ta_score:.0f}/100, RSI {rsi_val:.1f}) collide with Strongly Negative FinBERT News Sentiment ({fb_net:+.1f}). Potential Value/Falling-Knife Trap!"
        )
    if earnings_warn:
        trap_reasons.append(
            f"Imminent Earnings Event Risk ({ta_summary.get('earnings_badge', 'Earnings <7d')}). Binary gap risk can invalidate standard ATR stops."
        )

    trap_warning = len(trap_reasons) > 0
    if trap_warning and fb_net < -25.0:
        raw_master = min(raw_master, 44.0)

    master_score = float(np.clip(raw_master, 0.0, 100.0))

    if trap_warning and fb_net < -25.0:
        verdict = "HIGH RISK / AVOID"
        badge_color = "#ff3b5c"
    elif master_score >= 74.0:
        verdict = "STRONG BUY"
        badge_color = "#00e676"
    elif master_score >= 60.0:
        verdict = "BULLISH"
        badge_color = "#26a69a"
    elif master_score >= 45.0:
        verdict = "NEUTRAL / HOLD"
        badge_color = "#ffca28"
    elif master_score >= 32.0:
        verdict = "BEARISH"
        badge_color = "#ff7043"
    else:
        verdict = "HIGH RISK / AVOID"
        badge_color = "#ff3b5c"

    # Math vs News Agreement check (used in Tab 3 & Tab 5 Calibration)
    math_bullish = (0.64 * ta_score + 0.36 * ch_score) >= 55.0
    news_bullish = fb_net >= 10.0
    math_bearish = (0.64 * ta_score + 0.36 * ch_score) <= 45.0
    news_bearish = fb_net <= -10.0
    aligned = (math_bullish and news_bullish) or (math_bearish and news_bearish)

    return {
        "master_score": round(master_score, 1),
        "ta_score": round(ta_score, 1),
        "finbert_norm_score": round(fb_norm, 1),
        "finbert_net_score": round(fb_net, 1),
        "chronos_score": round(ch_score, 1),
        "verdict": verdict,
        "badge_color": badge_color,
        "trap_warning": trap_warning,
        "trap_reasons": trap_reasons,
        "math_news_aligned": aligned,
    }


def calculate_atr_trade_plan(
    price: float,
    atr: float,
    support_20: float,
    sma_44: float,
    chronos_p50: float,
    chronos_p90: float,
    risk_amount: float = 200.0,
    direction: str = "LONG",
) -> Dict[str, float]:
    """
    Compute exact ATR-based institutional trade plan in active currency ($ or ₹):
      - Entry Price: Current Price
      - Stop-Loss: Entry - 1.5 * ATR_14 (for LONG) or Entry + 1.5 * ATR_14 (for SHORT)
      - Take-Profit 1: max(Chronos 50th, Entry + 2.0 * Risk_Per_Share) for LONG
      - Take-Profit 2: max(Chronos 90th, Entry + 3.0 * Risk_Per_Share) for LONG
      - Position Size (Shares): floor(Risk_Amount / Risk_Per_Share)
    """
    safe_price = max(0.01, float(price))
    safe_atr = max(safe_price * 0.008, float(atr)) if atr and atr > 0 else safe_price * 0.02
    risk_per_share = 1.5 * safe_atr

    if direction == "SHORT":
        stop_loss = safe_price + risk_per_share
        tp1_r = safe_price - 2.0 * risk_per_share
        tp2_r = safe_price - 3.0 * risk_per_share
        tp1 = min(float(chronos_p50) if chronos_p50 < safe_price else tp1_r, tp1_r)
        tp2 = min(float(chronos_p90) if chronos_p90 < tp1 else tp2_r, tp2_r)
    else:
        stop_loss = max(0.01, safe_price - risk_per_share)
        tp1_r = safe_price + 2.0 * risk_per_share
        tp2_r = safe_price + 3.0 * risk_per_share
        tp1 = max(float(chronos_p50) if chronos_p50 > safe_price else tp1_r, tp1_r)
        tp2 = max(float(chronos_p90) if chronos_p90 > tp1 else tp2_r, tp2_r)

    shares = max(1, int(risk_amount // max(0.01, risk_per_share)))
    position_value = shares * safe_price

    return {
        "entry_price": round(safe_price, 2),
        "atr": round(safe_atr, 2),
        "risk_per_share": round(risk_per_share, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "shares": int(shares),
        "position_value": round(position_value, 2),
        "risk_amount": round(float(risk_amount), 2),
    }


# =====================================================================
# 6. OLLAMA QWEN 2.5 14B STREAMING ANALYST BRIEFING & CHAT (TAB 3)
# =====================================================================
def build_stock_context_payload(
    ticker: str,
    market_cfg: Dict[str, Any],
    interval: str,
    horizon: int,
    ta_summary: Dict[str, Any],
    finbert_summary: Dict[str, Any],
    chronos_summary: Dict[str, Any],
    conviction: Dict[str, Any],
    trade_plan: Dict[str, float],
) -> Dict[str, Any]:
    """Assemble structured JSON context payload for qwen2.5:14b."""
    top_headlines = [
        {
            "headline": a["Headline"],
            "sentiment": a["Sentiment"],
            "score": a["Net Score"],
        }
        for a in finbert_summary.get("articles", [])[:6]
    ]
    return {
        "ticker": ticker,
        "market_mode": market_cfg["market_label"],
        "market_code": market_cfg["market_code"],
        "currency_symbol": market_cfg["currency_symbol"],
        "benchmark": market_cfg["benchmark_ticker"],
        "timeframe_interval": interval,
        "forecast_horizon_bars": horizon,
        "current_price": ta_summary.get("current_price"),
        "pct_change": round(ta_summary.get("pct_change", 0.0), 2),
        "master_conviction_score": conviction.get("master_score"),
        "verdict_badge": conviction.get("verdict"),
        "trap_warning": conviction.get("trap_warning"),
        "trap_reasons": conviction.get("trap_reasons"),
        "technicals": {
            "ta_confluence_score": ta_summary.get("ta_score"),
            "sma_20": round(ta_summary.get("sma20_val", 0.0), 2),
            "sma_44": round(ta_summary.get("sma44_val", 0.0), 2),
            "sma_44_status": ta_summary.get("sma44_status"),
            "sma_44_rising": ta_summary.get("sma44_rising"),
            "sma_44_bounce": ta_summary.get("sma44_bounce"),
            "ema_50": round(ta_summary.get("ema50_val", 0.0), 2),
            "sma_200": round(ta_summary.get("sma200_val", 0.0), 2),
            "rsi_14": round(ta_summary.get("rsi_14", 50.0), 1),
            "macd_hist": round(ta_summary.get("macd_hist", 0.0), 3),
            "adx_14": round(ta_summary.get("adx_14", 20.0), 1),
            "atr_14": round(ta_summary.get("atr", 0.0), 2),
            "rvol": round(ta_summary.get("rvol", 1.0), 2),
            "vwap": round(ta_summary.get("vwap", 0.0), 2),
            "support_20": round(ta_summary.get("support_20", 0.0), 2),
            "resistance_20": round(ta_summary.get("resistance_20", 0.0), 2),
            "candlestick_patterns": ta_summary.get("active_patterns"),
            "relative_strength_20b_vs_benchmark_pct": round(ta_summary.get("rs_spread_20", 0.0), 2),
            "earnings_status": ta_summary.get("earnings_badge"),
        },
        "finbert_sentiment": {
            "net_sentiment_minus100_to_plus100": finbert_summary.get("net_sentiment"),
            "normalized_0_to_100": finbert_summary.get("normalized_score"),
            "regime": finbert_summary.get("regime_label"),
            "top_headlines": top_headlines,
        },
        "chronos_forecast": {
            "horizon_bars": horizon,
            "p10_bear_case": chronos_summary.get("end_p10"),
            "p50_base_case": chronos_summary.get("end_p50"),
            "p90_bull_case": chronos_summary.get("end_p90"),
            "expected_median_return_pct": chronos_summary.get("expected_return_pct"),
            "upside_downside_skew": chronos_summary.get("rr_skew"),
        },
        "atr_trade_plan": trade_plan,
    }


def stream_qwen_trading_briefing(
    selected_model: str,
    payload: Dict[str, Any],
) -> Generator[str, None, None]:
    """
    Step 2 of Smart GPU Relay:
    Ensure PyTorch VRAM is cleared, then stream a real-time institutional report from Ollama (qwen2.5:14b)
    with options={"num_ctx": 4096, "num_batch": 512, "temperature": 0.2} and keep_alive="5m".
    """
    release_pytorch_vram()
    curr = payload.get("currency_symbol", "$")
    ticker = payload.get("ticker", "NVDA")

    system_prompt = (
        "You are a Managing Director & Quantitative Swing Strategist at an institutional dual-market "
        "(Wall Street NYSE/NASDAQ & Mumbai NSE/BSE) trading desk. "
        f"Always quote every price, stop-loss, target, and position size strictly in the active currency symbol ({curr}). "
        "Be precise, data-driven, and structured."
    )

    user_prompt = f"""Analyze the following live multi-modal telemetry JSON for **{ticker}** and write an institutional Trading Desk Briefing with these exact 3 sections:

### 1. Executive Verdict & Bull vs. Bear Thesis
- State the Master Conviction Score ({payload.get('master_conviction_score')}/100) and Verdict ({payload.get('verdict_badge')}).
- Summarize the core Bull Case vs. Bear Case using the exact Moving Averages, RSI, ADX, and Chronos-Bolt {payload.get('forecast_horizon_bars')}-bar probabilistic cone targets ({curr}{payload['chronos_forecast']['p10_bear_case']} / {curr}{payload['chronos_forecast']['p50_base_case']} / {curr}{payload['chronos_forecast']['p90_bull_case']}).

### 2. Math vs. News & Volume Divergence Check
- Explicitly evaluate the **Rising 44-SMA Swing Structure** (`{payload['technicals']['sma_44_status']}`, SMA_44 at {curr}{payload['technicals']['sma_44']}), **RVOL institutional participation** (`{payload['technicals']['rvol']}x` avg volume), **Relative Strength vs. {payload.get('benchmark')}** (`{payload['technicals']['relative_strength_20b_vs_benchmark_pct']:+.2f}%`), and **FinBERT News Sentiment** (`{payload['finbert_sentiment']['net_sentiment_minus100_to_plus100']:+.1f}` / 100).
- Highlight whether technical price action and news catalysts agree or diverge, and note any Trap or Earnings warnings.

### 3. Actionable ATR Trade Plan (in `{curr}`)
- **Exact Entry Zone**: around {curr}{payload['atr_trade_plan']['entry_price']}
- **Stop-Loss (1.5x ATR / Swing Structure)**: {curr}{payload['atr_trade_plan']['stop_loss']} (ATR_14 = {curr}{payload['technicals']['atr_14']}, 20-bar Support = {curr}{payload['technicals']['support_20']})
- **Take-Profit 1 (Chronos 50th / +2R)**: {curr}{payload['atr_trade_plan']['take_profit_1']}
- **Take-Profit 2 (Chronos 90th / +3R)**: {curr}{payload['atr_trade_plan']['take_profit_2']}
- **Position Sizing**: Recommended **{payload['atr_trade_plan']['shares']} shares** (Total Notional: {curr}{payload['atr_trade_plan']['position_value']:,.2f}) for a max defined risk of **{curr}{payload['atr_trade_plan']['risk_amount']:,.2f}**.

TELEMETRY JSON:
```json
{json.dumps(payload, indent=2)}
```"""

    if HAS_OLLAMA:
        try:
            stream = ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
        except Exception as exc:
            yield f"\n> ⚠️ *Ollama local server warning ({exc}). Rendering quantitative desk synthesis directly from engine telemetry:*\n\n"

    # Fallback deterministic synthesis if Ollama service is unreachable
    tp = payload["atr_trade_plan"]
    tech = payload["technicals"]
    fb = payload["finbert_sentiment"]
    ch = payload["chronos_forecast"]
    yield (
        f"### 1. Executive Verdict & Bull vs. Bear Thesis\n"
        f"- **Master Conviction:** **{payload.get('master_conviction_score')}/100 ({payload.get('verdict_badge')})** on `{ticker}` ({payload.get('timeframe_interval')} timeframe).\n"
        f"- **Bull Thesis:** Price ({curr}{ payload.get('current_price'):,.2f}) vs SMA_44 ({curr}{tech['sma_44']:,.2f}, `{tech['sma_44_status']}`), "
        f"Chronos {ch['horizon_bars']}-bar Median Target at **{curr}{ch['p50_base_case']:,.2f}** ({ch['expected_median_return_pct']:+.2f}%) and 90th percentile bull target at **{curr}{ch['p90_bull_case']:,.2f}**.\n"
        f"- **Bear Thesis:** Downside 10th percentile risk cone sits at **{curr}{ch['p10_bear_case']:,.2f}** with 20-bar structural support at **{curr}{tech['support_20']:,.2f}**.\n\n"
        f"### 2. Math vs. News & Volume Divergence Check\n"
        f"- **44-SMA & RVOL Structure:** SMA_44 status is `{tech['sma_44_status']}` with Relative Volume at **{tech['rvol']}x** 20-bar average and Relative Strength of **{tech['relative_strength_20b_vs_benchmark_pct']:+.2f}%** vs `{payload.get('benchmark')}`.\n"
        f"- **FinBERT News Sentiment:** Net institutional sentiment is **{fb['net_sentiment_minus100_to_plus100']:+.1f}** ({fb['regime']}).\n\n"
        f"### 3. Actionable ATR Trade Plan (in `{curr}`)\n"
        f"- **Entry Zone:** `{curr}{tp['entry_price']:,.2f}`\n"
        f"- **Stop-Loss (1.5x ATR):** `{curr}{tp['stop_loss']:,.2f}` (ATR_14 = `{curr}{tp['atr']:,.2f}`)\n"
        f"- **Take-Profit 1 (Chronos P50 / +2R):** `{curr}{tp['take_profit_1']:,.2f}`\n"
        f"- **Take-Profit 2 (Chronos P90 / +3R):** `{curr}{tp['take_profit_2']:,.2f}`\n"
        f"- **Position Sizing:** **{tp['shares']} shares** (Notional `{curr}{tp['position_value']:,.2f}`) risking `{curr}{tp['risk_amount']:,.2f}`.\n"
    )


def stream_qwen_stock_chat(
    selected_model: str,
    payload: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    user_question: str,
) -> Generator[str, None, None]:
    """
    Stream interactive follow-up answers from Ollama qwen2.5:14b with full stock context preserved.
    """
    release_pytorch_vram()
    curr = payload.get("currency_symbol", "$")
    ticker = payload.get("ticker", "NVDA")

    sys_msg = (
        f"You are an institutional quantitative trading assistant analyzing {ticker} ({payload.get('market_mode')}). "
        f"Always use currency symbol {curr}. Reference this live telemetry context in your answers:\n"
        f"{json.dumps(payload)}"
    )
    messages = [{"role": "system", "content": sys_msg}]
    for m in chat_history[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_question})

    if HAS_OLLAMA:
        try:
            stream = ollama.chat(
                model=selected_model,
                messages=messages,
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
        except Exception as exc:
            yield f"*(Ollama connection error: {exc})* Based on `{ticker}` telemetry: Current Price is {curr}{payload.get('current_price')}, TA Score is {payload['technicals']['ta_confluence_score']}/100, FinBERT Sentiment is {payload['finbert_sentiment']['net_sentiment_minus100_to_plus100']:+.1f}, and Chronos P50 target is {curr}{payload['chronos_forecast']['p50_base_case']}."
    else:
        yield f"Based on `{ticker}` telemetry: Current Price is {curr}{payload.get('current_price')}, TA Score is {payload['technicals']['ta_confluence_score']}/100, and Chronos P50 target is {curr}{payload['chronos_forecast']['p50_base_case']}."
