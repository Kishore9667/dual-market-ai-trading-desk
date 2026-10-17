"""
Dual-Market (US NYSE/NASDAQ & India NSE/BSE) GPU-Accelerated Institutional Stock Intelligence,
Trusted Internet Stock Discovery (Blue-Chip Movers + Valuable Mid-Cap GARP Underdogs),
Deep Learning Forecasting (Chronos-Bolt + FinBERT + Qwen 2.5 14B)
& Multi-Currency SQLite Paper-Trading Audit Dashboard.
Optimized for Windows + NVIDIA RTX 3050 (8GB VRAM) + 32GB System RAM.
"""

import os

# Set Ollama Flash Attention before any other imports to optimize RTX 3050 8GB VRAM usage
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"

import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from market_config import (
    MARKET_OPTIONS,
    INTERVAL_LOOKBACK_MAP,
    get_market_config,
    normalize_ticker,
    format_currency,
)
from ta_engine import (
    HAS_TALIB,
    fetch_ohlcv,
    compute_indicators,
    compute_benchmark_relative_strength,
    check_upcoming_earnings,
    evaluate_technical_confluence,
)
from gpu_relay_engine import (
    RELAY_MODES,
    get_hardware_telemetry,
    get_ollama_models,
    run_finbert_sentiment,
    run_chronos_forecast,
    compute_master_conviction,
    calculate_atr_trade_plan,
    build_stock_context_payload,
    stream_qwen_trading_briefing,
    stream_qwen_stock_chat,
)
from discovery_scanner import (
    run_institutional_discovery_scan,
    stream_top3_discovery_summary,
    run_valuable_underdog_scan,
    stream_underdog_qwen_thesis,
)
from db_ledger import (
    init_db,
    add_to_saved_watchlist,
    remove_from_saved_watchlist,
    get_saved_watchlist,
    get_watchlist_live_quote,
    log_paper_trade,
    get_trades_df,
    manual_close_trade,
    reset_market_portfolio,
    sync_and_audit_open_trades,
    compute_reliability_metrics,
)
from charts_ui import (
    inject_institutional_css,
    render_kpi_card,
    build_multi_panel_technical_chart,
    build_sentiment_gauge,
    build_equity_curve_chart,
    build_calibration_chart,
)

# =====================================================================
# PAGE CONFIG & INITIALIZATION
# =====================================================================
st.set_page_config(
    page_title="Institutional Dual-Market AI Trading Desk (US & India)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_institutional_css()
init_db()

# Session state defaults
if "active_market_label" not in st.session_state:
    st.session_state["active_market_label"] = MARKET_OPTIONS[0]
if "indian_suffix" not in st.session_state:
    st.session_state["indian_suffix"] = ".NS"
if "custom_ticker_us" not in st.session_state:
    st.session_state["custom_ticker_us"] = "NVDA"
if "custom_ticker_in" not in st.session_state:
    st.session_state["custom_ticker_in"] = "RELIANCE.NS"
if "briefing_cache" not in st.session_state:
    st.session_state["briefing_cache"] = {}
if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}
if "discovery_results" not in st.session_state:
    st.session_state["discovery_results"] = {}
if "discovery_summary" not in st.session_state:
    st.session_state["discovery_summary"] = {}
if "underdog_results" not in st.session_state:
    st.session_state["underdog_results"] = {}
if "underdog_thesis" not in st.session_state:
    st.session_state["underdog_thesis"] = {}
if "prefill_trade" not in st.session_state:
    st.session_state["prefill_trade"] = None


# =====================================================================
# SIDEBAR: DUAL-MARKET SELECTOR, WATCHLISTS & HARDWARE TELEMETRY
# =====================================================================
with st.sidebar:
    st.markdown("## 🏛️ Institutional AI Desk")
    st.caption("100% Local GPU • US (NYSE/NASDAQ) & India (NSE/BSE)")

    # 1. Top-Level Market Selector
    selected_market_label = st.selectbox(
        "🌍 Active Market Regime",
        options=MARKET_OPTIONS,
        index=0 if st.session_state["active_market_label"] == MARKET_OPTIONS[0] else 1,
        key="market_selector_widget",
    )
    st.session_state["active_market_label"] = selected_market_label
    is_india_mode = "Indian" in selected_market_label

    if is_india_mode:
        indian_suffix = st.radio(
            "🇮🇳 Preferred Indian Exchange Suffix",
            options=[".NS", ".BO"],
            format_func=lambda x: ".NS (NSE - National Stock Exchange)" if x == ".NS" else ".BO (BSE - Bombay Stock Exchange)",
            horizontal=True,
            index=0 if st.session_state.get("indian_suffix", ".NS") == ".NS" else 1,
        )
        st.session_state["indian_suffix"] = indian_suffix
    else:
        indian_suffix = ".NS"

    market_cfg = get_market_config(selected_market_label, indian_suffix=indian_suffix)
    m_code = market_cfg["market_code"]
    curr = market_cfg["currency_symbol"]
    bench = market_cfg["benchmark_ticker"]

    st.divider()

    # 2. Core Market Watchlist + Persistent Saved Underdog Watchlist + Smart Ticker Input
    st.markdown(f"### 🎯 {m_code} Ticker Selection")
    watchlist = market_cfg["watchlist"]
    active_key = "custom_ticker_in" if m_code == "IN" else "custom_ticker_us"

    def _on_watchlist_change():
        chosen = st.session_state.get(f"wl_select_{m_code}")
        if chosen:
            st.session_state[active_key] = chosen

    default_wl_idx = (
        watchlist.index(st.session_state[active_key])
        if st.session_state[active_key] in watchlist
        else 0
    )
    st.selectbox(
        f"Core {m_code} Market Watchlist",
        options=watchlist,
        index=default_wl_idx,
        key=f"wl_select_{m_code}",
        on_change=_on_watchlist_change,
    )

    # Persistent SQLite Saved Underdog Watchlist Selector
    saved_wl_rows = get_saved_watchlist(m_code)
    if saved_wl_rows:
        saved_tickers = [r["ticker"] for r in saved_wl_rows]
        saved_labels_map: Dict[str, str] = {}
        for r in saved_wl_rows:
            t_sym = r["ticker"]
            add_p = float(r.get("price_at_add") or 0.0)
            _, pct_gain = get_watchlist_live_quote(t_sym, add_p)
            cat_tag = "🚀 Gem" if "Underdog" in str(r.get("category", "")) else "🏛️ Blue-Chip"
            saved_labels_map[t_sym] = (
                f"{t_sym} ({cat_tag} • {pct_gain:+.1f}% since {curr}{add_p:,.2f})"
            )

        def _on_saved_wl_change():
            chosen_saved = st.session_state.get(f"saved_wl_select_{m_code}")
            if chosen_saved and chosen_saved != "— Select Saved Gem —":
                st.session_state[active_key] = chosen_saved

        saved_options = ["— Select Saved Gem —"] + saved_tickers
        current_saved_idx = (
            saved_options.index(st.session_state[active_key])
            if st.session_state[active_key] in saved_options
            else 0
        )
        chosen_saved_sym = st.selectbox(
            f"🚀 Saved Underdog Watchlist ({len(saved_wl_rows)})",
            options=saved_options,
            index=current_saved_idx,
            format_func=lambda x: x if x == "— Select Saved Gem —" else saved_labels_map.get(x, x),
            key=f"saved_wl_select_{m_code}",
            on_change=_on_saved_wl_change,
        )
        rem_target = chosen_saved_sym if chosen_saved_sym != "— Select Saved Gem —" else saved_tickers[0]
        if st.button(f"🗑️ Remove {rem_target} from Watchlist", use_container_width=True, key=f"rem_wl_{m_code}"):
            remove_from_saved_watchlist(rem_target)
            st.rerun()
    else:
        st.caption("🚀 *Saved Underdog Watchlist is empty — add gems from Tab 4B!*")

    raw_ticker_input = st.text_input(
        "Ticker Symbol (Auto-Normalized)" if m_code == "IN" else "Ticker Symbol",
        value=st.session_state[active_key],
        help="For India, typing TCS or RELIANCE automatically normalizes to TCS.NS / RELIANCE.NS.",
    )
    active_ticker = normalize_ticker(raw_ticker_input, m_code, indian_suffix=indian_suffix)
    st.session_state[active_key] = active_ticker
    if m_code == "IN" and raw_ticker_input.strip().upper() != active_ticker:
        st.caption(f"✅ Normalized `{raw_ticker_input.strip().upper()}` → **`{active_ticker}`**")

    # 3. Chart Interval & Automatic Yahoo Finance Lookback Caps
    col_int, col_per = st.columns(2)
    with col_int:
        interval = st.selectbox(
            "⏱️ Interval",
            options=["15m", "1h", "1d", "1wk"],
            index=2,
        )
    lookback_cfg = INTERVAL_LOOKBACK_MAP[interval]
    with col_per:
        period_options = lookback_cfg["options"]
        default_per_idx = period_options.index(lookback_cfg["default"])
        period = st.selectbox(
            "📅 Lookback",
            options=period_options,
            index=default_per_idx,
            help=f"Automatically capped at {lookback_cfg['max_days']}d for {interval} bars so Yahoo Finance never fails.",
        )

    prepost = st.toggle(
        "🕒 Include Pre/Post Market Hours",
        value=False,
        disabled=not market_cfg["supports_prepost"],
        help="Supported on US equities; automatically disabled for NSE/BSE.",
    )

    horizon = st.slider(
        "🔮 Chronos Forecast Horizon (Bars)",
        min_value=10,
        max_value=30,
        value=14,
        step=1,
    )

    risk_per_trade = st.number_input(
        f"🛡️ Max Risk Per Trade ({curr})",
        min_value=10.0,
        max_value=1000000.0,
        value=float(market_cfg["default_risk"]),
        step=50.0 if m_code == "US" else 1000.0,
    )

    st.divider()

    # 4. Ollama LLM & Smart GPU Relay Configuration
    st.markdown("### ⚡ Smart GPU Relay & LLM")
    ollama_online, ollama_models, default_model_idx = get_ollama_models()
    if ollama_online:
        selected_llm = st.selectbox(
            "🤖 Local Ollama Analyst Model",
            options=ollama_models,
            index=default_model_idx,
        )
    else:
        st.warning("Ollama server offline at localhost:11434 — using fallback input.")
        selected_llm = st.text_input("🤖 Ollama Model Name", value="qwen2.5:14b")

    relay_mode = st.selectbox(
        "🔄 32GB RAM + 8GB VRAM Relay Mode",
        options=RELAY_MODES,
        index=0,
        help="Moves FinBERT & Chronos-Bolt to RTX 3050 CUDA for <1s inference, then offloads to 32GB System RAM and clears CUDA cache before Qwen 2.5 14B runs.",
    )

    # 5. Live Hardware Telemetry Card
    hw = get_hardware_telemetry()
    ta_backend_str = "TA-Lib C-Ext" if HAS_TALIB else "Vectorized NumPy/Pandas"
    st.markdown(
        f"""
        <div class="telemetry-box">
            <div style="font-weight:700;color:#00e676;margin-bottom:4px;">🖥️ LIVE HARDWARE TELEMETRY</div>
            <div><b>GPU:</b> {hw['gpu_name']}</div>
            <div><b>VRAM:</b> {hw['vram_used_gb']:.2f} GB used / {hw['vram_free_gb']:.2f} GB free ({hw['vram_total_gb']:.1f} GB)</div>
            <div><b>System RAM:</b> {hw['ram_used_gb']:.1f} / {hw['ram_total_gb']:.1f} GB ({hw['ram_percent']:.0f}%)</div>
            <div><b>Ollama Flash Attn:</b> Enabled (<code>1</code>)</div>
            <div><b>TA Engine:</b> {ta_backend_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# CORE DATA PIPELINE EXECUTION (STEP 1 OF SMART GPU RELAY)
# =====================================================================
@st.cache_data(ttl=180, show_spinner=False)
def _cached_technical_bundle(
    ticker: str,
    benchmark: str,
    interval: str,
    period: str,
    prepost: bool,
    currency_symbol: str,
) -> Dict[str, Any]:
    df_raw = fetch_ohlcv(ticker, interval=interval, period=period, prepost=prepost)
    if df_raw.empty:
        return {"df": pd.DataFrame(), "rs": {}, "earnings": {}, "ta_summary": {}}
    df_ind = compute_indicators(df_raw)
    rs_data = compute_benchmark_relative_strength(df_ind, benchmark_ticker=benchmark, interval=interval, period=period)
    earn_data = check_upcoming_earnings(ticker)
    ta_sum = evaluate_technical_confluence(df_ind, rs_data, earn_data, currency_symbol=currency_symbol)
    return {
        "df": df_ind,
        "rs": rs_data,
        "earnings": earn_data,
        "ta_summary": ta_sum,
    }


def load_and_compute_technical_bundle(
    ticker: str,
    benchmark: str,
    interval: str,
    period: str,
    prepost: bool,
    currency_symbol: str,
) -> Dict[str, Any]:
    """Load technical bundle and automatically evict + retry if an empty DataFrame was ever cached."""
    bundle = _cached_technical_bundle(
        ticker, benchmark, interval, period, prepost, currency_symbol
    )
    if bundle["df"].empty:
        _cached_technical_bundle.clear()
        bundle = _cached_technical_bundle(
            ticker, benchmark, interval, period, prepost, currency_symbol
        )
        if bundle["df"].empty:
            _cached_technical_bundle.clear()
    return bundle


with st.spinner(f"Running Smart GPU Relay Pipeline on {active_ticker} ({m_code} • {interval})..."):
    tech_bundle = load_and_compute_technical_bundle(
        active_ticker, bench, interval, period, prepost, curr
    )
    df_ind = tech_bundle["df"]

    if not df_ind.empty:
        ta_summary = tech_bundle["ta_summary"]
        finbert_summary = run_finbert_sentiment(
            active_ticker, market_code=m_code, relay_mode=relay_mode, max_items=20
        )
        chronos_summary = run_chronos_forecast(
            df_ind, horizon=horizon, interval=interval, relay_mode=relay_mode
        )
        conviction = compute_master_conviction(ta_summary, finbert_summary, chronos_summary)
        trade_plan = calculate_atr_trade_plan(
            price=ta_summary["current_price"],
            atr=ta_summary["atr"],
            support_20=ta_summary["support_20"],
            sma_44=ta_summary["sma44_val"],
            chronos_p50=chronos_summary["end_p50"],
            chronos_p90=chronos_summary["end_p90"],
            risk_amount=risk_per_trade,
        )
        stock_payload = build_stock_context_payload(
            ticker=active_ticker,
            market_cfg=market_cfg,
            interval=interval,
            horizon=horizon,
            ta_summary=ta_summary,
            finbert_summary=finbert_summary,
            chronos_summary=chronos_summary,
            conviction=conviction,
            trade_plan=trade_plan,
        )
    else:
        ta_summary, finbert_summary, chronos_summary, conviction, trade_plan, stock_payload = {}, {}, {}, {}, {}, {}


# =====================================================================
# HEADER BANNER
# =====================================================================
st.markdown(
    f"### 🏛️ {market_cfg['market_label']} — `{active_ticker}` Institutional Intelligence & Deep Learning Desk"
)

if df_ind.empty:
    st.error(
        f"Could not retrieve historical OHLCV data for `{active_ticker}` ({interval}, {period}). "
        "Please verify the ticker symbol or click Retry below."
    )
    rc1, rc2, _ = st.columns([1, 1, 2])
    with rc1:
        if st.button("🔄 Clear Cache & Retry Fetch", type="primary", use_container_width=True):
            _cached_technical_bundle.clear()
            st.cache_data.clear()
            st.rerun()
    with rc2:
        default_sym = market_cfg["watchlist"][0]
        if st.button(f"🎯 Reset to {default_sym}", use_container_width=True):
            _cached_technical_bundle.clear()
            st.session_state[active_key] = default_sym
            st.rerun()
    st.stop()



# =====================================================================
# MAIN 5-TAB INSTITUTIONAL WORKSPACE
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Tab 1: Technicals & Swing Structure",
        "📰 Tab 2: FinBERT Institutional News",
        "🧠 Tab 3: Chronos-Bolt, Qwen 14B & Trade Plan",
        "🔍 Tab 4: Trusted Stock Discovery",
        "🛡️ Tab 5: Paper Trading & Reliability Auditor",
    ]
)


# =====================================================================
# TAB 1: TECHNICAL ANALYSIS & SWING STRUCTURE (PLOTLY MULTI-PANEL)
# =====================================================================
with tab1:
    if ta_summary.get("earnings_warning"):
        st.markdown(
            f'<div class="earnings-alert">{ta_summary["earnings_badge"]} — Elevated binary gap risk! Tighten stops or reduce position size before earnings release.</div>',
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    price_val = ta_summary["current_price"]
    pct_chg = ta_summary["pct_change"]
    with k1:
        render_kpi_card(
            f"Current Price ({curr})",
            f"{curr}{price_val:,.2f}",
            f"{pct_chg:+.2f}% ({interval})",
            "#00e676" if pct_chg >= 0 else "#ff3b5c",
        )
    with k2:
        ta_sc = ta_summary["ta_score"]
        render_kpi_card(
            "Tech Confluence",
            f"{ta_sc:.1f} / 100",
            "Bullish Structure" if ta_sc >= 58 else ("Bearish Structure" if ta_sc <= 42 else "Neutral / Mixed"),
            "#00e676" if ta_sc >= 58 else ("#ff3b5c" if ta_sc <= 42 else "#ffca28"),
        )
    with k3:
        render_kpi_card(
            "44-SMA Trend Status",
            f"{curr}{ta_summary['sma44_val']:,.2f}",
            ta_summary["sma44_status"],
            "#ffd740" if ta_summary["sma44_bounce"] else ("#00e676" if ta_summary["sma44_rising"] else "#ff7043"),
        )
    with k4:
        rv = ta_summary["rvol"]
        render_kpi_card(
            "Relative Vol (RVOL)",
            f"{rv:.2f}x Avg",
            "🔥 High Institutional Flow" if rv >= 1.5 else ("Normal Participation" if rv >= 0.9 else "Low Volume"),
            "#00e676" if rv >= 1.5 else "#cbd5e1",
        )
    with k5:
        rs_sp = ta_summary["rs_spread_20"]
        render_kpi_card(
            f"Rel Strength vs {bench}",
            f"{rs_sp:+.2f}%",
            f"Outperforming {bench}" if rs_sp >= 0 else f"Lagging {bench}",
            "#00e676" if rs_sp >= 0 else "#ff3b5c",
        )
    with k6:
        render_kpi_card(
            f"ATR(14) & Earnings",
            f"{curr}{ta_summary['atr']:,.2f}",
            ta_summary["earnings_badge"],
            "#ffb300" if ta_summary["earnings_warning"] else "#cbd5e1",
        )

    fig_ta = build_multi_panel_technical_chart(
        df=df_ind,
        ticker=active_ticker,
        interval=interval,
        currency_symbol=curr,
        chronos_data=chronos_summary,
    )
    st.plotly_chart(fig_ta, use_container_width=True)

    st.markdown("#### 📋 Institutional Technical Signal Breakdown")
    bd_df = pd.DataFrame(ta_summary["breakdown_rows"])

    def _color_status(val: str) -> str:
        if val == "Bullish":
            return "color: #00e676; font-weight: 700;"
        if val == "Bearish":
            return "color: #ff3b5c; font-weight: 700;"
        return "color: #ffca28; font-weight: 600;"

    st.dataframe(
        bd_df.style.map(_color_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# TAB 2: FINBERT GPU INSTITUTIONAL NEWS SENTIMENT (ProsusAI/finbert)
# =====================================================================
with tab2:
    st.markdown(f"#### 📰 FinBERT GPU Institutional News Sentiment — `{active_ticker}`")
    st.caption(
        f"Scored via `ProsusAI/finbert` across deduplicated Yahoo Finance + Localized Google News RSS (`gl={'IN' if m_code == 'IN' else 'US'}`). "
        "VRAM automatically reclaimed via Smart GPU Relay."
    )

    g_col, m_col = st.columns([1.15, 1.85])
    with g_col:
        fig_gauge = build_sentiment_gauge(
            finbert_summary["net_sentiment"], finbert_summary["regime_label"]
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with m_col:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_kpi_card(
                "Net Sentiment",
                f"{finbert_summary['net_sentiment']:+.1f}",
                "Range: -100 to +100",
                "#00e676" if finbert_summary["net_sentiment"] >= 10 else ("#ff3b5c" if finbert_summary["net_sentiment"] <= -10 else "#ffca28"),
            )
        with s2:
            render_kpi_card(
                "Normalized Score",
                f"{finbert_summary['normalized_score']:.1f} / 100",
                "30% Weight in Conviction",
                "#38bdf8",
            )
        with s3:
            render_kpi_card(
                "🟢 Bullish Articles",
                str(finbert_summary["bullish_count"]),
                f"⚪ Neutral: {finbert_summary['neutral_count']}",
                "#00e676",
            )
        with s4:
            render_kpi_card(
                "🔴 Bearish Articles",
                str(finbert_summary["bearish_count"]),
                f"Total Scored: {len(finbert_summary['articles'])}",
                "#ff3b5c",
            )

        st.info(
            f"**Institutional Media Regime:** `{finbert_summary['regime_label']}` — "
            f"Combined {len(finbert_summary['articles'])} deduplicated headlines from `yf.Ticker('{active_ticker}').news` "
            f"and localized Google News RSS (`{m_code}`)."
        )

    articles_list = finbert_summary.get("articles", [])
    if articles_list:
        news_df = pd.DataFrame(articles_list)
        st.dataframe(
            news_df,
            column_config={
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "Sentiment": st.column_config.TextColumn("FinBERT Verdict", width="small"),
                "Net Score": st.column_config.NumberColumn("Net Score (-100..+100)", format="%+.1f"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Publisher": st.column_config.TextColumn("Publisher"),
                "Published": st.column_config.TextColumn("Timestamp"),
                "Source": st.column_config.TextColumn("Feed"),
                "Link": st.column_config.LinkColumn("Article URL", display_text="Open Article ↗"),
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No recent news headlines found for this ticker.")


# =====================================================================
# TAB 3: CHRONOS-BOLT FORECAST, QWEN 2.5 14B ANALYST & 1-CLICK PAPER TRADE
# =====================================================================
with tab3:
    v_color = conviction["badge_color"]
    st.markdown(
        f"""
        <div class="verdict-banner" style="border-left: 6px solid {v_color};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                <div>
                    <span style="font-size:0.82rem;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.08em;">
                        MULTI-MODAL MASTER CONVICTION SCORE (45% TA • 30% FINBERT • 25% CHRONOS-BOLT)
                    </span>
                    <div style="font-size:2.0rem;font-weight:800;color:#ffffff;margin-top:2px;">
                        {conviction['master_score']:.1f} / 100
                        <span style="background:{v_color};color:#0b0f19;padding:4px 14px;border-radius:6px;font-size:1.05rem;margin-left:12px;vertical-align:middle;">
                            {conviction['verdict']}
                        </span>
                    </div>
                </div>
                <div style="text-align:right;font-size:0.88rem;color:#e2e8f0;">
                    <div><b>Enhanced TA (45%):</b> {conviction['ta_score']:.1f}/100 | <b>FinBERT News (30%):</b> {conviction['finbert_norm_score']:.1f}/100</div>
                    <div><b>Chronos-Bolt (25%):</b> {conviction['chronos_score']:.1f}/100 | <b>Forecast Engine:</b> {chronos_summary['engine_used']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if conviction.get("trap_warning"):
        for reason in conviction.get("trap_reasons", []):
            st.markdown(
                f'<div class="trap-alert">🚨 <b>TRAP WARNING ALERT:</b> {reason}</div>',
                unsafe_allow_html=True,
            )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card(
            f"Chronos Bear (P10)",
            f"{curr}{chronos_summary['end_p10']:,.2f}",
            f"{chronos_summary['downside_pct']:+.2f}% ({horizon} bars)",
            "#ff3b5c",
        )
    with c2:
        render_kpi_card(
            f"Chronos Median (P50)",
            f"{curr}{chronos_summary['end_p50']:,.2f}",
            f"{chronos_summary['expected_return_pct']:+.2f}% Expected",
            "#38bdf8",
        )
    with c3:
        render_kpi_card(
            f"Chronos Bull (P90)",
            f"{curr}{chronos_summary['end_p90']:,.2f}",
            f"{chronos_summary['upside_pct']:+.2f}% (Skew {chronos_summary['rr_skew']:.2f}x)",
            "#00e676",
        )
    with c4:
        render_kpi_card(
            f"Stop-Loss (1.5x ATR)",
            f"{curr}{trade_plan['stop_loss']:,.2f}",
            f"Risk/Sh: {curr}{trade_plan['risk_per_share']:,.2f}",
            "#ff5252",
        )
    with c5:
        render_kpi_card(
            f"Take-Profit 1 & 2",
            f"{curr}{trade_plan['take_profit_1']:,.2f}",
            f"TP2: {curr}{trade_plan['take_profit_2']:,.2f}",
            "#00e676",
        )
    with c6:
        render_kpi_card(
            f"Position Size ({curr})",
            f"{trade_plan['shares']} Shares",
            f"Notional: {curr}{trade_plan['position_value']:,.0f}",
            "#ffd740",
        )

    st.divider()

    st.markdown(f"#### 🤖 `{selected_llm}` Institutional Trading Desk Briefing")
    brief_key = f"{active_ticker}_{interval}_{horizon}"

    b_col1, b_col2 = st.columns([1.4, 2.6])
    with b_col1:
        gen_clicked = st.button(
            f"⚡ Generate {selected_llm} Trading Desk Briefing",
            type="primary",
            use_container_width=True,
        )
    with b_col2:
        st.caption(
            f"Smart GPU Relay Step 2: Runs `{selected_llm}` (`num_ctx=4096, num_batch=512, temp=0.2, keep_alive=5m`) after clearing PyTorch VRAM."
        )

    if gen_clicked:
        with st.container(border=True):
            streamed_text = st.write_stream(
                stream_qwen_trading_briefing(selected_llm, stock_payload)
            )
            st.session_state["briefing_cache"][brief_key] = streamed_text
    elif brief_key in st.session_state["briefing_cache"]:
        with st.container(border=True):
            st.markdown(st.session_state["briefing_cache"][brief_key])
    else:
        st.info(
            f"Click **⚡ Generate {selected_llm} Trading Desk Briefing** above to stream a live institutional synthesis of "
            f"`{active_ticker}` covering Executive Verdict, Rising 44-SMA & RVOL Confluence, FinBERT News Catalysts, and the exact `{curr}` ATR Trade Plan."
        )

    st.markdown(f"#### 💬 Chat With `{selected_llm}` About `{active_ticker}`")
    if active_ticker not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][active_ticker] = []

    chat_container = st.container(border=True)
    with chat_container:
        if not st.session_state["chat_histories"][active_ticker]:
            st.caption(
                f"Ask `{selected_llm}` anything about `{active_ticker}` (e.g., *'Is the 44-SMA bounce confirmed by RVOL?'*, *'What are the main bearish news risks?'*, or *'How should I scale into this position?'*)."
            )
        for msg in st.session_state["chat_histories"][active_ticker]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_q = st.chat_input(f"Ask {selected_llm} about {active_ticker} ({curr})...")
    if user_q:
        st.session_state["chat_histories"][active_ticker].append({"role": "user", "content": user_q})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                ans = st.write_stream(
                    stream_qwen_stock_chat(
                        selected_llm,
                        stock_payload,
                        st.session_state["chat_histories"][active_ticker][:-1],
                        user_q,
                    )
                )
        st.session_state["chat_histories"][active_ticker].append({"role": "assistant", "content": ans})

    st.divider()

    st.markdown(f"#### 📝 Log Setup to `{m_code}` Virtual Paper Trading Ledger ({curr})")
    default_dir_idx = 0 if conviction["master_score"] >= 45.0 else 1
    with st.form("log_paper_trade_form"):
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            form_dir = st.selectbox("Direction", options=["LONG", "SHORT"], index=default_dir_idx)
        with f2:
            form_entry = st.number_input(f"Entry Price ({curr})", value=float(trade_plan["entry_price"]), format="%.2f")
        with f3:
            form_shares = st.number_input("Shares", value=int(trade_plan["shares"]), min_value=1, step=1)
        with f4:
            form_sl = st.number_input(f"Stop-Loss (1.5x ATR) ({curr})", value=float(trade_plan["stop_loss"]), format="%.2f")
        with f5:
            form_tp = st.number_input(f"Take-Profit (Chronos) ({curr})", value=float(trade_plan["take_profit_1"]), format="%.2f")

        form_notes = st.text_input(
            "Trade Thesis / Notes",
            value=f"{conviction['verdict']} ({conviction['master_score']}/100) | 44-SMA: {ta_summary['sma44_status']} | RVOL: {ta_summary['rvol']:.2f}x",
        )
        submit_trade = st.form_submit_button(
            f"✅ Log {active_ticker} Trade to {m_code} SQLite Portfolio ({curr})",
            type="primary",
            use_container_width=True,
        )
        if submit_trade:
            ok, msg = log_paper_trade(
                market=m_code,
                currency=curr,
                ticker=active_ticker,
                interval=interval,
                horizon_bars=horizon,
                direction=form_dir,
                entry_price=form_entry,
                shares=form_shares,
                stop_loss=form_sl,
                take_profit=form_tp,
                chronos_p10=chronos_summary["end_p10"],
                chronos_p50=chronos_summary["end_p50"],
                chronos_p90=chronos_summary["end_p90"],
                ta_score=conviction["ta_score"],
                finbert_score=conviction["finbert_norm_score"],
                chronos_score=conviction["chronos_score"],
                master_conviction=conviction["master_score"],
                sma44_setup=int(bool(ta_summary["sma44_bounce"] or ta_summary["sma44_rising"])),
                rvol=ta_summary["rvol"],
                notes=form_notes,
            )
            if ok:
                st.success(msg)


# =====================================================================
# TAB 4: DUAL-MARKET TRUSTED STOCK DISCOVERY (SPLIT INTO SUB-TABS 4A & 4B)
# =====================================================================
with tab4:
    st.markdown(
        f"#### 🔍 {market_cfg['market_label']} — Trusted Internet Stock Discovery Engine"
    )
    subtab_4a, subtab_4b = st.tabs(
        [
            "🏛️ Sub-Tab 4A: Institutional Blue-Chip Movers",
            "🚀 Sub-Tab 4B: Valuable Underdogs (Undervalued Mid-Cap Growth Scanner)",
        ]
    )

    # -----------------------------------------------------------------
    # SUB-TAB 4A: INSTITUTIONAL BLUE-CHIP MOVERS (LARGE / MEGA-CAP)
    # -----------------------------------------------------------------
    with subtab_4a:
        st.caption(
            "Harvests live Google News RSS catalyst upgrades + S&P 500 / Nifty 50 liquid leaders, enforces the Stage 2 Hard Institutional Gatekeeper, "
            "and ranks Blue-Chip survivors using Rising 44-SMA Swing Structure, RVOL, FinBERT GPU Sentiment, and Chronos-Bolt."
        )

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            if m_code == "IN":
                min_mcap_val = st.slider(
                    "Min Market Cap (₹ Crores)",
                    min_value=5000,
                    max_value=100000,
                    value=20000,
                    step=5000,
                    help="₹1 Crore = ₹10,000,000. Default ₹20,000 Cr = ₹200B raw INR.",
                    key="bc_mcap_in",
                )
            else:
                min_mcap_val = st.slider(
                    "Min Market Cap ($ Billions)",
                    min_value=2.0,
                    max_value=100.0,
                    value=10.0,
                    step=2.0,
                    key="bc_mcap_us",
                )
        with fc2:
            min_price_val = st.number_input(
                f"Min Share Price ({curr})",
                min_value=1.0,
                max_value=10000.0,
                value=float(market_cfg["min_price_default"]),
                step=5.0 if m_code == "US" else 25.0,
                key=f"bc_price_{m_code}",
            )
        with fc3:
            min_vol_val = st.number_input(
                "Min Avg Daily Volume",
                min_value=50000,
                max_value=50000000,
                value=int(market_cfg["min_vol_default"]),
                step=250000,
                key=f"bc_vol_{m_code}",
            )
        with fc4:
            min_analysts_val = st.number_input(
                "Min Covering Analysts",
                min_value=1,
                max_value=30,
                value=int(market_cfg["min_analysts_default"]),
                step=1,
                key=f"bc_an_{m_code}",
            )
        with fc5:
            scan_limit = st.slider(
                "Candidates to GPU-Scan",
                min_value=6,
                max_value=30,
                value=12,
                step=2,
                key=f"bc_lim_{m_code}",
            )

        st.caption(
            f"**Allowed Institutional Exchanges ({m_code}):** `{', '.join(market_cfg['allowed_exchanges'])}` • "
            f"**Active Blue-Chip Gatekeeper Floor:** Market Cap ≥ `{ '₹' + f'{min_mcap_val:,} Cr' if m_code == 'IN' else '$' + f'{min_mcap_val:,.0f}B' }`, "
            f"Price ≥ `{curr}{min_price_val:,.2f}`, Avg Vol ≥ `{min_vol_val:,}`, Analysts ≥ `{min_analysts_val}`"
        )

        scan_btn = st.button(
            f"🚀 Scan Internet & Institutional Feeds for Trusted {m_code} Blue-Chip Picks",
            type="primary",
            use_container_width=True,
            key=f"bc_scan_btn_{m_code}",
        )

        if scan_btn:
            prog_bar = st.progress(0.0, text="Initializing Smart GPU Relay Blue-Chip Discovery Scan...")

            def _update_prog(p: float, txt: str):
                prog_bar.progress(min(1.0, max(0.0, p)), text=txt)

            scan_out = run_institutional_discovery_scan(
                market_cfg=market_cfg,
                min_mcap_display=min_mcap_val,
                min_price=min_price_val,
                min_avg_volume=int(min_vol_val),
                min_analysts=int(min_analysts_val),
                max_candidates=int(scan_limit),
                interval=interval,
                horizon=horizon,
                relay_mode=relay_mode,
                selected_llm=selected_llm,
                progress_callback=_update_prog,
            )
            st.session_state["discovery_results"][m_code] = scan_out

            if scan_out["survivors"]:
                st.markdown(f"#### 🏆 `{selected_llm}` Executive Summary of Top 3 Blue-Chip Picks")
                with st.container(border=True):
                    summary_txt = st.write_stream(
                        stream_top3_discovery_summary(selected_llm, scan_out["survivors"], market_cfg)
                    )
                    st.session_state["discovery_summary"][m_code] = summary_txt

        saved_scan = st.session_state["discovery_results"].get(m_code)
        if saved_scan:
            survivors = saved_scan.get("survivors", [])
            if not scan_btn and m_code in st.session_state["discovery_summary"]:
                st.markdown(f"#### 🏆 `{selected_llm}` Executive Summary of Top 3 Blue-Chip Picks")
                with st.container(border=True):
                    st.markdown(st.session_state["discovery_summary"][m_code])

            if survivors:
                st.markdown(f"#### 📊 Top Trusted Blue-Chip Opportunities ({len(survivors)} Passed Gatekeeper)")
                disp_cols = [
                    "Ticker", "Company", "Price", "Market Cap", "Conviction", "Verdict",
                    "44-SMA Setup", "RVOL", "RS vs Bench (%)", "TA Score", "FinBERT Net",
                    "Chronos Exp Ret (%)", "Analyst Upside (%)", "Earnings Risk",
                ]
                surv_df = pd.DataFrame(survivors)[disp_cols]
                st.dataframe(surv_df, use_container_width=True, hide_index=True)

                st.markdown("##### ⚡ 1-Click Institutional Actions on Discovered Blue-Chips")
                act_c1, act_c2, act_c3, act_c4 = st.columns([1.4, 1.1, 1.1, 1.2])
                pick_tickers = [s["Ticker"] for s in survivors]
                with act_c1:
                    chosen_pick_sym = st.selectbox(
                        "Select Discovered Blue-Chip Stock",
                        options=pick_tickers,
                        key=f"disc_pick_{m_code}",
                    )
                chosen_row = next((s for s in survivors if s["Ticker"] == chosen_pick_sym), survivors[0])

                with act_c2:
                    st.write("")
                    if st.button(f"⭐ Save {chosen_pick_sym} to Watchlist", use_container_width=True, key=f"bc_wl_{m_code}"):
                        ok_w, msg_w = add_to_saved_watchlist(
                            market=m_code,
                            ticker=chosen_row["Ticker"],
                            company_name=chosen_row["Company"],
                            category="Blue-Chip",
                            price_at_add=chosen_row["Price_Raw"],
                            target_price=chosen_row["Take Profit"],
                            underdog_score=chosen_row["Conviction"],
                            pe_forward=chosen_row.get("Forward_PE", 0.0),
                            rev_growth_pct=chosen_row.get("Rev_Growth_Pct", 0.0),
                            catalyst_note=f"Blue-Chip {chosen_row['Verdict']} | {chosen_row['44-SMA Setup']}",
                        )
                        if ok_w:
                            st.success(msg_w)
                            st.rerun()

                with act_c3:
                    st.write("")
                    if st.button(f"📈 Load {chosen_pick_sym} in Analyzer", use_container_width=True, key=f"bc_load_{m_code}"):
                        st.session_state[active_key] = chosen_pick_sym
                        st.rerun()

                with act_c4:
                    st.write("")
                    if st.button(f"📝 Log {chosen_pick_sym} Paper Trade", type="primary", use_container_width=True, key=f"bc_log_{m_code}"):
                        ok, msg = log_paper_trade(
                            market=m_code,
                            currency=curr,
                            ticker=chosen_row["Ticker"],
                            interval=interval,
                            horizon_bars=horizon,
                            direction="LONG" if chosen_row["Conviction"] >= 45 else "SHORT",
                            entry_price=chosen_row["Price_Raw"],
                            shares=chosen_row["Shares"],
                            stop_loss=chosen_row["Stop Loss"],
                            take_profit=chosen_row["Take Profit"],
                            chronos_p10=chosen_row["Chronos P10"],
                            chronos_p50=chosen_row["Chronos P50"],
                            chronos_p90=chosen_row["Chronos P90"],
                            ta_score=chosen_row["TA Score"],
                            finbert_score=chosen_row["FinBERT Norm"],
                            chronos_score=chosen_row["ChronosScore"],
                            master_conviction=chosen_row["Conviction"],
                            sma44_setup=chosen_row["SMA44_Bounce_Flag"],
                            rvol=chosen_row["RVOL"],
                            notes=f"Tab 4A Blue-Chip Pick | {chosen_row['Verdict']} | {chosen_row['44-SMA Setup']}",
                        )
                        if ok:
                            st.success(msg)

            with st.expander("🛡️ View Anti-Penny-Stock Gatekeeper Rejections & Live RSS Catalyst Headlines"):
                if saved_scan.get("rejected"):
                    st.markdown("**Stocks Filtered Out by Hard Institutional Gatekeeper:**")
                    st.dataframe(pd.DataFrame(saved_scan["rejected"]), use_container_width=True, hide_index=True)
                else:
                    st.caption("All scanned candidates satisfied the institutional quality floor.")
                if saved_scan.get("catalyst_headlines"):
                    st.markdown("**Sample Live Google News RSS Catalyst Headlines Scanned:**")
                    for hl in saved_scan["catalyst_headlines"][:8]:
                        st.markdown(f"- {hl}")

    # -----------------------------------------------------------------
    # SUB-TAB 4B: VALUABLE UNDERDOGS (UNDERVALUED MID-CAP GROWTH & GARP)
    # -----------------------------------------------------------------
    with subtab_4b:
        st.markdown("##### 🚀 Valuable Underdogs — Undervalued Mid-Cap Growth & GARP Scanner")
        st.caption(
            "Engineered to uncover high-growth, fundamentally undervalued mid-cap stocks before they become crowded mega-caps. "
            "Combines Trusted Financial Media RSS (Barron's/Reuters/Zacks/Nasdaq & ET/Moneycontrol/Mint/Business Standard), "
            "`yfinance` Mid-Cap Growth Screeners, and a 4-Pillar 0–100 GARP + Coiled-Spring Technical Score."
        )

        # Adjustable Underdog Sweet-Spot Controls adapting to US vs India
        uf1, uf2, uf3, uf4, uf5, uf6 = st.columns([1.45, 0.9, 1.0, 0.95, 0.9, 0.8])
        with uf1:
            if m_code == "IN":
                ud_mcap_range = st.slider(
                    "Market Cap Sweet-Spot (₹ Crores)",
                    min_value=1000,
                    max_value=100000,
                    value=(3000, 35000),
                    step=1000,
                    help="Default ₹3,000 Cr to ₹35,000 Cr (blocks penny stocks <₹3,000 Cr and crowded mega-caps >₹35,000 Cr).",
                    key="ud_mcap_in",
                )
            else:
                ud_mcap_range = st.slider(
                    "Market Cap Sweet-Spot ($ Billions)",
                    min_value=0.5,
                    max_value=50.0,
                    value=(1.5, 15.0),
                    step=0.5,
                    help="Default $1.5B to $15.0B (strictly blocks penny stocks <$1.5B while filtering out mega-caps >$15B).",
                    key="ud_mcap_us",
                )
        with uf2:
            ud_min_price = st.number_input(
                f"Min Share Price ({curr})",
                min_value=1.0,
                max_value=5000.0,
                value=float(market_cfg["ud_min_price_default"]),
                step=2.0 if m_code == "US" else 10.0,
                key=f"ud_price_{m_code}",
            )
        with uf3:
            ud_min_vol = st.number_input(
                "Min Avg Daily Vol",
                min_value=50000,
                max_value=20000000,
                value=int(market_cfg["ud_min_vol_default"]),
                step=100000,
                key=f"ud_vol_{m_code}",
            )
        with uf4:
            ud_min_growth = st.number_input(
                "Min YoY Growth (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(market_cfg["ud_min_growth_default"]),
                step=5.0,
                help="Minimum YoY Revenue or Earnings growth %.",
                key=f"ud_growth_{m_code}",
            )
        with uf5:
            ud_max_fpe = st.number_input(
                "Max Forward P/E",
                min_value=5.0,
                max_value=100.0,
                value=float(market_cfg["ud_max_fpe_default"]),
                step=2.0,
                key=f"ud_fpe_{m_code}",
            )
        with uf6:
            ud_scan_limit = st.slider(
                "Max Scan",
                min_value=8,
                max_value=30,
                value=14,
                step=2,
                key=f"ud_lim_{m_code}",
            )

        mcap_str = (
            f"₹{ud_mcap_range[0]:,} Cr – ₹{ud_mcap_range[1]:,} Cr"
            if m_code == "IN"
            else f"${ud_mcap_range[0]:.1f}B – ${ud_mcap_range[1]:.1f}B"
        )
        st.caption(
            f"**Active Underdog Sweet-Spot ({m_code}):** Market Cap `{mcap_str}` • "
            f"Min Price `{curr}{ud_min_price:,.2f}` • Min Vol `{ud_min_vol:,}` • "
            f"YoY Growth ≥ `{ud_min_growth:.0f}%` • Forward P/E ≤ `{ud_max_fpe:.1f}x`"
        )

        ud_scan_btn = st.button(
            f"🚀 Scan for Valuable {m_code} Underdogs (Mid-Cap GARP & Coiled Springs)",
            type="primary",
            use_container_width=True,
            key=f"ud_scan_btn_{m_code}",
        )

        if ud_scan_btn:
            ud_prog = st.progress(0.0, text="Initializing Smart GPU Relay Valuable Underdog Scan...")

            def _update_ud_prog(p: float, txt: str):
                ud_prog.progress(min(1.0, max(0.0, p)), text=txt)

            ud_out = run_valuable_underdog_scan(
                market_cfg=market_cfg,
                mcap_range_display=ud_mcap_range,
                min_price=float(ud_min_price),
                min_avg_volume=int(ud_min_vol),
                min_growth_pct=float(ud_min_growth),
                max_forward_pe=float(ud_max_fpe),
                max_candidates=int(ud_scan_limit),
                interval=interval,
                horizon=horizon,
                relay_mode=relay_mode,
                selected_llm=selected_llm,
                progress_callback=_update_ud_prog,
            )
            st.session_state["underdog_results"][m_code] = ud_out

        saved_ud = st.session_state["underdog_results"].get(m_code)
        if saved_ud:
            underdogs = saved_ud.get("underdogs", [])
            if underdogs:
                st.markdown(
                    f"#### 🏆 Valuable Underdog Leaderboard ({len(underdogs)} Ranked Mid-Cap GARP Gems)"
                )
                ud_disp_cols = [
                    "Ticker",
                    "Company Name",
                    "Price",
                    "Market Cap",
                    "Underdog Score",
                    "Forward P/E",
                    "YoY Rev Growth (%)",
                    "ROE (%)",
                    "% Below 52W High",
                    "Analyst Upside (%)",
                    "44-SMA Status",
                    "FinBERT Sentiment",
                ]
                ud_df = pd.DataFrame(underdogs)[ud_disp_cols]
                st.dataframe(ud_df, use_container_width=True, hide_index=True)

                # Qwen 2.5 14B Underdog Thesis Generator Button & Stream
                th_col1, th_col2 = st.columns([1.5, 2.5])
                with th_col1:
                    gen_ud_thesis = st.button(
                        f"🧠 Generate {selected_llm} Underdog Thesis (Top 3 Gems)",
                        type="primary",
                        use_container_width=True,
                        key=f"gen_ud_thesis_{m_code}",
                    )
                with th_col2:
                    st.caption(
                        f"Streams a concise `{selected_llm}` comparison explaining why the Top 3 Underdogs are undervalued relative to their growth rate and identifying their exact 44-SMA / ATR entry zones."
                    )

                if gen_ud_thesis:
                    with st.container(border=True):
                        th_txt = st.write_stream(
                            stream_underdog_qwen_thesis(selected_llm, underdogs, market_cfg)
                        )
                        st.session_state["underdog_thesis"][m_code] = th_txt
                elif m_code in st.session_state["underdog_thesis"]:
                    with st.container(border=True):
                        st.markdown(st.session_state["underdog_thesis"][m_code])

                st.divider()

                # Top 5 Individual Underdog Spotlight Cards
                st.markdown("#### 🌟 Top 5 Underdog Spotlight Cards (1-Click Watchlist, Analyzer & Paper Trade)")
                for rank_idx, u_item in enumerate(underdogs[:5], start=1):
                    u_sym = u_item["Ticker"]
                    st.markdown(
                        f"""
                        <div class="underdog-card">
                            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                                <div>
                                    <span style="font-size:1.15rem;font-weight:800;color:#ffffff;">
                                        #{rank_idx} {u_sym} — {u_item['Company Name']}
                                    </span>
                                    <span style="margin-left:10px;color:#38bdf8;font-weight:700;">
                                        {u_item['Price']} ({u_item['Market Cap']})
                                    </span>
                                </div>
                                <div style="background:#00e676;color:#0b0f19;padding:3px 12px;border-radius:6px;font-weight:800;font-size:0.95rem;">
                                    Underdog Score: {u_item['Underdog Score']:.1f} / 100
                                </div>
                            </div>
                            <div class="underdog-badge">
                                💎 {u_item['Spotlight_Badge']}
                            </div>
                            <div style="font-size:0.84rem;color:#cbd5e1;margin-top:4px;">
                                <b>ROE:</b> {u_item['ROE (%)']:.1f}% •
                                <b>Discount vs 52W High:</b> -{u_item['% Below 52W High']:.1f}% •
                                <b>RVOL:</b> {u_item['RVOL']:.2f}x •
                                <b>FinBERT:</b> {u_item['FinBERT Sentiment']} •
                                <b>ATR Entry / Stop / Target:</b> Entry {u_item['Price']} | SL {curr}{u_item['Stop_Loss']:,.2f} | TP {curr}{u_item['Take_Profit']:,.2f} ({u_item['Shares']} sh)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button(
                            f"⭐ Add {u_sym} to Underdog Watchlist",
                            key=f"ud_add_wl_{m_code}_{u_sym}",
                            use_container_width=True,
                        ):
                            ok_w, msg_w = add_to_saved_watchlist(
                                market=m_code,
                                ticker=u_sym,
                                company_name=u_item["Company Name"],
                                category="Underdog Gem",
                                price_at_add=u_item["Price_Raw"],
                                target_price=u_item["Target_Price_Raw"],
                                underdog_score=u_item["Underdog Score"],
                                pe_forward=u_item["Forward P/E"],
                                rev_growth_pct=u_item["YoY Rev Growth (%)"],
                                catalyst_note=u_item["Spotlight_Badge"],
                            )
                            if ok_w:
                                st.success(msg_w)
                                st.rerun()
                    with b2:
                        if st.button(
                            f"📈 Load {u_sym} in Technical Analyzer",
                            key=f"ud_load_ta_{m_code}_{u_sym}",
                            use_container_width=True,
                        ):
                            st.session_state[active_key] = u_sym
                            st.rerun()
                    with b3:
                        if st.button(
                            f"📝 Send {u_sym} to Paper Trade / Sizer",
                            type="primary",
                            key=f"ud_send_pt_{m_code}_{u_sym}",
                            use_container_width=True,
                        ):
                            st.session_state["prefill_trade"] = {
                                "market": m_code,
                                "currency": curr,
                                "ticker": u_sym,
                                "company": u_item["Company Name"],
                                "entry_price": u_item["Price_Raw"],
                                "shares": u_item["Shares"],
                                "stop_loss": u_item["Stop_Loss"],
                                "take_profit": u_item["Take_Profit"],
                                "chronos_p10": u_item["Chronos_P10"],
                                "chronos_p50": u_item["Chronos_P50"],
                                "chronos_p90": u_item["Chronos_P90"],
                                "ta_score": u_item["TA_Score"],
                                "finbert_score": u_item["FinBERT_Norm"],
                                "chronos_score": u_item["Chronos_Score"],
                                "master_conviction": u_item["Underdog Score"],
                                "sma44_setup": u_item["SMA44_Bounce_Flag"],
                                "rvol": u_item["RVOL"],
                                "notes": f"Underdog Gem ({u_item['Underdog Score']}/100) | {u_item['Spotlight_Badge']}",
                            }
                            ok_pt, msg_pt = log_paper_trade(
                                market=m_code,
                                currency=curr,
                                ticker=u_sym,
                                interval=interval,
                                horizon_bars=horizon,
                                direction="LONG",
                                entry_price=u_item["Price_Raw"],
                                shares=u_item["Shares"],
                                stop_loss=u_item["Stop_Loss"],
                                take_profit=u_item["Take_Profit"],
                                chronos_p10=u_item["Chronos_P10"],
                                chronos_p50=u_item["Chronos_P50"],
                                chronos_p90=u_item["Chronos_P90"],
                                ta_score=u_item["TA_Score"],
                                finbert_score=u_item["FinBERT_Norm"],
                                chronos_score=u_item["Chronos_Score"],
                                master_conviction=u_item["Underdog Score"],
                                sma44_setup=u_item["SMA44_Bounce_Flag"],
                                rvol=u_item["RVOL"],
                                notes=f"Underdog Gem ({u_item['Underdog Score']}/100) | {u_item['Spotlight_Badge']}",
                            )
                            if ok_pt:
                                st.success(f"{msg_pt} (Also pre-filled in Tab 5 Sizer!)")

            with st.expander("📰 View Trusted Financial Media RSS Headlines & Sweet-Spot Filter Log"):
                if saved_ud.get("media_headlines"):
                    st.markdown("**Trusted Financial Media RSS Headlines Scanned:**")
                    for m_hl in saved_ud["media_headlines"][:8]:
                        st.markdown(f"- {m_hl}")
                if saved_ud.get("rejected"):
                    st.markdown("**Candidates Outside Strict Sweet-Spot Bounds:**")
                    st.dataframe(pd.DataFrame(saved_ud["rejected"]), use_container_width=True, hide_index=True)


# =====================================================================
# TAB 5: MULTI-CURRENCY SQLITE PAPER TRADING & MODEL RELIABILITY AUDITOR
# =====================================================================
with tab5:
    st.markdown(
        f"#### 🛡️ {market_cfg['market_label']} Paper Trading & Model Reliability Auditor (`{curr}` Portfolio)"
    )

    # Optional Quick Sizer Form pre-filled from Sub-Tab 4B ("Send to Paper Trade / Sizer") or active ticker
    pf = st.session_state.get("prefill_trade")
    if pf and pf.get("market") == m_code:
        with st.expander(f"📐 Pre-Filled Underdog Position Sizer & Trade Logger — `{pf['ticker']}` ({pf['company']})", expanded=True):
            with st.form("tab5_prefill_sizer_form"):
                ps1, ps2, ps3, ps4, ps5 = st.columns(5)
                with ps1:
                    pf_ticker = st.text_input("Ticker", value=pf["ticker"])
                with ps2:
                    pf_entry = st.number_input(f"Entry Price ({curr})", value=float(pf["entry_price"]), format="%.2f")
                with ps3:
                    pf_shares = st.number_input("Shares", value=int(pf["shares"]), min_value=1, step=1)
                with ps4:
                    pf_sl = st.number_input(f"Stop-Loss ({curr})", value=float(pf["stop_loss"]), format="%.2f")
                with ps5:
                    pf_tp = st.number_input(f"Take-Profit ({curr})", value=float(pf["take_profit"]), format="%.2f")
                pf_notes = st.text_input("Notes", value=pf["notes"])
                if st.form_submit_button(f"✅ Confirm & Log {pf['ticker']} Paper Trade ({curr})", type="primary"):
                    ok_pf, msg_pf = log_paper_trade(
                        market=m_code,
                        currency=curr,
                        ticker=pf_ticker,
                        interval=interval,
                        horizon_bars=horizon,
                        direction="LONG",
                        entry_price=pf_entry,
                        shares=pf_shares,
                        stop_loss=pf_sl,
                        take_profit=pf_tp,
                        chronos_p10=pf["chronos_p10"],
                        chronos_p50=pf["chronos_p50"],
                        chronos_p90=pf["chronos_p90"],
                        ta_score=pf["ta_score"],
                        finbert_score=pf["finbert_score"],
                        chronos_score=pf["chronos_score"],
                        master_conviction=pf["master_conviction"],
                        sma44_setup=pf["sma44_setup"],
                        rvol=pf["rvol"],
                        notes=pf_notes,
                    )
                    if ok_pf:
                        st.success(msg_pf)
                        st.rerun()

    top_bar1, top_bar2, top_bar3 = st.columns([1.6, 1.4, 1.0])
    with top_bar1:
        if st.button(
            "🔄 Sync & Audit Open Trades with Live Market Data",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Auditing open trades bar-by-bar against Yahoo Finance High/Low/Close..."):
                audit_res = sync_and_audit_open_trades(market_code=m_code)
                st.success(
                    f"Audited {audit_res['audited']} open trade(s); {audit_res['closed']} hit Stop-Loss / Take-Profit / Horizon expiry."
                )
    with top_bar2:
        combined_calib = st.checkbox(
            "🌐 Combine US + India Trades for Model Calibration Accuracy",
            value=False,
        )
    with top_bar3:
        if st.button(f"🗑️ Reset {m_code} Portfolio", use_container_width=True):
            ok, msg = reset_market_portfolio(m_code)
            if ok:
                st.warning(msg)
                st.rerun()

    rel = compute_reliability_metrics(market_code=m_code, combined_calibration=combined_calib)

    rk1, rk2, rk3, rk4, rk5, rk6 = st.columns(6)
    with rk1:
        render_kpi_card(
            f"{m_code} Total Equity ({curr})",
            f"{curr}{rel['total_equity']:,.2f}",
            f"Cash: {curr}{rel['current_cash']:,.0f}",
            "#00e676" if rel["total_equity"] >= rel["starting_cash"] else "#ff3b5c",
        )
    with rk2:
        tot_pnl = rel["realized_pnl"] + rel["unrealized_pnl"]
        render_kpi_card(
            "Total Return (%)",
            f"{rel['total_return_pct']:+.2f}%",
            f"Net PnL: {curr}{tot_pnl:+,.2f}",
            "#00e676" if tot_pnl >= 0 else "#ff3b5c",
        )
    with rk3:
        render_kpi_card(
            "Win Rate (%)",
            f"{rel['win_rate']:.1f}%",
            f"Total Trades: {rel['total_trades']} ({rel['open_trades_count']} Open)",
            "#00e676" if rel["win_rate"] >= 50 else "#ffca28",
        )
    with rk4:
        render_kpi_card(
            "Profit Factor",
            f"{rel['profit_factor']:.2f}x",
            "Gross Wins / Gross Losses",
            "#38bdf8",
        )
    with rk5:
        render_kpi_card(
            "Chronos Cone Hit Rate",
            f"{rel['cone_hit_rate']:.1f}%",
            "Price Inside [P10, P90]",
            "#00e676" if rel["cone_hit_rate"] >= 65 else "#38bdf8",
        )
    with rk6:
        render_kpi_card(
            "Rising 44-SMA Win Rate",
            f"{rel['sma44_win_rate']:.1f}%",
            "44-SMA Swing Setups",
            "#ffd740",
        )

    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        fig_eq = build_equity_curve_chart(rel["equity_curve"], currency_symbol=curr)
        st.plotly_chart(fig_eq, use_container_width=True)
    with ch_col2:
        fig_cal = build_calibration_chart(rel["calibration_bars"])
        st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown(f"#### 📜 `{m_code}` Virtual Portfolio Trade Ledger ({curr})")
    trades_df = get_trades_df(m_code)
    if not trades_df.empty:
        open_trades = trades_df[trades_df["status"] == "OPEN"]
        if not open_trades.empty:
            mc1, mc2 = st.columns([2.5, 1.5])
            with mc1:
                close_id = st.selectbox(
                    "Select Open Trade to Close Manually at Live Market Price",
                    options=open_trades["id"].tolist(),
                    format_func=lambda tid: f"Trade #{tid} — {open_trades[open_trades['id'] == tid]['ticker'].values[0]} ({open_trades[open_trades['id'] == tid]['shares'].values[0]:.0f} sh @ {curr}{open_trades[open_trades['id'] == tid]['entry_price'].values[0]:,.2f})",
                )
            with mc2:
                st.write("")
                if st.button("⚡ Close Selected Trade at Live Price", use_container_width=True):
                    ok, msg = manual_close_trade(close_id)
                    if ok:
                        st.success(msg)
                        st.rerun()

        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            f"No paper trades logged yet in the `{m_code}` ({curr}) portfolio. "
            "Use the **1-Click Paper Trade** form in **Tab 3** or **Tab 4 (4A/4B)** to log your first institutional setup!"
        )
