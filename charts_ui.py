"""
Institutional Dark UI Theme & Interactive Plotly Charts for Tabs 1, 2, 3, and 5.
Includes:
  - High-Contrast Dark Institutional CSS (ensuring all labels, captions, inputs, buttons, and tabs are crystal-clear)
  - 3-Panel Technical + Rising 44-SMA + Chronos 10%-90% Probabilistic Cone Chart (Tab 1)
  - FinBERT Institutional News Sentiment Gauge [-100 to +100] (Tab 2)
  - Multi-Currency Virtual Equity Curve & Model Calibration Bar Chart (Tab 5)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def inject_institutional_css() -> None:
    """Inject high-contrast dark institutional trading desk CSS into Streamlit."""
    st.markdown(
        """
        <style>
        /* Global Dark Institutional Theme & Top Header */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: #0b0f19 !important;
            color: #f8fafc !important;
        }
        [data-testid="stHeader"] {
            background-color: #0b0f19 !important;
            border-bottom: 1px solid #1e293b !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background-color: #0f1626 !important;
            border-right: 1px solid #1e293b !important;
            color: #f8fafc !important;
        }

        /* High-Contrast Typography: Headings, Paragraphs, Labels, Captions */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        p, span, div, label, li, .stMarkdown p {
            color: #f1f5f9;
        }
        /* Widget Labels (Sidebar & Main Workspace) */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        .stSelectbox label, .stTextInput label, .stNumberInput label,
        .stSlider label, .stRadio label, .stCheckbox label, .stToggle label {
            color: #f8fafc !important;
            font-weight: 600 !important;
            font-size: 0.90rem !important;
        }
        /* Captions, Descriptions & Subtitles — Bright Silver-Blue so they never blend into dark backgrounds */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span,
        .stCaption, small {
            color: #cbd5e1 !important;
            font-size: 0.85rem !important;
            opacity: 1 !important;
        }
        /* Tooltip / Help Icons */
        [data-testid="stTooltipIcon"] svg {
            fill: #94a3b8 !important;
            color: #94a3b8 !important;
        }

        /* Inline Code Badges (`AAPL`, `$`, `NVDA`) */
        code {
            background-color: #1e293b !important;
            color: #38bdf8 !important;
            border: 1px solid #334155 !important;
            padding: 2px 6px !important;
            border-radius: 5px !important;
            font-weight: 600 !important;
        }

        /* Input Boxes, Selectboxes, Number Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        input[type="text"],
        input[type="number"],
        textarea {
            background-color: #162032 !important;
            color: #ffffff !important;
            border-color: #334155 !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: #ffffff !important;
        }
        div[data-baseweb="select"] svg {
            fill: #e2e8f0 !important;
        }
        /* Dropdown Popover Menu */
        ul[data-baseweb="menu"],
        div[data-baseweb="popover"] ul {
            background-color: #131c31 !important;
            border: 1px solid #334155 !important;
        }
        li[data-baseweb="option"] {
            color: #f8fafc !important;
        }
        li[data-baseweb="option"]:hover {
            background-color: #1e293b !important;
        }
        /* Number Input Step Buttons (- / +) */
        [data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInputStepUp"] {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border-color: #334155 !important;
        }

        /* Buttons (Primary & Secondary) */
        .stButton > button,
        .stFormSubmitButton > button {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #475569 !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.15s ease-in-out;
        }
        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background-color: #263554 !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            border: 1px solid #38bdf8 !important;
            font-weight: 700 !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, #0369a1 0%, #1d4ed8 100%) !important;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.45) !important;
        }

        /* Tabs Navigation Bar */
        button[data-baseweb="tab"] {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }
        button[data-baseweb="tab"] p {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom-color: #38bdf8 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: #38bdf8 !important;
            font-weight: 700 !important;
        }

        /* Alert / Info Boxes & Expanders */
        [data-testid="stAlert"] {
            background-color: #131c31 !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
        [data-testid="stAlert"] p {
            color: #f1f5f9 !important;
        }
        [data-testid="stExpander"] {
            background-color: #111827 !important;
            border: 1px solid #1e293b !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary p {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }

        /* KPI Metric Cards */
        .kpi-card {
            background: linear-gradient(145deg, #131c31 0%, #0f172a 100%);
            border: 1px solid #263554;
            border-radius: 10px;
            padding: 12px 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
            margin-bottom: 8px;
        }
        .kpi-title {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #cbd5e1 !important;
            margin-bottom: 4px;
            font-weight: 700;
        }
        .kpi-value {
            font-size: 1.28rem;
            font-weight: 800;
            color: #ffffff !important;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 0.80rem;
            margin-top: 4px;
            font-weight: 600;
        }
        /* Telemetry Card in Sidebar */
        .telemetry-box {
            background: #131c31;
            border: 1px solid #334155;
            border-left: 4px solid #00e676;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 0.82rem;
            color: #f1f5f9 !important;
            margin-bottom: 12px;
        }
        .telemetry-box div {
            color: #f1f5f9 !important;
            margin-bottom: 2px;
        }
        /* Verdict & Warning Badges */
        .verdict-banner {
            border-radius: 10px;
            padding: 16px 20px;
            border: 1px solid #334155;
            background: linear-gradient(135deg, #131c31 0%, #172135 100%);
            margin-bottom: 14px;
        }
        .trap-alert {
            background: rgba(255, 59, 92, 0.18);
            border: 1px solid #ff3b5c;
            border-left: 5px solid #ff3b5c;
            border-radius: 8px;
            padding: 12px 16px;
            color: #ffe4e8 !important;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .earnings-alert {
            background: rgba(255, 179, 0, 0.18);
            border: 1px solid #ffb300;
            border-left: 5px solid #ffb300;
            border-radius: 8px;
            padding: 10px 14px;
            color: #fff0b3 !important;
            margin-bottom: 12px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, sub: str = "", sub_color: str = "#cbd5e1") -> None:
    """Render a compact dark institutional KPI card with high-contrast text."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub" style="color: {sub_color};">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_multi_panel_technical_chart(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    currency_symbol: str = "$",
    chronos_data: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Build Tab 1 dark-themed 3-panel Plotly chart:
      - Panel 1 (Main Price): Candlesticks + SMA_20, SMA_44 (bright gold/orange), EMA_50, SMA_200
        + Bollinger Bands + VWAP (on 15m/1h) + 20-bar Support/Resistance dashed lines
        + Candlestick Pattern markers + Chronos 10%-90% Probabilistic Forecast Cone.
      - Panel 2 (Volume & RVOL): Volume bars color-coded by candle direction + 20-bar Avg Volume line.
      - Panel 3 (Momentum): RSI_14 (with 30/70 bands) and MACD Histogram + Signal lines.
    """
    plot_df = df.tail(180).copy()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.18, 0.22],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
        subplot_titles=(
            f"{ticker} ({interval}) — Institutional Structure, Rising 44-SMA & Chronos Forecast Cone ({currency_symbol})",
            "Volume & 20-Bar Institutional Average (RVOL)",
            "Momentum: RSI (14) [Left] & MACD (12, 26, 9) [Right]",
        ),
    )

    # 1. Bollinger Bands (Upper & Lower with translucent fill)
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["BB_UPPER"],
            name="BB Upper (20,2)",
            line=dict(color="rgba(203, 213, 225, 0.35)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["BB_LOWER"],
            name="Bollinger Bands (20,2)",
            line=dict(color="rgba(203, 213, 225, 0.35)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.07)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # 2. Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name=f"{ticker} OHLC",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff3b5c",
            increasing_fillcolor="rgba(0, 230, 118, 0.85)",
            decreasing_fillcolor="rgba(255, 59, 92, 0.85)",
        ),
        row=1,
        col=1,
    )

    # 3. Moving Averages: SMA_20, SMA_44 (Highlighted in Bright Gold/Orange), EMA_50, SMA_200
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["SMA_20"],
            name="SMA 20",
            line=dict(color="#00e5ff", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["SMA_44"],
            name="★ SMA 44 (Swing Line)",
            line=dict(color="#ffb300", width=2.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["EMA_50"],
            name="EMA 50",
            line=dict(color="#b388ff", width=1.6, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["SMA_200"],
            name="SMA 200",
            line=dict(color="#ff4081", width=2.0),
        ),
        row=1,
        col=1,
    )

    # 4. Rolling 20-bar VWAP (always shown on 15m/1h intraday, optional legend toggle on daily)
    vwap_visible = True if interval in ("15m", "1h") else "legendonly"
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["VWAP"],
            name="Rolling 20b VWAP",
            line=dict(color="#38bdf8", width=1.8, dash="dashdot"),
            visible=vwap_visible,
        ),
        row=1,
        col=1,
    )

    # 5. 20-Bar Support & Resistance Dashed Horizontal Lines
    sup_val = float(plot_df["SUPPORT_20"].iloc[-1])
    res_val = float(plot_df["RESISTANCE_20"].iloc[-1])
    fig.add_hline(
        y=res_val,
        line_dash="dash",
        line_color="rgba(255, 82, 82, 0.75)",
        annotation_text=f"20b Res: {currency_symbol}{res_val:,.2f}",
        annotation_position="top left",
        annotation_font_color="#ff8a80",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=sup_val,
        line_dash="dash",
        line_color="rgba(0, 230, 118, 0.75)",
        annotation_text=f"20b Sup: {currency_symbol}{sup_val:,.2f}",
        annotation_position="bottom left",
        annotation_font_color="#69f0ae",
        row=1,
        col=1,
    )

    # 6. Rising 44-SMA Bounce Markers & Candlestick Pattern Markers
    bounce_df = plot_df[plot_df["SMA44_BOUNCE"] == 1]
    if not bounce_df.empty:
        fig.add_trace(
            go.Scatter(
                x=bounce_df.index,
                y=bounce_df["Low"] * 0.992,
                mode="markers",
                name="44-SMA Bounce Setup",
                marker=dict(symbol="star", size=12, color="#ffd740", line=dict(color="#000", width=1)),
                text=["Rising 44-SMA Bounce" for _ in range(len(bounce_df))],
                hovertemplate="%{text}<br>Price: " + currency_symbol + "%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    pat_df = plot_df[plot_df["CANDLE_PATTERN"] != ""].tail(12)
    if not pat_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pat_df.index,
                y=pat_df["High"] * 1.008,
                mode="markers",
                name="Candlestick Pattern",
                marker=dict(
                    symbol="diamond",
                    size=8,
                    color=np.where(pat_df["CANDLE_BIAS"] >= 0, "#00e676", "#ff5252"),
                ),
                text=pat_df["CANDLE_PATTERN"],
                hovertemplate="<b>%{text}</b><br>Date: %{x}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 7. Chronos-Bolt 10%-90% Probabilistic Forecast Cone
    if chronos_data and chronos_data.get("future_dates"):
        last_dt = plot_df.index[-1]
        last_close = float(plot_df["Close"].iloc[-1])
        f_dates = [last_dt] + list(chronos_data["future_dates"])
        p10_vals = [last_close] + list(chronos_data["p10"])
        p50_vals = [last_close] + list(chronos_data["p50"])
        p90_vals = [last_close] + list(chronos_data["p90"])

        fig.add_trace(
            go.Scatter(
                x=f_dates,
                y=p90_vals,
                name="Chronos P90 (90th)",
                line=dict(color="rgba(56, 189, 248, 0.45)", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=f_dates,
                y=p10_vals,
                name="Chronos 10%-90% Cone",
                line=dict(color="rgba(56, 189, 248, 0.45)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.18)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=f_dates,
                y=p50_vals,
                name="Chronos Median (P50)",
                line=dict(color="#38bdf8", width=2.6, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # =================================================================
    # PANEL 2: VOLUME & 20-BAR AVERAGE VOLUME (RVOL)
    # =================================================================
    vol_colors = np.where(plot_df["Close"] >= plot_df["Open"], "rgba(0, 230, 118, 0.75)", "rgba(255, 59, 92, 0.75)")
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            customdata=plot_df["RVOL"],
            hovertemplate="Vol: %{y:,.0f}<br>RVOL: %{customdata:.2f}x<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["VOL_SMA_20"],
            name="20b Avg Volume",
            line=dict(color="#ffb300", width=1.6),
        ),
        row=2,
        col=1,
    )

    # =================================================================
    # PANEL 3: MOMENTUM — RSI(14) & MACD (12, 26, 9)
    # =================================================================
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["RSI_14"],
            name="RSI (14)",
            line=dict(color="#38bdf8", width=1.8),
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255, 82, 82, 0.55)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0, 230, 118, 0.55)", row=3, col=1)

    macd_colors = np.where(plot_df["MACD_HIST"] >= 0, "rgba(0, 230, 118, 0.50)", "rgba(255, 59, 92, 0.50)")
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["MACD_HIST"],
            name="MACD Hist",
            marker_color=macd_colors,
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["MACD"],
            name="MACD",
            line=dict(color="#f472b6", width=1.4),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["MACD_SIGNAL"],
            name="MACD Signal",
            line=dict(color="#facc15", width=1.4, dash="dot"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9", size=12),
        height=820,
        margin=dict(l=15, r=15, t=50, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#f8fafc"),
            bgcolor="rgba(15, 23, 42, 0.85)",
        ),
    )
    # Ensure subplot title annotations are bright white
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=13, color="#ffffff")

    fig.update_yaxes(title_text=f"Price ({currency_symbol})", gridcolor="#1e293b", tickfont=dict(color="#e2e8f0"), title_font=dict(color="#f8fafc"), row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="#1e293b", tickfont=dict(color="#e2e8f0"), title_font=dict(color="#f8fafc"), row=2, col=1)
    fig.update_yaxes(title_text="RSI (14)", range=[0, 100], gridcolor="#1e293b", tickfont=dict(color="#e2e8f0"), title_font=dict(color="#f8fafc"), row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="MACD", showgrid=False, tickfont=dict(color="#e2e8f0"), title_font=dict(color="#f8fafc"), row=3, col=1, secondary_y=True)
    fig.update_xaxes(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0"))

    return fig


def build_sentiment_gauge(net_sentiment: float, regime_label: str) -> go.Figure:
    """
    Build Tab 2 Plotly Sentiment Gauge from -100 (Extreme Bearish) to +100 (Extreme Bullish).
    """
    bar_color = "#00e676" if net_sentiment >= 15 else ("#ff3b5c" if net_sentiment <= -15 else "#ffca28")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(net_sentiment),
            number={"suffix": " / 100", "font": {"size": 28, "color": "#ffffff"}},
            title={
                "text": f"<b style='color:#ffffff'>FinBERT Institutional Sentiment</b><br><span style='font-size:13px;color:#cbd5e1'>{regime_label}</span>",
                "font": {"color": "#ffffff", "size": 15},
            },
            gauge={
                "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": "#cbd5e1", "tickfont": {"color": "#e2e8f0"}},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [-100, -35], "color": "rgba(255, 59, 92, 0.25)"},
                    {"range": [-35, -12], "color": "rgba(255, 112, 67, 0.20)"},
                    {"range": [-12, 12], "color": "rgba(148, 163, 184, 0.15)"},
                    {"range": [12, 35], "color": "rgba(38, 166, 154, 0.20)"},
                    {"range": [35, 100], "color": "rgba(0, 230, 118, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.75,
                    "value": float(net_sentiment),
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
        height=310,
        margin=dict(l=25, r=25, t=60, b=20),
    )
    return fig


def build_equity_curve_chart(equity_curve: List[Dict[str, Any]], currency_symbol: str = "$") -> go.Figure:
    """Build Tab 5 Virtual Equity Curve over time for the active market ($ or ₹)."""
    df_eq = pd.DataFrame(equity_curve)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_eq["Step"],
            y=df_eq["Equity"],
            mode="lines+markers",
            name=f"Portfolio Equity ({currency_symbol})",
            line=dict(color="#00e676", width=3),
            marker=dict(size=8, color="#38bdf8"),
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.10)",
            text=df_eq["Timestamp"],
            hovertemplate="<b>%{x}</b><br>Equity: " + currency_symbol + "%{y:,.2f}<br>Time: %{text}<extra></extra>",
        )
    )
    if not df_eq.empty:
        seed_val = float(df_eq["Equity"].iloc[0])
        fig.add_hline(
            y=seed_val,
            line_dash="dash",
            line_color="#cbd5e1",
            annotation_text=f"Starting Capital ({currency_symbol}{seed_val:,.0f})",
            annotation_font_color="#f8fafc",
        )
    fig.update_layout(
        title=dict(text=f"<b>Virtual Portfolio Equity Curve ({currency_symbol})</b>", font=dict(color="#ffffff", size=15)),
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9", size=12),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(title=dict(text=f"Total Equity ({currency_symbol})", font=dict(color="#f8fafc")), tickfont=dict(color="#e2e8f0"), gridcolor="#1e293b"),
        xaxis=dict(title=dict(text="Trade Sequence", font=dict(color="#f8fafc")), tickfont=dict(color="#e2e8f0"), gridcolor="#1e293b"),
    )
    return fig


def build_calibration_chart(calibration_bars: List[Dict[str, Any]]) -> go.Figure:
    """
    Build Tab 5 Model Calibration & Component Accuracy Bar Chart comparing Win Rate across
    Conviction Score buckets (0-40, 40-60, 60-75, 75-100) and Math + FinBERT Agreement vs. Divergence.
    """
    df_cal = pd.DataFrame(calibration_bars)
    colors = [
        "#ff5252",
        "#ffca28",
        "#26a69a",
        "#00e676",
        "#38bdf8",
        "#f472b6",
    ]
    fig = go.Figure(
        go.Bar(
            x=df_cal["Bucket"],
            y=df_cal["Win Rate (%)"],
            marker_color=colors[: len(df_cal)],
            text=[f"{w:.1f}% ({n}t)" for w, n in zip(df_cal["Win Rate (%)"], df_cal["Trades"])],
            textposition="outside",
            textfont=dict(color="#f8fafc", size=12),
        )
    )
    fig.update_layout(
        title=dict(text="<b>Model Calibration: Win Rate by Conviction Tier & Math/News Confluence</b>", font=dict(color="#ffffff", size=15)),
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9", size=12),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(title=dict(text="Win Rate (%)", font=dict(color="#f8fafc")), range=[0, 115], tickfont=dict(color="#e2e8f0"), gridcolor="#1e293b"),
        xaxis=dict(title=dict(text="Calibration Bucket", font=dict(color="#f8fafc")), tickfont=dict(color="#e2e8f0"), gridcolor="#1e293b"),
    )
    return fig
