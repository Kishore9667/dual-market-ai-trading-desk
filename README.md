# 🏛️ Institutional Dual-Market AI Trading Desk (US & India)

A **100% local, Windows-compatible, GPU-accelerated** Dual-Market (**US NYSE/NASDAQ** & **India NSE/BSE**) Stock Intelligence, Trusted Internet Stock Discovery, Deep Learning Probabilistic Forecasting & Multi-Currency SQLite Paper-Trading Audit Dashboard.

Optimized for **NVIDIA GeForce RTX 3050 (8GB VRAM) + 32GB System RAM** with a **Smart GPU Relay** pipeline that runs **FinBERT (`ProsusAI/finbert`)**, **Amazon Chronos-Bolt (`amazon/chronos-bolt-small`)**, and **Ollama (`qwen2.5:14b`)** locally without VRAM collisions.

---

## ✨ Key Capabilities

1. **Smart GPU Relay (RTX 3050 8GB VRAM + 32GB System RAM)**
   - Sets `OLLAMA_FLASH_ATTENTION=1` at startup.
   - Moves `ProsusAI/finbert` and `amazon/chronos-bolt-small` to `cuda:0` for `<1s` batch inference, then immediately offloads both models back to the 32GB System RAM pool (`cpu`) and runs `gc.collect()` + `torch.cuda.empty_cache()` so `qwen2.5:14b` has full usable GPU VRAM.
   - Sends an automatic `keep_alive=0` unload ping to Ollama before running multi-stock GPU discovery scans in Tab 4.

2. **Dual-Market Engine (US NYSE/NASDAQ `$` & India NSE/BSE `₹`)**
   - Seamless switching between US Equities (`$`, benchmark `SPY`) and Indian Equities (`₹`, benchmark `^NSEI`).
   - Smart Indian ticker normalization (`TCS` → `TCS.NS` or `TCS.BO`) and localized Google News RSS feeds (`gl=US` vs. `gl=IN`).
   - Automatic Yahoo Finance interval & lookback safeguards (`15m`, `1h`, `1d`, `1wk`).

3. **Windows-Safe Technical Analysis Engine (`ta_engine.py`)**
   - Uses `TA-Lib` when installed with an automatic pure vectorized NumPy/Pandas fallback so the dashboard never crashes on Windows.
   - Computes `SMA_20`, `SMA_44`, `EMA_50`, `SMA_200`, **Rising 44-SMA Swing Setup & Bounce Detector**, Bollinger Bands & Squeeze, `RSI_14`, `MACD`, `ADX_14`, `ATR_14`, `RVOL`, `OBV` 10-bar slope, Rolling 20-bar `VWAP`, 20-bar Benchmark Relative Strength, 20-bar Support/Resistance, 7 Japanese Candlestick Patterns, and `<7d` Upcoming Earnings Warnings.

4. **5 Interactive Institutional Tabs**
   - **📈 Tab 1: Technicals & Swing Structure** — Multi-panel dark Plotly chart with Candlesticks, bright gold `SMA_44`, Bollinger Bands, `VWAP`, Support/Resistance, Candlestick Pattern markers, Chronos 10%–90% Forecast Cone, Volume/RVOL, and RSI/MACD.
   - **📰 Tab 2: FinBERT Institutional News** — GPU batch sentiment scoring across deduplicated `yfinance` + localized Google News RSS headlines with a `-100` to `+100` Plotly Sentiment Gauge.
   - **🧠 Tab 3: Chronos-Bolt, Qwen 14B & Trade Plan** — Master Conviction Score (`45% TA + 30% FinBERT + 25% Chronos-Bolt`), Automatic Trap Detection, streaming `qwen2.5:14b` Trading Desk Briefing, interactive stock chat, and 1-click ATR Paper Trade logging.
   - **🔍 Tab 4: Trusted Stock Discovery ("Anti-Penny-Stock Scanner")** — Enforces hard institutional floors (`$10B` / `₹20,000 Cr` Market Cap, `$20` / `₹100` Price, Volume, Exchange, Analyst coverage), ranks survivors on GPU, and streams a `qwen2.5:14b` Top 3 comparison.
   - **🛡️ Tab 5: Multi-Currency SQLite Paper Trading & Reliability Auditor** — Separate `US` (`$100,000`) and `IN` (`₹10,00,000`) virtual portfolios in `paper_trading.db`, bar-by-bar live audit (`CLOSED_SL`, `CLOSED_TP`, `CLOSED_HORIZON`, `cone_hit`), Virtual Equity Curve, and Model Calibration charts.

---

## 🚀 Quick Start (Windows)

```powershell
# 1. Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Run automated TA Engine verification self-test
.\.venv\Scripts\python.exe ta_engine.py

# 3. Launch the dashboard (or double-click run_app.bat)
.\.venv\Scripts\python.exe -m streamlit run app.py
```
