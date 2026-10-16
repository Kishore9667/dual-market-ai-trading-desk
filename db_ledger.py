"""
Multi-Currency SQLite Paper Trading & Model Reliability Auditor (Tab 5).
Maintains isolated US ($100,000 USD) and India (₹10,00,000 INR) virtual portfolios in paper_trading.db,
logs multi-modal paper trades, audits open trades bar-by-bar against live Yahoo Finance data
for Stop-Loss / Take-Profit / Horizon expiry / Chronos 10%-90% Cone Hit accuracy,
and computes Equity Curves and Model Calibration statistics.
"""

import os
import sqlite3
import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from ta_engine import fetch_ohlcv

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trading.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize paper_trading.db with portfolio_config and paper_trades tables,
    seeding separate US ($100,000) and IN (₹10,00,000) portfolios if not present.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_config (
            market TEXT PRIMARY KEY,
            currency TEXT,
            starting_cash REAL,
            current_cash REAL,
            updated_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT,
            currency TEXT,
            timestamp TEXT,
            ticker TEXT,
            interval TEXT,
            horizon_bars INTEGER,
            direction TEXT,
            entry_price REAL,
            shares REAL,
            stop_loss REAL,
            take_profit REAL,
            chronos_p10 REAL,
            chronos_p50 REAL,
            chronos_p90 REAL,
            ta_score REAL,
            finbert_score REAL,
            chronos_score REAL,
            master_conviction REAL,
            sma44_setup INTEGER,
            rvol REAL,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_timestamp TEXT,
            pnl_dollars REAL,
            pnl_percent REAL,
            cone_hit INTEGER,
            notes TEXT
        )
        """
    )

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        """
        INSERT OR IGNORE INTO portfolio_config (market, currency, starting_cash, current_cash, updated_at)
        VALUES ('US', '$', 100000.0, 100000.0, ?)
        """,
        (now_str,),
    )
    cur.execute(
        """
        INSERT OR IGNORE INTO portfolio_config (market, currency, starting_cash, current_cash, updated_at)
        VALUES ('IN', '₹', 1000000.0, 1000000.0, ?)
        """,
        (now_str,),
    )

    conn.commit()
    conn.close()


def get_portfolio_config(market_code: str = "US") -> Dict[str, Any]:
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM portfolio_config WHERE market = ?", (market_code,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "market": market_code,
        "currency": "₹" if market_code == "IN" else "$",
        "starting_cash": 1000000.0 if market_code == "IN" else 100000.0,
        "current_cash": 1000000.0 if market_code == "IN" else 100000.0,
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def log_paper_trade(
    market: str,
    currency: str,
    ticker: str,
    interval: str,
    horizon_bars: int,
    direction: str,
    entry_price: float,
    shares: float,
    stop_loss: float,
    take_profit: float,
    chronos_p10: float,
    chronos_p50: float,
    chronos_p90: float,
    ta_score: float,
    finbert_score: float,
    chronos_score: float,
    master_conviction: float,
    sma44_setup: int,
    rvol: float,
    notes: str = "",
) -> Tuple[bool, str]:
    """
    Insert a new paper trade into paper_trades and deduct position cost from current_cash.
    """
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notional = float(entry_price) * float(shares)

    cur.execute("SELECT current_cash FROM portfolio_config WHERE market = ?", (market,))
    row = cur.fetchone()
    curr_cash = float(row["current_cash"]) if row else (1000000.0 if market == "IN" else 100000.0)

    new_cash = curr_cash - notional
    cur.execute(
        "UPDATE portfolio_config SET current_cash = ?, updated_at = ? WHERE market = ?",
        (new_cash, now_str, market),
    )

    cur.execute(
        """
        INSERT INTO paper_trades (
            market, currency, timestamp, ticker, interval, horizon_bars, direction,
            entry_price, shares, stop_loss, take_profit,
            chronos_p10, chronos_p50, chronos_p90,
            ta_score, finbert_score, chronos_score, master_conviction,
            sma44_setup, rvol, status, exit_price, exit_timestamp,
            pnl_dollars, pnl_percent, cone_hit, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, NULL, 0.0, 0.0, NULL, ?)
        """,
        (
            market,
            currency,
            now_str,
            ticker.upper(),
            interval,
            int(horizon_bars),
            direction.upper(),
            round(float(entry_price), 2),
            round(float(shares), 2),
            round(float(stop_loss), 2),
            round(float(take_profit), 2),
            round(float(chronos_p10), 2),
            round(float(chronos_p50), 2),
            round(float(chronos_p90), 2),
            round(float(ta_score), 1),
            round(float(finbert_score), 1),
            round(float(chronos_score), 1),
            round(float(master_conviction), 1),
            int(bool(sma44_setup)),
            round(float(rvol), 2),
            round(float(entry_price), 2),
            notes,
        ),
    )

    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return True, f"Logged Paper Trade #{trade_id}: {direction} {shares:,.0f} shares of {ticker} at {currency}{entry_price:,.2f}"


def get_trades_df(market_code: Optional[str] = None) -> pd.DataFrame:
    """Return paper_trades filtered by market_code (or all markets if None)."""
    init_db()
    conn = get_connection()
    if market_code and market_code in ("US", "IN"):
        df = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE market = ? ORDER BY id DESC",
            conn,
            params=(market_code,),
        )
    else:
        df = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id DESC", conn)
    conn.close()
    return df


def manual_close_trade(trade_id: int) -> Tuple[bool, str]:
    """Manually close an OPEN trade at the latest live market price."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM paper_trades WHERE id = ?", (int(trade_id),))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, f"Trade #{trade_id} not found."
    trade = dict(row)
    if trade["status"] != "OPEN":
        conn.close()
        return False, f"Trade #{trade_id} is already closed ({trade['status']})."

    ticker = trade["ticker"]
    interval = trade["interval"] or "1d"
    entry_price = float(trade["entry_price"])
    shares = float(trade["shares"])
    direction = str(trade["direction"]).upper()
    market = trade["market"]
    curr = trade["currency"]

    df = fetch_ohlcv(ticker, interval=interval, period="5d")
    live_price = float(df["Close"].iloc[-1]) if not df.empty else float(trade["exit_price"] or entry_price)

    if direction == "SHORT":
        pnl_d = (entry_price - live_price) * shares
        pnl_pct = ((entry_price - live_price) / entry_price * 100.0) if entry_price else 0.0
    else:
        pnl_d = (live_price - entry_price) * shares
        pnl_pct = ((live_price - entry_price) / entry_price * 100.0) if entry_price else 0.0

    p10 = float(trade["chronos_p10"] or 0.0)
    p90 = float(trade["chronos_p90"] or 0.0)
    cone_hit = 1 if (p10 > 0 and p90 > 0 and p10 <= live_price <= p90) else 0

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        """
        UPDATE paper_trades
        SET status = 'CLOSED_MANUAL',
            exit_price = ?,
            exit_timestamp = ?,
            pnl_dollars = ?,
            pnl_percent = ?,
            cone_hit = ?
        WHERE id = ?
        """,
        (round(live_price, 2), now_str, round(pnl_d, 2), round(pnl_pct, 2), cone_hit, int(trade_id)),
    )

    # Return original notional + realized PnL to portfolio current_cash
    proceeds = (entry_price * shares) + pnl_d
    cur.execute(
        "UPDATE portfolio_config SET current_cash = current_cash + ?, updated_at = ? WHERE market = ?",
        (proceeds, now_str, market),
    )

    conn.commit()
    conn.close()
    return True, f"Closed Trade #{trade_id} ({ticker}) at {curr}{live_price:,.2f} | Realized PnL: {curr}{pnl_d:+,.2f} ({pnl_pct:+.2f}%)"


def reset_market_portfolio(market_code: str = "US") -> Tuple[bool, str]:
    """Reset the active market's virtual portfolio cash and clear its trade history."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_cash = 1000000.0 if market_code == "IN" else 100000.0
    curr = "₹" if market_code == "IN" else "$"

    cur.execute("DELETE FROM paper_trades WHERE market = ?", (market_code,))
    cur.execute(
        "UPDATE portfolio_config SET starting_cash = ?, current_cash = ?, updated_at = ? WHERE market = ?",
        (default_cash, default_cash, now_str, market_code),
    )
    conn.commit()
    conn.close()
    return True, f"Reset {market_code} Virtual Portfolio to {curr}{default_cash:,.2f} and cleared trade ledger."


def sync_and_audit_open_trades(market_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Download historical/latest yfinance bars since each OPEN trade's entry timestamp:
      - Evaluates bar-by-bar whether High/Low crossed stop_loss (CLOSED_SL) or take_profit (CLOSED_TP),
        or if horizon_bars elapsed (CLOSED_HORIZON), otherwise updates unrealized PnL (OPEN).
      - Evaluates whether the actual price after horizon_bars (or at close) landed inside [chronos_p10, chronos_p90] (cone_hit = 1 vs 0).
    """
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    if market_code and market_code in ("US", "IN"):
        cur.execute("SELECT * FROM paper_trades WHERE status = 'OPEN' AND market = ?", (market_code,))
    else:
        cur.execute("SELECT * FROM paper_trades WHERE status = 'OPEN'")

    open_rows = [dict(r) for r in cur.fetchall()]
    updated_count = 0
    closed_count = 0
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for trade in open_rows:
        t_id = trade["id"]
        ticker = trade["ticker"]
        interval = trade["interval"] or "1d"
        horizon = int(trade["horizon_bars"] or 14)
        direction = str(trade["direction"] or "LONG").upper()
        entry_p = float(trade["entry_price"])
        shares = float(trade["shares"])
        sl = float(trade["stop_loss"])
        tp = float(trade["take_profit"])
        p10 = float(trade["chronos_p10"] or 0.0)
        p90 = float(trade["chronos_p90"] or 0.0)
        market = trade["market"]

        period = "5d" if interval == "15m" else ("1mo" if interval == "1h" else "3mo")
        df = fetch_ohlcv(ticker, interval=interval, period=period)
        if df.empty:
            continue

        try:
            entry_dt = pd.to_datetime(trade["timestamp"])
            df_idx = pd.to_datetime(df.index)
            if df_idx.tz is not None:
                df_idx = df_idx.tz_localize(None)
            df_clean = df.copy()
            df_clean.index = df_idx
            bars_after = df_clean[df_clean.index > entry_dt]
        except Exception:
            bars_after = pd.DataFrame()

        new_status = "OPEN"
        exit_p = float(df["Close"].iloc[-1])
        exit_ts = None
        cone_hit = None

        if not bars_after.empty:
            for b_idx, (bar_ts, bar) in enumerate(bars_after.iterrows(), start=1):
                b_high = float(bar["High"])
                b_low = float(bar["Low"])
                b_close = float(bar["Close"])

                if direction == "SHORT":
                    if sl > 0 and b_high >= sl:
                        new_status = "CLOSED_SL"
                        exit_p = sl
                        exit_ts = str(bar_ts)[:19]
                        break
                    if tp > 0 and b_low <= tp:
                        new_status = "CLOSED_TP"
                        exit_p = tp
                        exit_ts = str(bar_ts)[:19]
                        break
                else:
                    if sl > 0 and b_low <= sl:
                        new_status = "CLOSED_SL"
                        exit_p = sl
                        exit_ts = str(bar_ts)[:19]
                        break
                    if tp > 0 and b_high >= tp:
                        new_status = "CLOSED_TP"
                        exit_p = tp
                        exit_ts = str(bar_ts)[:19]
                        break

                if b_idx >= horizon:
                    new_status = "CLOSED_HORIZON"
                    exit_p = b_close
                    exit_ts = str(bar_ts)[:19]
                    break

        if direction == "SHORT":
            pnl_d = (entry_p - exit_p) * shares
            pnl_pct = ((entry_p - exit_p) / entry_p * 100.0) if entry_p else 0.0
        else:
            pnl_d = (exit_p - entry_p) * shares
            pnl_pct = ((exit_p - entry_p) / entry_p * 100.0) if entry_p else 0.0

        if new_status != "OPEN":
            closed_count += 1
            if p10 > 0 and p90 > 0:
                cone_hit = 1 if (p10 <= exit_p <= p90) else 0
            proceeds = (entry_p * shares) + pnl_d
            cur.execute(
                "UPDATE portfolio_config SET current_cash = current_cash + ?, updated_at = ? WHERE market = ?",
                (proceeds, now_str, market),
            )
        else:
            # Even for OPEN trades, check if current price is currently tracking inside the Chronos cone
            if p10 > 0 and p90 > 0:
                cone_hit = 1 if (p10 <= exit_p <= p90) else 0

        cur.execute(
            """
            UPDATE paper_trades
            SET status = ?,
                exit_price = ?,
                exit_timestamp = ?,
                pnl_dollars = ?,
                pnl_percent = ?,
                cone_hit = ?
            WHERE id = ?
            """,
            (new_status, round(exit_p, 2), exit_ts, round(pnl_d, 2), round(pnl_pct, 2), cone_hit, t_id),
        )
        updated_count += 1

    conn.commit()
    conn.close()
    return {
        "audited": updated_count,
        "closed": closed_count,
        "timestamp": now_str,
    }


def compute_reliability_metrics(
    market_code: str = "US",
    combined_calibration: bool = False,
) -> Dict[str, Any]:
    """
    Compute top-level Reliability KPI Cards, Virtual Equity Curve, and Model Calibration buckets:
      - Virtual Portfolio Total Equity ($ or ₹)
      - Realized + Unrealized Return (%)
      - Win Rate (%)
      - Profit Factor
      - Chronos 10%-90% Cone Hit Rate (%)
      - Rising 44-SMA Setup Win Rate (%)
    """
    port = get_portfolio_config(market_code)
    starting_cash = float(port["starting_cash"])
    current_cash = float(port["current_cash"])
    curr = port["currency"]

    market_df = get_trades_df(market_code)
    calib_df = get_trades_df(None if combined_calibration else market_code)

    # Calculate open position market value in active market
    open_market_val = 0.0
    unrealized_pnl = 0.0
    realized_pnl = 0.0

    if not market_df.empty:
        open_rows = market_df[market_df["status"] == "OPEN"]
        closed_rows = market_df[market_df["status"] != "OPEN"]
        for _, r in open_rows.iterrows():
            notional = float(r["entry_price"]) * float(r["shares"])
            u_pnl = float(r["pnl_dollars"] or 0.0)
            open_market_val += notional + u_pnl
            unrealized_pnl += u_pnl
        for _, r in closed_rows.iterrows():
            realized_pnl += float(r["pnl_dollars"] or 0.0)

    total_equity = current_cash + open_market_val
    total_return_pct = ((total_equity - starting_cash) / starting_cash * 100.0) if starting_cash else 0.0

    # Evaluate Win Rate, Profit Factor, Cone Hit Rate, and 44-SMA Win Rate on calibration dataset
    # Include both closed trades and audited trades with non-zero PnL
    if not calib_df.empty:
        eval_df = calib_df.copy()
        wins = eval_df[eval_df["pnl_dollars"] > 0]
        losses = eval_df[eval_df["pnl_dollars"] < 0]
        decided_count = len(wins) + len(losses)
        win_rate = (len(wins) / decided_count * 100.0) if decided_count > 0 else 0.0

        gross_profit = float(wins["pnl_dollars"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["pnl_dollars"].sum())) if not losses.empty else 0.0
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = 9.99
        else:
            profit_factor = 0.0

        cone_valid = eval_df[eval_df["cone_hit"].notna()]
        cone_hit_rate = (
            float(cone_valid["cone_hit"].mean() * 100.0) if not cone_valid.empty else 0.0
        )

        sma44_trades = eval_df[eval_df["sma44_setup"] == 1]
        sma44_wins = sma44_trades[sma44_trades["pnl_dollars"] > 0]
        sma44_decided = sma44_trades[sma44_trades["pnl_dollars"] != 0]
        sma44_win_rate = (
            (len(sma44_wins) / len(sma44_decided) * 100.0) if len(sma44_decided) > 0 else 0.0
        )
    else:
        win_rate = 0.0
        profit_factor = 0.0
        cone_hit_rate = 0.0
        sma44_win_rate = 0.0

    # Build Virtual Equity Curve points for active market
    equity_curve = [{"Step": "Initial Seed", "Timestamp": port["updated_at"][:10], "Equity": starting_cash}]
    if not market_df.empty:
        chronological = market_df.sort_values("id", ascending=True)
        running_eq = starting_cash
        for _, r in chronological.iterrows():
            running_eq += float(r["pnl_dollars"] or 0.0)
            equity_curve.append(
                {
                    "Step": f"#{r['id']} {r['ticker']}",
                    "Timestamp": str(r["exit_timestamp"] or r["timestamp"])[:16],
                    "Equity": round(running_eq, 2),
                }
            )

    # Build Model Calibration Buckets (0-40, 40-60, 60-75, 75-100) + Math & FinBERT Agreement vs Divergence
    buckets = [
        ("Conviction 0-40", 0.0, 40.0),
        ("Conviction 40-60", 40.0, 60.0),
        ("Conviction 60-75", 60.0, 75.0),
        ("Conviction 75-100", 75.0, 100.1),
    ]
    calibration_bars: List[Dict[str, Any]] = []
    for label, lo, hi in buckets:
        if not calib_df.empty:
            sub = calib_df[(calib_df["master_conviction"] >= lo) & (calib_df["master_conviction"] < hi)]
            dec = sub[sub["pnl_dollars"] != 0]
            w = sub[sub["pnl_dollars"] > 0]
            wr = (len(w) / len(dec) * 100.0) if len(dec) > 0 else 0.0
            cnt = len(sub)
        else:
            wr, cnt = 0.0, 0
        calibration_bars.append({"Bucket": label, "Win Rate (%)": round(wr, 1), "Trades": cnt, "Group": "Conviction Tier"})

    # Math + FinBERT Agreed vs Diverged
    if not calib_df.empty:
        math_bull = (0.64 * calib_df["ta_score"] + 0.36 * calib_df["chronos_score"]) >= 52.0
        news_bull = calib_df["finbert_score"] >= 52.0
        agreed_mask = math_bull == news_bull

        agreed_df = calib_df[agreed_mask]
        diverged_df = calib_df[~agreed_mask]

        ag_dec = agreed_df[agreed_df["pnl_dollars"] != 0]
        ag_wr = (len(agreed_df[agreed_df["pnl_dollars"] > 0]) / len(ag_dec) * 100.0) if len(ag_dec) > 0 else 0.0

        div_dec = diverged_df[diverged_df["pnl_dollars"] != 0]
        div_wr = (len(diverged_df[diverged_df["pnl_dollars"] > 0]) / len(div_dec) * 100.0) if len(div_dec) > 0 else 0.0

        calibration_bars.append(
            {"Bucket": "Math + News Agreed", "Win Rate (%)": round(ag_wr, 1), "Trades": len(agreed_df), "Group": "Confluence Alignment"}
        )
        calibration_bars.append(
            {"Bucket": "Math vs News Diverged", "Win Rate (%)": round(div_wr, 1), "Trades": len(diverged_df), "Group": "Confluence Alignment"}
        )
    else:
        calibration_bars.append({"Bucket": "Math + News Agreed", "Win Rate (%)": 0.0, "Trades": 0, "Group": "Confluence Alignment"})
        calibration_bars.append({"Bucket": "Math vs News Diverged", "Win Rate (%)": 0.0, "Trades": 0, "Group": "Confluence Alignment"})

    return {
        "currency": curr,
        "starting_cash": starting_cash,
        "current_cash": round(current_cash, 2),
        "open_market_value": round(open_market_val, 2),
        "total_equity": round(total_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": profit_factor,
        "cone_hit_rate": round(cone_hit_rate, 1),
        "sma44_win_rate": round(sma44_win_rate, 1),
        "total_trades": len(market_df),
        "open_trades_count": len(market_df[market_df["status"] == "OPEN"]) if not market_df.empty else 0,
        "equity_curve": equity_curve,
        "calibration_bars": calibration_bars,
    }
