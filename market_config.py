"""
Dual-Market Configuration & Safeguards (US NYSE/NASDAQ & India NSE/BSE).
Handles currency symbols, benchmarks, watchlists, smart Indian ticker normalization,
Google News RSS templates, Yahoo Finance interval/lookback caps,
Institutional Blue-Chip Universes, and Valuable Underdog (Mid-Cap GARP) Universes.
"""

import re
from typing import Dict, List, Any

MARKET_OPTIONS = [
    "🇺🇸 US Market (NYSE / NASDAQ)",
    "🇮🇳 Indian Market (NSE / BSE)",
]

US_WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "AMD", "META", "AMZN",
    "GOOGL", "TSLA", "JPM", "LLY", "SPY", "QQQ"
]

INDIA_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "ITC.NS", "SUNPHARMA.NS",
    "^NSEI", "^BSESN"
]

US_DISCOVERY_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "LLY", "JPM",
    "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "NFLX", "AMD", "CRM",
    "ORCL", "BAC", "ABBV", "CVX", "KO", "MRK", "WMT", "PEP", "TMO", "ACN"
]

INDIA_DISCOVERY_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS",
    "SBIN.NS", "LT.NS", "ITC.NS", "HCLTECH.NS", "SUNPHARMA.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "M&M.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS",
    "TITAN.NS", "ADANIENT.NS", "TATASTEEL.NS", "POWERGRID.NS", "ASIANPAINT.NS", "WIPRO.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "HAL.NS", "BEL.NS", "TRENT.NS", "ZOMATO.NS"
]

# Curated Liquid Mid-Cap Growth & GARP Underdog Universes (Sub-Tab 4B)
US_UNDERDOG_UNIVERSE = [
    "CELH", "DUOL", "ELF", "SOFI", "ROKU", "TOST", "CART", "CAVA", "BILL", "GTLB",
    "PATH", "APPS", "DOCS", "HALO", "EXEL", "LNTH", "MEDP", "ONTO", "RMBS", "NOVT",
    "BOOT", "WING", "SHAK", "CROX", "DECK", "ANF", "URBN", "POWL", "MOD", "FIX"
]

INDIA_UNDERDOG_UNIVERSE = [
    "DIXON.NS", "POLYCAB.NS", "PERSISTENT.NS", "COFORGE.NS", "KPITTECH.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "CUMMINSIND.NS", "ASTRAL.NS", "SUZLON.NS",
    "BHARATFORG.NS", "ASHOKLEY.NS", "AUROPHARMA.NS", "LUPIN.NS", "ALKEM.NS",
    "MAXHEALTH.NS", "SYNGENE.NS", "PIIND.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS",
    "TATACOMM.NS", "TATAELXSI.NS", "SONACOMS.NS", "KEI.NS", "KAYNES.NS",
    "JYOTICNC.NS", "CGPOWER.NS", "TIINDIA.NS", "VOLTAS.NS", "BLUESTARCO.NS"
]

US_CATALYST_QUERIES = [
    "Wall Street stock upgrade buy rating",
    "S&P 500 stock earnings beat guidance raised",
    "Nasdaq institutional breakout stock"
]

INDIA_CATALYST_QUERIES = [
    "NSE stock upgrade",
    "Nifty stock buy target",
    "Q2 Q3 earnings beat India"
]

# Trusted Financial Media RSS Queries for Sub-Tab 4B (Valuable Underdogs)
US_UNDERDOG_RSS_QUERY = (
    'site:barrons.com OR site:reuters.com OR site:zacks.com OR site:nasdaq.com '
    '("initiated with a buy" OR "undervalued growth" OR "mid-cap" OR "strong buy upgrade")'
)

INDIA_UNDERDOG_RSS_QUERY = (
    'site:economictimes.indiatimes.com OR site:moneycontrol.com OR site:livemint.com OR site:business-standard.com '
    '("midcap pick" OR "initiate coverage buy" OR "undervalued" OR "Nifty Midcap")'
)

INTERVAL_LOOKBACK_MAP: Dict[str, Dict[str, Any]] = {
    "15m": {
        "options": ["5d", "1mo", "60d"],
        "default": "1mo",
        "max_days": 60,
    },
    "1h": {
        "options": ["1mo", "3mo", "6mo", "1y", "2y"],
        "default": "6mo",
        "max_days": 730,
    },
    "1d": {
        "options": ["6mo", "1y", "2y", "5y", "max"],
        "default": "1y",
        "max_days": 3650,
    },
    "1wk": {
        "options": ["6mo", "1y", "2y", "5y", "max"],
        "default": "1y",
        "max_days": 3650,
    },
}


def get_market_config(market_label: str, indian_suffix: str = ".NS") -> Dict[str, Any]:
    """Return full market configuration dictionary based on selected sidebar label."""
    is_india = "Indian" in market_label or "🇮🇳" in market_label or market_label == "IN"
    if not is_india:
        return {
            "market_label": MARKET_OPTIONS[0],
            "market_code": "US",
            "currency_symbol": "$",
            "currency_code": "USD",
            "benchmark_ticker": "SPY",
            "benchmark_name": "S&P 500 (SPY)",
            "default_ticker": "NVDA",
            "watchlist": US_WATCHLIST,
            "rss_template": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
            "default_risk": 200.0,
            "supports_prepost": True,
            "allowed_exchanges": ["NYQ", "NMS", "NGM", "PCX", "NASDAQ", "NYSE"],
            "discovery_universe": US_DISCOVERY_UNIVERSE,
            "underdog_universe": US_UNDERDOG_UNIVERSE,
            "catalyst_queries": US_CATALYST_QUERIES,
            "underdog_rss_query": US_UNDERDOG_RSS_QUERY,
            # Blue-chip defaults (Sub-Tab 4A)
            "min_mcap_default": 10.0,   # $10 Billion
            "min_mcap_min": 2.0,        # $2 Billion
            "min_mcap_max": 100.0,      # $100 Billion
            "min_price_default": 20.0,
            "min_vol_default": 1_500_000,
            "min_analysts_default": 5,
            # Underdog Sweet-Spot defaults (Sub-Tab 4B)
            "ud_mcap_min_limit": 0.5,
            "ud_mcap_max_limit": 50.0,
            "ud_mcap_default_range": (1.5, 15.0),  # $1.5B to $15.0B
            "ud_min_price_default": 12.0,
            "ud_min_vol_default": 500_000,
            "ud_min_growth_default": 15.0,
            "ud_max_fpe_default": 28.0,
            "indian_suffix": indian_suffix,
        }
    else:
        watchlist = []
        for t in INDIA_WATCHLIST:
            if t.startswith("^"):
                watchlist.append(t)
            else:
                base = clean_indian_ticker(t)
                watchlist.append(f"{base}{indian_suffix}")
        return {
            "market_label": MARKET_OPTIONS[1],
            "market_code": "IN",
            "currency_symbol": "₹",
            "currency_code": "INR",
            "benchmark_ticker": "^NSEI",
            "benchmark_name": "Nifty 50 (^NSEI)",
            "default_ticker": f"RELIANCE{indian_suffix}",
            "watchlist": watchlist,
            "rss_template": "https://news.google.com/rss/search?q={clean_ticker}+share+price+NSE+India&hl=en-IN&gl=IN&ceid=IN:en",
            "default_risk": 5000.0,
            "supports_prepost": False,
            "allowed_exchanges": ["NSI", "BSE"],
            "discovery_universe": INDIA_DISCOVERY_UNIVERSE,
            "underdog_universe": INDIA_UNDERDOG_UNIVERSE,
            "catalyst_queries": INDIA_CATALYST_QUERIES,
            "underdog_rss_query": INDIA_UNDERDOG_RSS_QUERY,
            # Blue-chip defaults (Sub-Tab 4A)
            "min_mcap_default": 20000,   # ₹20,000 Crores
            "min_mcap_min": 5000,        # ₹5,000 Crores
            "min_mcap_max": 100000,      # ₹1,00,000 Crores
            "min_price_default": 100.0,
            "min_vol_default": 1_000_000,
            "min_analysts_default": 3,
            # Underdog Sweet-Spot defaults (Sub-Tab 4B)
            "ud_mcap_min_limit": 1000,
            "ud_mcap_max_limit": 100000,
            "ud_mcap_default_range": (3000, 35000),  # ₹3,000 Cr to ₹35,000 Cr
            "ud_min_price_default": 80.0,
            "ud_min_vol_default": 300_000,
            "ud_min_growth_default": 15.0,
            "ud_max_fpe_default": 32.0,
            "indian_suffix": indian_suffix,
        }


def clean_indian_ticker(ticker: str) -> str:
    """Strip .NS or .BO suffix from an Indian ticker."""
    if not ticker:
        return ""
    return re.sub(r"\.(NS|BO)$", "", ticker.strip(), flags=re.IGNORECASE).upper()


def normalize_ticker(ticker: str, market_code: str, indian_suffix: str = ".NS") -> str:
    """
    Normalize user-provided ticker symbol.
    For Indian market ('IN'), if no .NS/.BO suffix and not an index ('^'), append indian_suffix.
    """
    if not ticker:
        return "NVDA" if market_code == "US" else f"RELIANCE{indian_suffix}"
    t = ticker.strip().upper()
    if market_code == "IN":
        if not t.startswith("^") and not (t.endswith(".NS") or t.endswith(".BO")):
            t = f"{t}{indian_suffix}"
    return t


def build_google_news_rss_url(ticker: str, market_code: str) -> str:
    """Construct localized Google News RSS URL for US or Indian stocks."""
    if market_code == "IN":
        clean_t = clean_indian_ticker(ticker).replace("^", "")
        return f"https://news.google.com/rss/search?q={clean_t}+share+price+NSE+India&hl=en-IN&gl=IN&ceid=IN:en"
    else:
        clean_t = ticker.strip().upper().replace("^", "")
        return f"https://news.google.com/rss/search?q={clean_t}+stock&hl=en-US&gl=US&ceid=US:en"


def format_currency(value: float, currency_symbol: str = "$", decimals: int = 2) -> str:
    """Format numeric value with currency symbol ($ or ₹)."""
    if value is None or (isinstance(value, float) and (value != value)):
        return f"{currency_symbol}0.00"
    sign = "-" if value < 0 else ""
    abs_val = abs(float(value))
    return f"{sign}{currency_symbol}{abs_val:,.{decimals}f}"


def format_market_cap(raw_mcap: float, market_code: str) -> str:
    """
    Format market cap in Billions ($B) for US or Crores (₹ Cr) for India.
    Note: 1 Crore = 10,000,000 (1e7).
    """
    if not raw_mcap or raw_mcap <= 0:
        return "N/A"
    if market_code == "IN":
        crores = raw_mcap / 1e7
        return f"₹{crores:,.0f} Cr"
    else:
        billions = raw_mcap / 1e9
        return f"${billions:,.1f}B"
