"""Central configuration: credentials, risk limits, instruments, timing, model knobs.

Everything tunable lives here so the other modules stay free of magic numbers.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import upstox_client

# --------------------------------------------------------------------------- #
# Paths / time
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "upstox.txt"
DB_PATH = BASE_DIR / "paper_trades.db"
CACHE_DIR = BASE_DIR / ".cache"

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


class ConfigError(RuntimeError):
    """Raised for missing / invalid configuration (e.g. absent access token)."""


# --------------------------------------------------------------------------- #
# Upstox credentials / client
# --------------------------------------------------------------------------- #
def load_access_token(path: Path | str = TOKEN_FILE) -> str:
    """Read the Upstox access token from ``upstox.txt`` (whitespace stripped)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(
            f"Access-token file not found: {p}. Create it with your current "
            "Upstox access token as its only content."
        )
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ConfigError(f"Access-token file is empty: {p}")
    if any(ch.isspace() for ch in token):
        raise ConfigError(f"Access-token file {p} must contain a single token, no inner whitespace.")
    return token


def build_configuration(token: str | None = None) -> upstox_client.Configuration:
    cfg = upstox_client.Configuration()
    cfg.access_token = token or load_access_token()
    return cfg


def build_api_client(token: str | None = None) -> upstox_client.ApiClient:
    return upstox_client.ApiClient(build_configuration(token))


# --------------------------------------------------------------------------- #
# Capital & risk
# --------------------------------------------------------------------------- #
CAPITAL: float = 200_000.0
MAX_RISK_PER_TRADE: float = 0.015          # of *current* equity -> Rs 3,000 at start
MAX_MARGIN_PER_CATEGORY: float = 0.30      # of current equity   -> Rs 60,000 at start
SLIPPAGE: float = 0.001                    # 0.1 % adverse on every simulated fill
BROKERAGE_PER_ORDER: float = 20.0          # flat Rs per leg-order (set 0 to disable)
MAX_OPEN_POSITIONS_PER_CATEGORY: int = 2

# Option liquidity filters used when mapping targets to contracts
MIN_OPTION_VOLUME: int = 50
MAX_OPTION_SPREAD_PCT: float = 0.05        # (ask-bid)/mid
MIN_OPTION_PREMIUM: float = 2.0            # Rs; avoids lottery tickets

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
CATEGORY_NIFTY = "NIFTY"
CATEGORY_STOCK = "STOCK"
CATEGORY_MCX = "MCX"
CATEGORIES = (CATEGORY_NIFTY, CATEGORY_STOCK, CATEGORY_MCX)


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str                      # human label & key throughout the system
    category: str
    exchange: str                    # "NSE" | "MCX"
    option_names: tuple[str, ...]    # candidate `underlying_symbol`/`name` values in the master
    fallback_underlying_key: str | None = None
    fallback_lot_size: int = 1
    strike_step: float = 1.0         # only used by the chain-less fallback path
    weekly_expiry_weekday: int | None = None   # Mon=0; used for historical is_expiry_day only


UNIVERSE: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("NIFTY", CATEGORY_NIFTY, "NSE", ("NIFTY",),
                   "NSE_INDEX|Nifty 50", 65, 50.0, weekly_expiry_weekday=1),
    InstrumentSpec("RELIANCE", CATEGORY_STOCK, "NSE", ("RELIANCE",), None, 500, 10.0),
    InstrumentSpec("HDFCBANK", CATEGORY_STOCK, "NSE", ("HDFCBANK",), None, 550, 10.0),
    InstrumentSpec("ICICIBANK", CATEGORY_STOCK, "NSE", ("ICICIBANK",), None, 700, 10.0),
    # Tata Motors was demerged in late 2025; the F&O underlying may now be TMPV.
    InstrumentSpec("TATAMOTORS", CATEGORY_STOCK, "NSE", ("TATAMOTORS", "TMPV"), None, 800, 5.0),
    # Crude options are on the MCX CRUDEOIL future; the key is resolved from the master.
    InstrumentSpec("CRUDEOIL", CATEGORY_MCX, "MCX", ("CRUDEOIL",), None, 100, 50.0),
)
SPEC_BY_SYMBOL = {s.symbol: s for s in UNIVERSE}

# Fallback lot-size tables. The live instrument master always wins; exchanges revise
# lot sizes periodically, so treat these as last-resort defaults.
LOT_SIZES: dict[str, int] = {s.symbol: s.fallback_lot_size for s in UNIVERSE}

# --------------------------------------------------------------------------- #
# Market timing (IST)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionTimes:
    open: dt.time
    close: dt.time
    first_entry: dt.time      # cold start: no new trades before this
    last_entry: dt.time       # no new entries after this
    square_off: dt.time       # hard exit of every position


NSE_SESSION = SessionTimes(dt.time(9, 15), dt.time(15, 30), dt.time(9, 30),
                           dt.time(14, 45), dt.time(15, 15))
MCX_SESSION = SessionTimes(dt.time(9, 0), dt.time(23, 30), dt.time(9, 30),
                           dt.time(22, 45), dt.time(23, 15))
SESSIONS = {"NSE": NSE_SESSION, "MCX": MCX_SESSION}
US_OVERLAP_START = dt.time(18, 30)
BOT_SHUTDOWN_TIME = dt.time(23, 35)   # process exits after the MCX square-off

# --------------------------------------------------------------------------- #
# Candles / model
# --------------------------------------------------------------------------- #
CANDLE_MINUTES = 5
HISTORY_DAYS = 8                 # calendar days of 5m history used to seed context
MAX_CANDLES_KEPT = 1500

MODEL_ID = "google/timesfm-3.0-pytorch"
CONTEXT_LEN = 512                # 5-min bars fed to the model (~7 NSE sessions)
HORIZON = 12                     # 12 x 5m = 1 hour ahead
MODEL_BATCH_SIZE = 4             # kept small for a 6 GB GTX 1660 Ti
BATCH_COLLECT_SECONDS = 1.5      # wait this long after the first job to batch symbols
ALLOW_CPU_FALLBACK = os.environ.get("ALLOW_CPU_FALLBACK", "0") == "1"
ATR_PERIOD = 20
PARKINSON_WINDOW = 12

# --------------------------------------------------------------------------- #
# Signal thresholds (strategy_engine)
# --------------------------------------------------------------------------- #
# Setup 1: momentum / option buying
MIN_DRIFT_ATR_MULT = 1.5         # |q50 - px| must exceed this many 20-bar ATRs
MIN_DRIFT_TO_SPREAD = 0.15       # ... and this fraction of the (q90 - q10) envelope
SPREAD_EXPANSION_MIN = 0.05      # envelope must be >= 5 % wider than its recent mean
SPREAD_HISTORY = 4               # predictions remembered per symbol
TARGET_DELTA_RANGE = (0.40, 0.50)
TARGET_DELTA = 0.45
MIN_STOP_ATR_MULT = 0.75         # floor on the underlying stop distance, in ATRs
# Setup 2: range compression / credit spreads
# The q90-q10 envelope is a horizon-wide quantity while ATR is per-bar, so ATR is scaled by
# sqrt(HORIZON) (random-walk scaling). For a driftless walk the envelope is ~2.0 x ATR x sqrt(H);
# "compressed" means clearly tighter than that.
COMPRESSION_RATIO = 1.6          # envelope < ratio * ATR20 * sqrt(HORIZON)
NEUTRAL_DRIFT_ATR_MULT = 0.75    # condor only when |q50 - px| < this many ATRs
CONDOR_TRAIL_R_FRACTION = 0.30   # trailing "R" for credit structures = 30 % of max profit
MAX_HOLD_BARS_SPREAD = 3 * HORIZON
SPREAD_WIDTH_STRIKES = 2         # long wing this many strikes beyond the short strike
MIN_CREDIT_TO_WIDTH = 0.12
CREDIT_PROFIT_TARGET = 0.70      # close when 70 % of the credit is captured
# Position management
MAX_HOLD_BARS = HORIZON          # stale-signal exit
TRAIL_START_R = 1.0              # peak P&L (in R) at which stop moves to break-even
TRAIL_LOCK_R = 2.0               # beyond this, lock TRAIL_LOCK_FRACTION of peak
TRAIL_LOCK_FRACTION = 0.5
REENTRY_COOLDOWN_BARS = 3


@dataclass
class RuntimeOptions:
    """CLI-driven switches (see main.py)."""
    mock_model: bool = False
    poll_only: bool = False
    symbols: tuple[str, ...] = field(default_factory=lambda: tuple(s.symbol for s in UNIVERSE))
    log_level: str = "INFO"
    flatten_on_exit: bool = True
    dashboard: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050
