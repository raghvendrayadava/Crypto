"""Feature engineering for TimesFM-3.

Design points
-------------
* The model never sees rupee prices. The target channel is ``z_t = ln(P_t / P_anchor)`` with the
  anchor being the latest closed bar, so the last value is always 0 and predictions map back to
  prices via ``P_anchor * exp(z)``.
* **Gap shielding**: the overnight (session-to-session) jump is removed before cumulating, i.e. the
  first bar of each session contributes only its intrabar return ``ln(C/O)``. Opening gaps therefore
  never masquerade as intraday momentum.
* Covariates: ``intraday_progress`` / ``is_us_overlap`` / ``is_expiry_day`` are *known in advance*, so
  they are passed as past-and-future covariates; Parkinson volatility is past-only.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

import config as cfg
from upstox_feed import Candle

_LN2_4 = 4.0 * math.log(2.0)
MIN_BARS = 64
SESSION_BREAK = dt.timedelta(minutes=90)


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
@dataclass
class ExpiryCalendar:
    """Answers ``is_expiry_day`` for past, present and future dates.

    The instrument master only lists *live* expiries, so history is approximated with the
    exchange's weekly-expiry weekday when one is configured (NIFTY); otherwise past days read 0.
    """
    known: frozenset[dt.date]
    weekly_weekday: Optional[int] = None
    today: Optional[dt.date] = None

    def is_expiry(self, d: dt.date) -> bool:
        if d in self.known:
            return True
        today = self.today or cfg.now_ist().date()
        return self.weekly_weekday is not None and d < today and d.weekday() == self.weekly_weekday


# --------------------------------------------------------------------------- #
# Core numerics
# --------------------------------------------------------------------------- #
def log_returns(close: np.ndarray) -> np.ndarray:
    r = np.zeros_like(close, dtype=np.float64)
    if len(close) > 1:
        r[1:] = np.log(close[1:] / close[:-1])
    return r


def parkinson_vol(high: np.ndarray, low: np.ndarray, window: int = cfg.PARKINSON_WINDOW) -> np.ndarray:
    """Rolling Parkinson range-based volatility (per-bar sigma of log price).

    sigma^2 = mean(ln(H/L)^2) / (4 ln 2) over ``window`` bars (expanding until full).
    """
    hl = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12)) ** 2
    csum = np.cumsum(np.insert(hl, 0, 0.0))
    idx = np.arange(1, len(hl) + 1)
    lo = np.maximum(0, idx - window)
    mean = (csum[idx] - csum[lo]) / (idx - lo)
    return np.sqrt(mean / _LN2_4)


def session_ids(starts: Sequence[dt.datetime]) -> np.ndarray:
    """0-based session index per bar; a new session starts on a date change or a >90 min hole."""
    ids = np.zeros(len(starts), dtype=np.int64)
    for i in range(1, len(starts)):
        new = starts[i].date() != starts[i - 1].date() or (starts[i] - starts[i - 1]) > SESSION_BREAK
        ids[i] = ids[i - 1] + (1 if new else 0)
    return ids


def gap_shielded_log_returns(candles: Sequence[Candle]) -> np.ndarray:
    """Log returns with the overnight gap replaced by the first bar's intrabar return."""
    o = np.array([c.open for c in candles], dtype=np.float64)
    c_ = np.array([c.close for c in candles], dtype=np.float64)
    r = log_returns(c_)
    sid = session_ids([c.start for c in candles])
    first = np.concatenate(([True], sid[1:] != sid[:-1]))
    r[first] = np.log(c_[first] / np.maximum(o[first], 1e-12))
    r[0] = 0.0
    return r


def anchored_log_prices(candles: Sequence[Candle]) -> np.ndarray:
    """z_t = ln(P_t / P_anchor) on the gap-shielded path; anchor = latest bar so z[-1] == 0."""
    cs = np.cumsum(gap_shielded_log_returns(candles))
    return cs - cs[-1]


def atr(candles: Sequence[Candle], period: int = cfg.ATR_PERIOD) -> float:
    """Simple-average true range (price units); session-opening bars ignore the prior close."""
    h = np.array([c.high for c in candles]); l = np.array([c.low for c in candles])
    cl = np.array([c.close for c in candles])
    tr = h - l
    if len(cl) > 1:
        sid = session_ids([c.start for c in candles])
        same = sid[1:] == sid[:-1]
        alt = np.maximum(np.abs(h[1:] - cl[:-1]), np.abs(l[1:] - cl[:-1]))
        tr[1:] = np.where(same, np.maximum(tr[1:], alt), tr[1:])
    return float(np.mean(tr[-period:])) if len(tr) else 0.0


def session_covariates(times: Sequence[dt.datetime], exchange: str,
                       is_expiry: Callable[[dt.date], bool]) -> np.ndarray:
    """Array (3, n): intraday_progress in [0,1], is_us_overlap, is_expiry_day."""
    s = cfg.SESSIONS[exchange]
    o = s.open.hour * 60 + s.open.minute
    span = float((s.close.hour * 60 + s.close.minute) - o)
    us = cfg.US_OVERLAP_START
    out = np.zeros((3, len(times)), dtype=np.float32)
    for i, t in enumerate(times):
        m = t.hour * 60 + t.minute
        out[0, i] = min(1.0, max(0.0, (m - o) / span))
        out[1, i] = 1.0 if (t.hour, t.minute) >= (us.hour, us.minute) else 0.0
        out[2, i] = 1.0 if is_expiry(t.date()) else 0.0
    return out


def future_bar_times(last_start: dt.datetime, horizon: int, exchange: str) -> list[dt.datetime]:
    """Start times of the next ``horizon`` *trading* bars.

    Steps are 5 minutes within a session; once a step would reach the exchange close it rolls to the
    next weekday's open, exactly how the (gap-shielded) history is laid out.
    """
    s = cfg.SESSIONS[exchange]
    step = dt.timedelta(minutes=cfg.CANDLE_MINUTES)
    out: list[dt.datetime] = []
    t = last_start
    for _ in range(horizon):
        t = t + step
        if t.time() >= s.close:
            d = t.date() + dt.timedelta(days=1)
            while d.weekday() >= 5:
                d += dt.timedelta(days=1)
            t = dt.datetime.combine(d, s.open, tzinfo=t.tzinfo)
        out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Bundle handed to the model worker
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    symbol: str
    exchange: str
    bar_start: dt.datetime           # start of the last *closed* bar (the anchor bar)
    anchor_price: float
    target: np.ndarray               # (n,)    z_t, last element == 0
    past_only: np.ndarray            # (1, n)  Parkinson vol (percent per bar)
    past_future: np.ndarray          # (3, n+H) progress / us-overlap / expiry
    horizon: int
    atr20: float                     # price units
    parkinson_now: float             # per-bar sigma of log price
    spot_is_expiry: bool

    def prices(self, z: np.ndarray) -> np.ndarray:
        """Map z (log-return from anchor) back to rupee prices."""
        return self.anchor_price * np.exp(z)

    def to_torch(self, device: str = "cuda"):
        """fp16 tensors for the three model inputs (lazy torch import)."""
        import torch
        kw = dict(device=device, dtype=torch.float16)
        return (torch.as_tensor(self.target, **kw), torch.as_tensor(self.past_only, **kw),
                torch.as_tensor(self.past_future, **kw))


def build_features(symbol: str, exchange: str, candles: Sequence[Candle], calendar: ExpiryCalendar,
                   horizon: int = cfg.HORIZON, ctx_len: int = cfg.CONTEXT_LEN) -> Optional[FeatureBundle]:
    """Build model inputs from closed candles; None when history is too short / corrupt."""
    candles = [c for c in candles if c.close > 0 and c.high >= c.low > 0][-ctx_len:]
    if len(candles) < MIN_BARS:
        return None
    last = candles[-1]
    target = anchored_log_prices(candles).astype(np.float32)
    if not np.all(np.isfinite(target)):
        return None
    hi = np.array([c.high for c in candles]); lo = np.array([c.low for c in candles])
    pv = parkinson_vol(hi, lo)
    times = [c.start for c in candles] + future_bar_times(last.start, horizon, exchange)
    cov = session_covariates(times, exchange, calendar.is_expiry)
    return FeatureBundle(
        symbol=symbol, exchange=exchange, bar_start=last.start, anchor_price=last.close,
        target=target, past_only=(pv * 100.0).astype(np.float32)[None, :], past_future=cov,
        horizon=horizon, atr20=atr(candles), parkinson_now=float(pv[-1]),
        spot_is_expiry=calendar.is_expiry(last.start.date()),
    )
