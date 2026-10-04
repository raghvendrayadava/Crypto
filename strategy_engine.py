"""Signal evaluation, risk sizing and paper execution.

Setups (from the 9-quantile forecast; q10/q50/q90 taken at the horizon end)
---------------------------------------------------------------------------
1. MOMENTUM  - clear drift (|q50-px| >= 1.5 ATR and >= 15 % of the envelope) while the
               q90-q10 envelope is *expanding* vs. recent forecasts -> buy a ~0.45-delta CE (up) / PE (down).
               Stop on the underlying at q10 (longs) / q90 (shorts), target at the opposite quantile.
2. CONDOR    - envelope compressed below ``COMPRESSION_RATIO * ATR20 * sqrt(H)`` and drift small ->
               iron condor: bull-put spread below q10 + bear-call spread above q90.
               Stops: underlying crossing q10 / q90.

Risk
----
* Per-trade risk <= 1.5 % of current equity (loss to the stop, estimated with option delta,
  clamped to the premium). A hard mark-to-market cap at the same amount backs the model-based stop.
* Credit structures are sized on *max loss* (width - credit), so risk is bounded even on a gap.
* Margin per asset category <= 30 % of equity. Premium (long) / width (spreads) counts as margin.
* Cold start 09:30, last entry cut-off, hard square-off 15:15 NSE / 23:15 MCX.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import config as cfg
from model_worker import Prediction
from upstox_feed import OptionChain, OptionQuote, select_credit_spread, select_directional_option

log = logging.getLogger(__name__)

SETUP_MOMENTUM = "MOMENTUM"
SETUP_CONDOR = "CONDOR"


# =========================================================================== #
# Persistence
# =========================================================================== #
class TradeDB:
    """SQLite log of predictions, fills, trades and equity (thread-safe)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, bar_start TEXT, anchor REAL,
        q10 REAL, q50 REAL, q90 REAL, drift REAL, spread REAL, atr20 REAL, backend TEXT, latency_s REAL);
    CREATE TABLE IF NOT EXISTS trades(
        id TEXT PRIMARY KEY, symbol TEXT, category TEXT, setup TEXT, direction INTEGER, status TEXT,
        opened_at TEXT, closed_at TEXT, expiry TEXT, lots INTEGER, lot_size INTEGER,
        risk_limit REAL, risk_at_stop REAL, margin REAL, entry_underlying REAL, stop_underlying REAL,
        target_underlying REAL, exit_underlying REAL, exit_reason TEXT,
        gross_pnl REAL, charges REAL, net_pnl REAL, notes TEXT);
    CREATE TABLE IF NOT EXISTS fills(
        id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id TEXT, ts TEXT, purpose TEXT,
        instrument_key TEXT, trading_symbol TEXT, side TEXT, qty INTEGER,
        ref_price REAL, fill_price REAL, slippage_cost REAL, charges REAL);
    CREATE TABLE IF NOT EXISTS equity(
        ts TEXT, equity REAL, realized REAL, unrealized REAL, open_positions INTEGER);
    """

    def __init__(self, path: Path | str = cfg.DB_PATH) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(self.SCHEMA)

    def _exec(self, sql: str, args: Sequence = ()) -> None:
        with self._lock:
            self._db.execute(sql, args)

    def query(self, sql: str, args: Sequence = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def log_prediction(self, ts: dt.datetime, p: Prediction) -> None:
        self._exec("INSERT INTO predictions(ts,symbol,bar_start,anchor,q10,q50,q90,drift,spread,atr20,backend,latency_s)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (ts.isoformat(), p.symbol, p.bar_start.isoformat(), p.anchor_price, p.q10, p.q50, p.q90,
                    p.expected_drift, p.vol_spread, p.atr20, p.backend, p.latency_s))

    def open_trade(self, pos: "Position") -> None:
        self._exec("INSERT INTO trades(id,symbol,category,setup,direction,status,opened_at,expiry,lots,lot_size,"
                   "risk_limit,risk_at_stop,margin,entry_underlying,stop_underlying,target_underlying,notes)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (pos.id, pos.symbol, pos.category, pos.setup, pos.direction, "OPEN", pos.opened_at.isoformat(),
                    pos.expiry.isoformat(), pos.lots, pos.lot_size, pos.risk_limit, pos.risk_at_stop, pos.margin,
                    pos.entry_underlying, pos.stop_underlying, pos.target_underlying, json.dumps(pos.notes)))

    def log_fill(self, trade_id: str, ts: dt.datetime, purpose: str, leg: "Leg", side: str,
                 ref: float, fill: float, charges: float) -> None:
        self._exec("INSERT INTO fills(trade_id,ts,purpose,instrument_key,trading_symbol,side,qty,ref_price,"
                   "fill_price,slippage_cost,charges) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (trade_id, ts.isoformat(), purpose, leg.instrument_key, leg.trading_symbol, side, leg.qty,
                    ref, fill, abs(fill - ref) * leg.qty, charges))

    def close_trade(self, pos: "Position", ts: dt.datetime, reason: str, exit_underlying: float,
                    gross: float, charges: float) -> None:
        self._exec("UPDATE trades SET status='CLOSED',closed_at=?,exit_reason=?,exit_underlying=?,gross_pnl=?,"
                   "charges=?,net_pnl=? WHERE id=?",
                   (ts.isoformat(), reason, exit_underlying, gross, charges, gross - charges, pos.id))

    def snapshot_equity(self, ts: dt.datetime, equity: float, realized: float, unrealized: float, n: int) -> None:
        self._exec("INSERT INTO equity VALUES(?,?,?,?,?)", (ts.isoformat(), equity, realized, unrealized, n))

    def abandon_open_trades(self) -> int:
        rows = self.query("SELECT COUNT(*) FROM trades WHERE status='OPEN'")
        self._exec("UPDATE trades SET status='ABANDONED',exit_reason='process restart (paper positions not recovered)'"
                   " WHERE status='OPEN'")
        return int(rows[0][0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


# =========================================================================== #
# Domain objects
# =========================================================================== #
@dataclass
class Leg:
    instrument_key: str
    trading_symbol: str
    kind: str
    strike: float
    side: int                        # +1 long / -1 short
    qty: int                         # contracts (units), = lots * lot_size
    entry_px: float                  # fill price incl. slippage
    last_px: float                   # latest mark
    exit_px: Optional[float] = None

    def pnl(self, px: float) -> float:
        return self.side * (px - self.entry_px) * self.qty


@dataclass
class Position:
    id: str
    symbol: str
    category: str
    exchange: str
    setup: str
    direction: int                   # +1 bullish / -1 bearish / 0 neutral
    legs: list[Leg]
    lots: int
    lot_size: int
    expiry: dt.date
    opened_at: dt.datetime
    risk_limit: float                # hard MTM cap (Rs), 1.5 % of equity at entry
    risk_at_stop: float              # model estimate of loss at the underlying stop
    margin: float
    entry_underlying: float
    stop_underlying: float           # momentum stop (or lower stop for condors)
    target_underlying: float         # momentum target (or upper stop for condors)
    r_unit: float                    # Rs per "R" for trailing
    max_profit: float = math.inf
    max_hold: dt.timedelta = dt.timedelta(minutes=cfg.MAX_HOLD_BARS * cfg.CANDLE_MINUTES)
    entry_charges: float = 0.0
    peak_pnl: float = 0.0
    pnl_floor: Optional[float] = None
    best_underlying: float = 0.0
    initial_stop_dist: float = 0.0
    notes: dict = field(default_factory=dict)

    def pnl(self) -> float:
        return sum(l.pnl(l.last_px) for l in self.legs)


@dataclass(frozen=True)
class Sizing:
    lots: int
    risk_at_stop: float
    margin: float


@dataclass(frozen=True)
class Signal:
    setup: str
    direction: int
    reason: str
    stop: float                      # momentum stop / condor lower stop
    target: float                    # momentum target / condor upper stop


class FeedLike(Protocol):            # the surface of UpstoxFeed that the engine needs
    underlyings: dict

    def underlying_price(self, symbol: str) -> Optional[float]: ...
    def last_price(self, key: str, max_age: float = 30.0) -> Optional[float]: ...
    def pick_expiry(self, symbol: str, today: Optional[dt.date] = None) -> Optional[dt.date]: ...
    def get_option_chain(self, symbol: str, expiry: dt.date) -> Optional[OptionChain]: ...
    def subscribe(self, keys) -> None: ...


# =========================================================================== #
# Signal rules (pure)
# =========================================================================== #
def evaluate_signal(pred: Prediction, spread_history: Sequence[float]) -> Optional[Signal]:
    atr = pred.atr20
    if atr <= 0 or pred.vol_spread <= 0:
        return None
    drift = pred.expected_drift
    px = pred.anchor_price

    # Setup 1 - momentum
    directional = (abs(drift) >= cfg.MIN_DRIFT_ATR_MULT * atr
                   and abs(drift) >= cfg.MIN_DRIFT_TO_SPREAD * pred.vol_spread)
    expanding = bool(spread_history) and \
        pred.vol_spread >= (1.0 + cfg.SPREAD_EXPANSION_MIN) * (sum(spread_history) / len(spread_history))
    if directional and expanding:
        d = 1 if drift > 0 else -1
        min_dist = cfg.MIN_STOP_ATR_MULT * atr
        if d > 0:
            stop = min(pred.q10, px - min_dist)           # q10, but never tighter than the floor
            target = max(pred.q90, px + abs(drift))
        else:
            stop = max(pred.q90, px + min_dist)
            target = min(pred.q10, px - abs(drift))
        return Signal(SETUP_MOMENTUM, d, f"drift {drift:+.2f} ({abs(drift) / atr:.1f} ATR), envelope expanding",
                      stop, target)

    # Setup 2 - range compression
    compressed = pred.vol_spread < cfg.COMPRESSION_RATIO * atr * math.sqrt(pred.horizon)
    if compressed and abs(drift) < cfg.NEUTRAL_DRIFT_ATR_MULT * atr and pred.q10 < px < pred.q90:
        return Signal(SETUP_CONDOR, 0, f"envelope {pred.vol_spread:.2f} < {cfg.COMPRESSION_RATIO}xATRxsqrt(H)",
                      pred.q10, pred.q90)
    return None


# =========================================================================== #
# Sizing (pure)
# =========================================================================== #
def slip_buy(px: float) -> float:
    return round(px * (1 + cfg.SLIPPAGE), 2)


def slip_sell(px: float) -> float:
    return round(px * (1 - cfg.SLIPPAGE), 2)


def size_long_option(q: OptionQuote, spot: float, stop_underlying: float, risk_budget: float,
                     margin_available: float) -> Optional[Sizing]:
    """Lots such that the estimated loss at the underlying stop is <= ``risk_budget``."""
    entry = slip_buy(q.ltp)
    delta = abs(q.delta) if q.delta else cfg.TARGET_DELTA
    dist = abs(spot - stop_underlying)
    premium_at_stop = max(0.0, entry - delta * dist)         # linear delta: conservative vs gamma
    loss_unit = entry - slip_sell(premium_at_stop)
    if loss_unit <= 0:
        return None
    budget = risk_budget - 2 * cfg.BROKERAGE_PER_ORDER
    lots = min(int(budget // (loss_unit * q.lot_size)), int(margin_available // (entry * q.lot_size)))
    if lots < 1:
        return None
    return Sizing(lots, lots * q.lot_size * loss_unit, lots * q.lot_size * entry)


def size_condor(put_short: OptionQuote, put_long: OptionQuote, call_short: OptionQuote, call_long: OptionQuote,
                risk_budget: float, margin_available: float) -> Optional[Sizing]:
    credit = (slip_sell(put_short.ltp) - slip_buy(put_long.ltp)) + (slip_sell(call_short.ltp) - slip_buy(call_long.ltp))
    width = max(put_short.strike - put_long.strike, call_long.strike - call_short.strike)
    max_loss_unit = width - credit
    if credit <= 0 or max_loss_unit <= 0 or credit / width < cfg.MIN_CREDIT_TO_WIDTH:
        return None
    lot = put_short.lot_size
    budget = risk_budget - 4 * cfg.BROKERAGE_PER_ORDER * 2
    lots = min(int(budget // (max_loss_unit * lot)), int(margin_available // (width * lot)))
    if lots < 1:
        return None
    return Sizing(lots, lots * lot * max_loss_unit, lots * lot * width)


# =========================================================================== #
# Engine
# =========================================================================== #
class StrategyEngine(threading.Thread):
    def __init__(self, feed: FeedLike, db: TradeDB, clock: Callable[[], dt.datetime] = cfg.now_ist) -> None:
        super().__init__(name="strategy-engine", daemon=True)
        self.feed, self.db, self.clock = feed, db, clock
        self.specs = {sym: u.spec for sym, u in feed.underlyings.items()}
        self.positions: dict[str, Position] = {}               # symbol -> open position
        self.realized = 0.0                                    # net of charges
        self._events: "queue.Queue[Prediction]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._spread_hist: dict[str, list[float]] = {s: [] for s in self.specs}
        self._cooldown_until: dict[str, dt.datetime] = {}
        self._last_snapshot = dt.datetime.min.replace(tzinfo=cfg.IST)
        self.rejections: list[str] = []                        # last few skip reasons (diagnostics)

    # ---- public API ----------------------------------------------------- #
    def on_prediction(self, pred: Prediction) -> None:
        self._events.put(pred)

    def stop(self) -> None:
        self._stop_evt.set()

    @property
    def equity(self) -> float:
        return cfg.CAPITAL + self.realized + sum(p.pnl() for p in self.positions.values())

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                try:
                    pred = self._events.get(timeout=1.0)
                except queue.Empty:
                    pred = None
                if pred is not None:
                    self.handle_prediction(pred)
                self.manage_positions()
                self._maybe_snapshot()
            except Exception:                                  # noqa: BLE001
                log.exception("strategy loop error")

    # ---- entries -------------------------------------------------------- #
    def can_enter(self, exchange: str, now: dt.datetime) -> bool:
        if now.weekday() >= 5:
            return False
        s = cfg.SESSIONS[exchange]
        return s.first_entry <= now.time() < s.last_entry

    def handle_prediction(self, pred: Prediction) -> None:
        now = self.clock()
        self.db.log_prediction(now, pred)
        hist = self._spread_hist.setdefault(pred.symbol, [])
        prior = list(hist)
        hist.append(pred.vol_spread)
        del hist[:-cfg.SPREAD_HISTORY - 1]
        prior = prior[-cfg.SPREAD_HISTORY:]

        spec = self.specs.get(pred.symbol)
        if spec is None or pred.symbol in self.positions:
            return
        if not self.can_enter(spec.exchange, now):
            return self._skip(pred.symbol, "outside entry window")
        if now - pred.bar_start > dt.timedelta(minutes=cfg.CANDLE_MINUTES * 3):
            return self._skip(pred.symbol, "stale prediction")
        if now < self._cooldown_until.get(pred.symbol, now):
            return self._skip(pred.symbol, "cooldown")
        sig = evaluate_signal(pred, prior)
        if sig is None:
            return
        spot = self.feed.underlying_price(pred.symbol)
        if not spot or abs(spot - pred.anchor_price) > 2.0 * pred.atr20:
            return self._skip(pred.symbol, "price moved away from forecast anchor")
        cat_positions = [p for p in self.positions.values() if p.category == spec.category]
        if len(cat_positions) >= cfg.MAX_OPEN_POSITIONS_PER_CATEGORY:
            return self._skip(pred.symbol, "category position limit")
        risk_budget = cfg.MAX_RISK_PER_TRADE * self.equity
        margin_avail = cfg.MAX_MARGIN_PER_CATEGORY * self.equity - sum(p.margin for p in cat_positions)
        if margin_avail <= 0:
            return self._skip(pred.symbol, "category margin exhausted")
        expiry = self.feed.pick_expiry(pred.symbol, now.date())
        chain = self.feed.get_option_chain(pred.symbol, expiry) if expiry else None
        if chain is None:
            return self._skip(pred.symbol, "no option chain")
        if sig.setup == SETUP_MOMENTUM:
            self._enter_momentum(pred, sig, spec, chain, spot, risk_budget, margin_avail, now)
        else:
            self._enter_condor(pred, sig, spec, chain, spot, risk_budget, margin_avail, now)

    def _skip(self, symbol: str, why: str) -> None:
        log.debug("%s: skip - %s", symbol, why)
        self.rejections = (self.rejections + [f"{symbol}: {why}"])[-20:]

    def _enter_momentum(self, pred: Prediction, sig: Signal, spec: cfg.InstrumentSpec, chain: OptionChain,
                        spot: float, risk_budget: float, margin_avail: float, now: dt.datetime) -> None:
        q = select_directional_option(chain, sig.direction, sig.target)
        if q is None:
            return self._skip(pred.symbol, "no liquid ~0.45-delta contract")
        size = size_long_option(q, spot, sig.stop, risk_budget, margin_avail)
        if size is None:
            return self._skip(pred.symbol, f"1 lot of {q.trading_symbol} exceeds risk/margin limits")
        qty = size.lots * q.lot_size
        leg = Leg(q.instrument_key, q.trading_symbol, q.kind, q.strike, +1, qty, slip_buy(q.ltp), q.ltp)
        dist = abs(spot - sig.stop)
        pos = Position(
            id=uuid.uuid4().hex[:12], symbol=pred.symbol, category=spec.category, exchange=spec.exchange,
            setup=SETUP_MOMENTUM, direction=sig.direction, legs=[leg], lots=size.lots, lot_size=q.lot_size,
            expiry=q.expiry, opened_at=now, risk_limit=risk_budget, risk_at_stop=size.risk_at_stop,
            margin=size.margin, entry_underlying=spot, stop_underlying=sig.stop, target_underlying=sig.target,
            r_unit=max(size.risk_at_stop, 1.0), best_underlying=spot, initial_stop_dist=dist,
            notes={"reason": sig.reason, "delta": q.delta, "q10": pred.q10, "q50": pred.q50, "q90": pred.q90},
        )
        self._open(pos, {leg.instrument_key: q.ltp}, now)

    def _enter_condor(self, pred: Prediction, sig: Signal, spec: cfg.InstrumentSpec, chain: OptionChain,
                      spot: float, risk_budget: float, margin_avail: float, now: dt.datetime) -> None:
        puts = select_credit_spread(chain, "PE", sig.stop)
        calls = select_credit_spread(chain, "CE", sig.target)
        if puts is None or calls is None:
            return self._skip(pred.symbol, "no liquid wings outside q10/q90")
        ps, pl = puts
        cs, cl = calls
        size = size_condor(ps, pl, cs, cl, risk_budget, margin_avail)
        if size is None:
            return self._skip(pred.symbol, "condor fails credit/risk/margin checks")
        qty = size.lots * ps.lot_size
        legs = [
            Leg(ps.instrument_key, ps.trading_symbol, "PE", ps.strike, -1, qty, slip_sell(ps.ltp), ps.ltp),
            Leg(pl.instrument_key, pl.trading_symbol, "PE", pl.strike, +1, qty, slip_buy(pl.ltp), pl.ltp),
            Leg(cs.instrument_key, cs.trading_symbol, "CE", cs.strike, -1, qty, slip_sell(cs.ltp), cs.ltp),
            Leg(cl.instrument_key, cl.trading_symbol, "CE", cl.strike, +1, qty, slip_buy(cl.ltp), cl.ltp),
        ]
        credit_total = sum(-l.side * l.entry_px * l.qty for l in legs)
        pos = Position(
            id=uuid.uuid4().hex[:12], symbol=pred.symbol, category=spec.category, exchange=spec.exchange,
            setup=SETUP_CONDOR, direction=0, legs=legs, lots=size.lots, lot_size=ps.lot_size, expiry=ps.expiry,
            opened_at=now, risk_limit=risk_budget, risk_at_stop=size.risk_at_stop, margin=size.margin,
            entry_underlying=spot, stop_underlying=sig.stop, target_underlying=sig.target,
            r_unit=max(cfg.CONDOR_TRAIL_R_FRACTION * credit_total, 1.0), max_profit=credit_total,
            max_hold=dt.timedelta(minutes=cfg.MAX_HOLD_BARS_SPREAD * cfg.CANDLE_MINUTES),
            best_underlying=spot, initial_stop_dist=0.0,
            notes={"reason": sig.reason, "credit": credit_total, "q10": pred.q10, "q50": pred.q50, "q90": pred.q90},
        )
        self._open(pos, {l.instrument_key: l.last_px for l in legs}, now)

    def _open(self, pos: Position, refs: dict[str, float], now: dt.datetime) -> None:
        pos.entry_charges = cfg.BROKERAGE_PER_ORDER * len(pos.legs)
        self.db.open_trade(pos)
        for leg in pos.legs:
            self.db.log_fill(pos.id, now, "ENTRY", leg, "BUY" if leg.side > 0 else "SELL",
                             refs[leg.instrument_key], leg.entry_px, cfg.BROKERAGE_PER_ORDER)
        self.feed.subscribe([l.instrument_key for l in pos.legs])
        self.positions[pos.symbol] = pos
        log.info("ENTER %s %s %s lots=%d risk@stop=Rs%.0f margin=Rs%.0f stop=%.2f target=%.2f | %s",
                 pos.setup, pos.symbol, "/".join(l.trading_symbol for l in pos.legs), pos.lots, pos.risk_at_stop,
                 pos.margin, pos.stop_underlying, pos.target_underlying, pos.notes.get("reason"))

    # ---- management ----------------------------------------------------- #
    def _mark(self, pos: Position) -> None:
        for leg in pos.legs:
            px = self.feed.last_price(leg.instrument_key)
            if px and px > 0:
                leg.last_px = px

    def manage_positions(self) -> None:
        now = self.clock()
        for symbol, pos in list(self.positions.items()):
            self._mark(pos)
            spot = self.feed.underlying_price(symbol) or pos.entry_underlying
            reason = self._exit_reason(pos, spot, now)
            if reason:
                self._close(pos, reason, spot, now)

    def _exit_reason(self, pos: Position, spot: float, now: dt.datetime) -> Optional[str]:
        if now.time() >= cfg.SESSIONS[pos.exchange].square_off or now.weekday() >= 5:
            return "EOD_SQUARE_OFF"
        pnl = pos.pnl()
        if pnl <= -pos.risk_limit:
            return "RISK_CAP"
        pos.peak_pnl = max(pos.peak_pnl, pnl)

        if pos.setup == SETUP_MOMENTUM:
            d = pos.direction
            pos.best_underlying = max(pos.best_underlying, spot) if d > 0 else min(pos.best_underlying, spot)
            self._trail_underlying(pos)
            if d * (pos.stop_underlying - spot) >= 0:
                return "STOP_UNDERLYING"
            if d * (spot - pos.target_underlying) >= 0:
                return "TARGET"
        else:
            if spot <= pos.stop_underlying or spot >= pos.target_underlying:
                return "STOP_BOUNDARY"
            if pos.max_profit > 0 and pnl >= cfg.CREDIT_PROFIT_TARGET * pos.max_profit:
                return "TARGET"

        # premium-based trailing floor (breakeven after 1R, lock half the peak after 2R)
        if pos.peak_pnl >= cfg.TRAIL_LOCK_R * pos.r_unit:
            pos.pnl_floor = max(pos.pnl_floor or 0.0, cfg.TRAIL_LOCK_FRACTION * pos.peak_pnl)
        elif pos.peak_pnl >= cfg.TRAIL_START_R * pos.r_unit:
            pos.pnl_floor = max(pos.pnl_floor or 0.0, 0.0)
        if pos.pnl_floor is not None and pnl <= pos.pnl_floor:
            return "TRAILING_STOP"
        if now - pos.opened_at >= pos.max_hold:
            return "TIME_STOP"
        return None

    @staticmethod
    def _trail_underlying(pos: Position) -> None:
        """After one initial-stop-distance of favourable move, ratchet the stop to trail by that distance."""
        d, dist = pos.direction, pos.initial_stop_dist
        if dist <= 0 or d * (pos.best_underlying - pos.entry_underlying) < dist:
            return
        trail = pos.best_underlying - d * dist
        pos.stop_underlying = max(pos.stop_underlying, trail) if d > 0 else min(pos.stop_underlying, trail)

    def _close(self, pos: Position, reason: str, spot: float, now: dt.datetime) -> None:
        self._mark(pos)
        charges = pos.entry_charges
        gross = 0.0
        for leg in pos.legs:
            ref = leg.last_px
            fill = slip_sell(ref) if leg.side > 0 else slip_buy(ref)
            leg.exit_px = fill
            gross += leg.pnl(fill)
            charges += cfg.BROKERAGE_PER_ORDER
            self.db.log_fill(pos.id, now, "EXIT", leg, "SELL" if leg.side > 0 else "BUY",
                             ref, fill, cfg.BROKERAGE_PER_ORDER)
        self.db.close_trade(pos, now, reason, spot, gross, charges)
        self.realized += gross - charges
        del self.positions[pos.symbol]
        self._cooldown_until[pos.symbol] = now + dt.timedelta(minutes=cfg.CANDLE_MINUTES * cfg.REENTRY_COOLDOWN_BARS)
        log.info("EXIT  %s %s reason=%s gross=Rs%.0f net=Rs%.0f equity=Rs%.0f",
                 pos.setup, pos.symbol, reason, gross, gross - charges, self.equity)

    def flatten_all(self, reason: str = "SHUTDOWN") -> None:
        now = self.clock()
        for symbol, pos in list(self.positions.items()):
            self._close(pos, reason, self.feed.underlying_price(symbol) or pos.entry_underlying, now)

    def _maybe_snapshot(self) -> None:
        now = self.clock()
        if (now - self._last_snapshot).total_seconds() < 60:
            return
        self._last_snapshot = now
        unreal = sum(p.pnl() for p in self.positions.values())
        self.db.snapshot_equity(now, cfg.CAPITAL + self.realized + unreal, self.realized, unreal, len(self.positions))

    def status(self) -> str:
        parts = [f"equity=Rs{self.equity:,.0f} realized=Rs{self.realized:,.0f} open={len(self.positions)}"]
        for p in self.positions.values():
            parts.append(f"{p.symbol}:{p.setup}:Rs{p.pnl():+.0f}")
        return " | ".join(parts)
