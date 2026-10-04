"""Upstox market-data layer.

* Instrument-master download + resolution (keys, lot sizes, expiries) -- no hard-coded ISINs.
* WebSocket (MarketDataStreamerV3) tick ingestion aggregated into 5-minute OHLCV candles,
  with a wall-clock flusher and a REST polling fallback when the stream goes quiet.
* Option-chain access plus pure selection helpers that map predicted target prices to
  liquid contracts (directional ATM options and credit spreads).
"""
from __future__ import annotations

import bisect
import datetime as dt
import gzip
import io
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import requests
import upstox_client

import config as cfg

log = logging.getLogger(__name__)

CandleCallback = Callable[[str, "Candle"], None]
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/{exch}.json.gz"
WING_SEARCH_EXTRA = 2          # how many strikes farther out to look for a liquid protective wing
WING_MIN_PREMIUM = 0.05         # far-OTM hedges are legitimately cheap
STALE_STREAM_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 20.0
CLOSE_GRACE_SECONDS = 2.0


# =========================================================================== #
# Data classes
# =========================================================================== #
@dataclass(frozen=True)
class Candle:
    start: dt.datetime            # tz-aware IST bucket start
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def end(self) -> dt.datetime:
        return self.start + dt.timedelta(minutes=cfg.CANDLE_MINUTES)


@dataclass(frozen=True)
class Contract:
    instrument_key: str
    trading_symbol: str
    kind: str                     # "CE" | "PE"
    strike: float
    expiry: dt.date
    lot_size: int
    underlying_key: str
    tick_size: float = 0.05


@dataclass
class Underlying:
    spec: cfg.InstrumentSpec
    instrument_key: str
    lot_size: int
    contracts_by_expiry: dict[dt.date, list[Contract]]

    @property
    def expiries(self) -> list[dt.date]:
        return sorted(self.contracts_by_expiry)


@dataclass
class OptionQuote:
    instrument_key: str
    trading_symbol: str
    strike: float
    kind: str
    expiry: dt.date
    lot_size: int
    ltp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    oi: float = 0.0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.ltp

    @property
    def spread_pct(self) -> float:
        if self.bid > 0 and self.ask > 0 and self.mid > 0:
            return (self.ask - self.bid) / self.mid
        return 0.0                      # unknown depth: don't penalise

    def is_liquid(self, min_premium: float = cfg.MIN_OPTION_PREMIUM) -> bool:
        return (self.ltp >= min_premium
                and self.volume >= cfg.MIN_OPTION_VOLUME
                and self.spread_pct <= cfg.MAX_OPTION_SPREAD_PCT)


@dataclass
class OptionChain:
    symbol: str
    expiry: dt.date
    spot: float
    calls: dict[float, OptionQuote] = field(default_factory=dict)
    puts: dict[float, OptionQuote] = field(default_factory=dict)

    @property
    def strikes(self) -> list[float]:
        return sorted(set(self.calls) | set(self.puts))

    def side(self, kind: str) -> dict[float, OptionQuote]:
        return self.calls if kind == "CE" else self.puts


# =========================================================================== #
# Candle aggregation
# =========================================================================== #
def bucket_start(ts: dt.datetime) -> dt.datetime:
    """Floor ``ts`` to its 5-minute bucket (IST offset is a multiple of 5 min)."""
    step = cfg.CANDLE_MINUTES * 60
    epoch = math.floor(ts.timestamp() / step) * step
    return dt.datetime.fromtimestamp(epoch, cfg.IST)


class CandleBuilder:
    """Aggregates ticks into fixed 5-minute candles for one symbol. Thread-safe."""

    def __init__(self, symbol: str, maxlen: int = cfg.MAX_CANDLES_KEPT) -> None:
        self.symbol = symbol
        self._lock = threading.Lock()
        self._closed: deque[Candle] = deque(maxlen=maxlen)
        self._forming: Optional[list[Any]] = None   # [start, o, h, l, c, v]
        self.last_price: float = 0.0
        self.last_tick_at: float = 0.0

    # -- seeding / reading ------------------------------------------------- #
    def seed(self, candles: Iterable[Candle]) -> None:
        with self._lock:
            for c in sorted(candles, key=lambda x: x.start):
                if not self._closed or c.start > self._closed[-1].start:
                    self._closed.append(c)
            if self._closed:
                self.last_price = self._closed[-1].close

    def closed_candles(self) -> list[Candle]:
        with self._lock:
            return list(self._closed)

    def forming_candle(self) -> Optional[Candle]:
        with self._lock:
            f = self._forming
            return Candle(*f) if f else None

    # -- ingestion --------------------------------------------------------- #
    def on_tick(self, ts: dt.datetime, price: float, volume: float = 0.0) -> list[Candle]:
        """Feed one trade. Returns candles that closed as a consequence (0 or 1)."""
        if price <= 0:
            return []
        start = bucket_start(ts)
        closed: list[Candle] = []
        with self._lock:
            self.last_price = price
            self.last_tick_at = time.monotonic()
            last_closed_start = self._closed[-1].start if self._closed else None
            if last_closed_start is not None and start <= last_closed_start:
                return []                                   # late tick for a finished bar
            f = self._forming
            if f is not None and start < f[0]:
                return []
            if f is not None and start > f[0]:
                closed.append(self._close_locked())
                f = None
            if f is None:
                self._forming = [start, price, price, price, price, volume]
            else:
                f[2] = max(f[2], price)
                f[3] = min(f[3], price)
                f[4] = price
                f[5] += volume
        return closed

    def flush(self, now: dt.datetime) -> list[Candle]:
        """Close the forming bar once wall-clock passes its end (illiquid-tick safety)."""
        with self._lock:
            f = self._forming
            if f is None:
                return []
            end = f[0] + dt.timedelta(minutes=cfg.CANDLE_MINUTES)
            if now >= end + dt.timedelta(seconds=CLOSE_GRACE_SECONDS):
                return [self._close_locked()]
        return []

    def ingest_completed(self, candle: Candle) -> list[Candle]:
        """Accept an authoritative completed candle (REST polling path)."""
        with self._lock:
            if self._closed and candle.start <= self._closed[-1].start:
                return []
            if self._forming is not None and self._forming[0] <= candle.start:
                self._forming = None
            self._closed.append(candle)
            self.last_price = candle.close
            self.last_tick_at = time.monotonic()
        return [candle]

    def _close_locked(self) -> Candle:
        f = self._forming
        assert f is not None
        c = Candle(*f)
        self._closed.append(c)
        self._forming = None
        return c


# =========================================================================== #
# Instrument master
# =========================================================================== #
class InstrumentMaster:
    """Downloads Upstox's instrument master (cached per day) and resolves option universes."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._http = session or requests.Session()
        self._records: dict[str, list[dict[str, Any]]] = {}

    def _load_exchange(self, exch: str) -> list[dict[str, Any]]:
        if exch in self._records:
            return self._records[exch]
        cfg.CACHE_DIR.mkdir(exist_ok=True)
        cache = cfg.CACHE_DIR / f"{exch}-{cfg.now_ist():%Y%m%d}.json.gz"
        if cache.is_file():
            raw = cache.read_bytes()
        else:
            log.info("Downloading instrument master for %s", exch)
            resp = self._http.get(MASTER_URL.format(exch=exch), timeout=120)
            resp.raise_for_status()
            raw = resp.content
            cache.write_bytes(raw)
            for old in cfg.CACHE_DIR.glob(f"{exch}-*.json.gz"):
                if old != cache:
                    old.unlink(missing_ok=True)
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
            self._records[exch] = json.loads(fh.read().decode("utf-8"))
        return self._records[exch]

    def resolve(self, spec: cfg.InstrumentSpec, today: Optional[dt.date] = None) -> Underlying:
        today = today or cfg.now_ist().date()
        names = {n.upper() for n in spec.option_names}
        by_expiry: dict[dt.date, list[Contract]] = {}
        for r in self._load_exchange(spec.exchange):
            if r.get("instrument_type") not in ("CE", "PE"):
                continue
            und = str(r.get("underlying_symbol") or r.get("name") or "").upper()
            if und not in names:
                continue
            exp = _ms_to_date(r.get("expiry"))
            if exp is None or exp < today:
                continue
            by_expiry.setdefault(exp, []).append(Contract(
                instrument_key=r["instrument_key"],
                trading_symbol=r.get("trading_symbol", ""),
                kind=r["instrument_type"],
                strike=float(r.get("strike_price") or 0.0),
                expiry=exp,
                lot_size=int(r.get("lot_size") or spec.fallback_lot_size),
                underlying_key=r.get("underlying_key") or "",
                tick_size=float(r.get("tick_size") or 0.05) / (100 if (r.get("tick_size") or 0) >= 1 else 1),
            ))
        if not by_expiry:
            raise LookupError(f"No option contracts found in master for {spec.symbol} {sorted(names)}")
        nearest = min(by_expiry)
        first = by_expiry[nearest][0]
        key = first.underlying_key or spec.fallback_underlying_key
        if not key:
            raise LookupError(f"Cannot determine underlying key for {spec.symbol}")
        return Underlying(spec, key, first.lot_size, by_expiry)


def _ms_to_date(ms: Any) -> Optional[dt.date]:
    if ms in (None, "", 0):
        return None
    try:
        return dt.datetime.fromtimestamp(float(ms) / 1000.0, cfg.IST).date()
    except (TypeError, ValueError, OverflowError):
        return None


# =========================================================================== #
# Helpers to read SDK objects or plain dicts uniformly
# =========================================================================== #
def _g(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def parse_candle_rows(rows: Sequence[Sequence[Any]]) -> list[Candle]:
    """Upstox rows: [iso_ts, o, h, l, c, volume, oi] (newest first)."""
    out: list[Candle] = []
    for r in rows or []:
        try:
            ts = dt.datetime.fromisoformat(str(r[0])).astimezone(cfg.IST)
            out.append(Candle(ts, _f(r[1]), _f(r[2]), _f(r[3]), _f(r[4]), _f(r[5]) if len(r) > 5 else 0.0))
        except (ValueError, IndexError, TypeError):
            continue
    out.sort(key=lambda c: c.start)
    return out


# =========================================================================== #
# The feed
# =========================================================================== #
class UpstoxFeed:
    def __init__(
        self,
        api_client: upstox_client.ApiClient,
        underlyings: Sequence[Underlying],
        on_candle_close: CandleCallback,
        poll_only: bool = False,
    ) -> None:
        self.api = api_client
        self.on_candle_close = on_candle_close
        self.poll_only = poll_only
        self.underlyings = {u.spec.symbol: u for u in underlyings}
        self._key_to_symbol = {u.instrument_key: u.spec.symbol for u in underlyings}
        self.builders = {u.spec.symbol: CandleBuilder(u.spec.symbol) for u in underlyings}
        self._ltp: dict[str, tuple[float, float]] = {}        # key -> (price, monotonic ts)
        self._last_vtt: dict[str, float] = {}
        self._rest_tried: dict[str, float] = {}
        self._extra_keys: set[str] = set()
        self._streamer: Optional[upstox_client.MarketDataStreamerV3] = None
        self._stream_open = threading.Event()
        self._last_msg = 0.0
        self._last_poll = 0.0
        self._stop = threading.Event()
        self._watchdog: Optional[threading.Thread] = None
        self._history_api = upstox_client.HistoryV3Api(api_client)
        self._options_api = upstox_client.OptionsApi(api_client)
        self._quote_api = upstox_client.MarketQuoteV3Api(api_client)

    # ------------------------------------------------------------------ #
    # History bootstrap
    # ------------------------------------------------------------------ #
    def bootstrap_history(self) -> None:
        today = cfg.now_ist().date()
        frm = today - dt.timedelta(days=cfg.HISTORY_DAYS)
        for sym, u in self.underlyings.items():
            candles: dict[dt.datetime, Candle] = {}
            try:
                resp = self._call(self._history_api.get_historical_candle_data1,
                                  u.instrument_key, "minutes", cfg.CANDLE_MINUTES,
                                  today.isoformat(), frm.isoformat())
                for c in parse_candle_rows(_g(_g(resp, "data"), "candles", [])):
                    candles[c.start] = c
            except Exception as exc:                          # noqa: BLE001
                log.warning("%s: historical candles failed: %s", sym, exc)
            candles.update({c.start: c for c in self._fetch_intraday(u.instrument_key)})
            now = cfg.now_ist()
            done = [c for c in candles.values() if c.end <= now]
            self.builders[sym].seed(done)
            log.info("%s: seeded %d x 5m candles (key=%s)", sym, len(done), u.instrument_key)

    def _fetch_intraday(self, key: str) -> list[Candle]:
        try:
            resp = self._call(self._history_api.get_intra_day_candle_data, key, "minutes", cfg.CANDLE_MINUTES)
            return parse_candle_rows(_g(_g(resp, "data"), "candles", []))
        except Exception as exc:                              # noqa: BLE001
            log.debug("intraday fetch failed for %s: %s", key, exc)
            return []

    @staticmethod
    def _call(fn: Callable[..., Any], *args: Any, retries: int = 3) -> Any:
        delay = 1.0
        for attempt in range(retries):
            try:
                return fn(*args)
            except Exception as exc:                          # noqa: BLE001
                status = getattr(exc, "status", None)
                if attempt == retries - 1 or status in (400, 401, 403, 404):
                    raise
                time.sleep(delay)
                delay *= 2

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._stop.clear()
        if not self.poll_only:
            keys = list(self._key_to_symbol)
            self._streamer = upstox_client.MarketDataStreamerV3(self.api, keys, "full")
            self._streamer.on("open", self._on_open)
            self._streamer.on("message", self._on_message)
            self._streamer.on("error", lambda e: log.error("stream error: %s", e))
            self._streamer.on("close", lambda *a: log.warning("stream closed %s", a))
            self._streamer.on("reconnecting", lambda m: log.warning("stream: %s", m))
            self._streamer.connect()                          # spawns its own thread
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="feed-watchdog", daemon=True)
        self._watchdog.start()

    def stop(self) -> None:
        self._stop.set()
        if self._streamer is not None:
            try:
                self._streamer.auto_reconnect(False)
                self._streamer.disconnect()
            except Exception as exc:                          # noqa: BLE001
                log.debug("streamer disconnect: %s", exc)

    def _on_open(self) -> None:
        log.info("Market-data stream open")
        self._stream_open.set()
        if self._extra_keys:
            self.subscribe(list(self._extra_keys))

    def subscribe(self, keys: Iterable[str]) -> None:
        """Subscribe extra instruments (option legs) in LTPC mode."""
        keys = [k for k in keys if k]
        self._extra_keys.update(keys)
        if self._streamer is not None and self._stream_open.is_set():
            try:
                self._streamer.subscribe(keys, "ltpc")
            except Exception as exc:                          # noqa: BLE001
                log.warning("subscribe failed: %s", exc)

    def _on_message(self, msg: dict[str, Any]) -> None:
        try:
            self._last_msg = time.monotonic()
            feeds = msg.get("feeds") or {}
            for key, feed in feeds.items():
                ltpc, vtt = _extract_ltpc(feed)
                if not ltpc:
                    continue
                ltp = _f(ltpc.get("ltp"))
                if ltp <= 0:
                    continue
                self._ltp[key] = (ltp, time.monotonic())
                sym = self._key_to_symbol.get(key)
                if sym is None:
                    continue
                ts = _tick_time(ltpc.get("ltt"))
                if vtt is not None:
                    prev = self._last_vtt.get(key)
                    vol = max(0.0, vtt - prev) if prev is not None else 0.0
                    self._last_vtt[key] = vtt
                else:
                    vol = _f(ltpc.get("ltq"))
                for c in self.builders[sym].on_tick(ts, ltp, vol):
                    self._emit(sym, c)
        except Exception:                                      # noqa: BLE001
            log.exception("tick handler failed")

    def _emit(self, sym: str, candle: Candle) -> None:
        try:
            self.on_candle_close(sym, candle)
        except Exception:                                      # noqa: BLE001
            log.exception("candle callback failed for %s", sym)

    # ------------------------------------------------------------------ #
    # Watchdog: wall-clock flush + REST polling fallback
    # ------------------------------------------------------------------ #
    def _in_session(self, exch: str, now: dt.datetime) -> bool:
        if now.weekday() >= 5:
            return False
        s = cfg.SESSIONS[exch]
        t = now.time()
        return (dt.datetime.combine(now.date(), s.open) - dt.timedelta(minutes=2)).time() <= t <= \
            (dt.datetime.combine(now.date(), s.close) + dt.timedelta(minutes=10)).time()

    def _watchdog_loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                now = cfg.now_ist()
                for sym, u in self.underlyings.items():
                    if not self._in_session(u.spec.exchange, now):
                        continue
                    for c in self.builders[sym].flush(now):
                        self._emit(sym, c)
                stale = (time.monotonic() - self._last_msg) > STALE_STREAM_SECONDS
                if (self.poll_only or stale) and time.monotonic() - self._last_poll >= POLL_INTERVAL_SECONDS:
                    self._last_poll = time.monotonic()
                    self.poll_once(now)
            except Exception:                                  # noqa: BLE001
                log.exception("watchdog iteration failed")

    def poll_once(self, now: Optional[dt.datetime] = None) -> None:
        now = now or cfg.now_ist()
        for sym, u in self.underlyings.items():
            if not self._in_session(u.spec.exchange, now):
                continue
            rows = self._fetch_intraday(u.instrument_key)
            if rows:
                self._ltp[u.instrument_key] = (rows[-1].close, time.monotonic())
            for c in rows:
                if c.end <= now:
                    for done in self.builders[sym].ingest_completed(c):
                        self._emit(sym, done)

    # ------------------------------------------------------------------ #
    # Prices
    # ------------------------------------------------------------------ #
    def snapshot(self, symbol: str) -> list[Candle]:
        return self.builders[symbol].closed_candles()

    def last_price(self, key: str, max_age: float = 30.0) -> Optional[float]:
        """Latest streamed price for any subscribed key, falling back to a REST LTP quote."""
        hit = self._ltp.get(key)
        now = time.monotonic()
        if hit and now - hit[1] <= max_age:
            return hit[0]
        if now - self._rest_tried.get(key, -1e9) >= 5.0:      # throttle REST fallback
            self._rest_tried[key] = now
            got = self.rest_ltp([key]).get(key)
            if got:
                return got
        return hit[0] if hit else None

    def rest_ltp(self, keys: Sequence[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        if not keys:
            return out
        try:
            resp = self._call(lambda k: self._quote_api.get_ltp(instrument_key=k), ",".join(keys))
        except Exception as exc:                              # noqa: BLE001
            log.warning("REST LTP failed: %s", exc)
            return out
        for _, q in (_g(resp, "data") or {}).items():
            tok, px = _g(q, "instrument_token"), _f(_g(q, "last_price"))
            if tok and px > 0:
                out[tok] = px
                self._ltp[tok] = (px, time.monotonic())
        return out

    def underlying_price(self, symbol: str) -> Optional[float]:
        u = self.underlyings[symbol]
        px = self.last_price(u.instrument_key)
        return px or (self.builders[symbol].last_price or None)

    def health(self) -> dict:
        """Per-symbol feed diagnostics for the dashboard."""
        now = time.monotonic()
        syms = []
        for sym, b in self.builders.items():
            syms.append({"symbol": sym, "price": b.last_price or None, "candles": len(b.closed_candles()),
                         "tick_age_s": (now - b.last_tick_at) if b.last_tick_at else None})
        return {"stream_open": self._stream_open.is_set(), "poll_only": self.poll_only,
                "msg_age_s": (now - self._last_msg) if self._last_msg else None, "symbols": syms}

    # ------------------------------------------------------------------ #
    # Calendar helpers
    # ------------------------------------------------------------------ #
    def expiry_dates(self, symbol: str) -> list[dt.date]:
        return self.underlyings[symbol].expiries

    def is_expiry_today(self, symbol: str, today: Optional[dt.date] = None) -> bool:
        today = today or cfg.now_ist().date()
        return today in self.underlyings[symbol].contracts_by_expiry

    def pick_expiry(self, symbol: str, today: Optional[dt.date] = None) -> Optional[dt.date]:
        today = today or cfg.now_ist().date()
        future = [e for e in self.underlyings[symbol].expiries if e >= today]
        return future[0] if future else None

    # ------------------------------------------------------------------ #
    # Option chain
    # ------------------------------------------------------------------ #
    def get_option_chain(self, symbol: str, expiry: dt.date) -> Optional[OptionChain]:
        """Live chain (LTP, bid/ask, volume, OI, Greeks). Falls back to master + quote API."""
        u = self.underlyings[symbol]
        contracts = u.contracts_by_expiry.get(expiry, [])
        if not contracts:
            return None
        und_key = contracts[0].underlying_key or u.instrument_key
        by_key = {c.instrument_key: c for c in contracts}
        try:
            resp = self._call(self._options_api.get_put_call_option_chain, und_key, expiry.isoformat())
            chain = self._build_chain(symbol, expiry, _g(resp, "data") or [], by_key, u.lot_size)
            if chain and (chain.calls or chain.puts):
                return chain
        except Exception as exc:                              # noqa: BLE001
            log.warning("%s: option-chain endpoint failed (%s); using quote fallback", symbol, exc)
        return self._chain_from_quotes(symbol, expiry, contracts, u)

    @staticmethod
    def _quote_from(leg: Any, kind: str, strike: float, expiry: dt.date,
                    by_key: dict[str, Contract], lot: int) -> Optional[OptionQuote]:
        if leg is None:
            return None
        key = _g(leg, "instrument_key")
        md, gk = _g(leg, "market_data"), _g(leg, "option_greeks")
        c = by_key.get(key)
        return OptionQuote(
            instrument_key=key, trading_symbol=c.trading_symbol if c else "", strike=strike, kind=kind,
            expiry=expiry, lot_size=c.lot_size if c else lot,
            ltp=_f(_g(md, "ltp")), bid=_f(_g(md, "bid_price")), ask=_f(_g(md, "ask_price")),
            volume=_f(_g(md, "volume")), oi=_f(_g(md, "oi")),
            delta=_opt(_g(gk, "delta")), gamma=_opt(_g(gk, "gamma")), theta=_opt(_g(gk, "theta")),
            vega=_opt(_g(gk, "vega")), iv=_opt(_g(gk, "iv")),
        )

    def _build_chain(self, symbol: str, expiry: dt.date, rows: Sequence[Any],
                     by_key: dict[str, Contract], lot: int) -> Optional[OptionChain]:
        spot = 0.0
        chain = OptionChain(symbol, expiry, 0.0)
        for row in rows:
            strike = _f(_g(row, "strike_price"))
            spot = spot or _f(_g(row, "underlying_spot_price"))
            ce = self._quote_from(_g(row, "call_options"), "CE", strike, expiry, by_key, lot)
            pe = self._quote_from(_g(row, "put_options"), "PE", strike, expiry, by_key, lot)
            if ce and ce.ltp > 0:
                chain.calls[strike] = ce
            if pe and pe.ltp > 0:
                chain.puts[strike] = pe
        chain.spot = spot or (self.underlying_price(symbol) or 0.0)
        return chain

    def _chain_from_quotes(self, symbol: str, expiry: dt.date, contracts: Sequence[Contract],
                           u: Underlying) -> Optional[OptionChain]:
        spot = self.underlying_price(symbol)
        if not spot:
            return None
        strikes = sorted({c.strike for c in contracts})
        i = bisect.bisect_left(strikes, spot)
        near = set(strikes[max(0, i - 12): i + 12])
        picked = [c for c in contracts if c.strike in near]
        chain = OptionChain(symbol, expiry, spot)
        for i0 in range(0, len(picked), 100):
            batch = picked[i0:i0 + 100]
            try:
                resp = self._call(lambda k: self._quote_api.get_market_quote_option_greek(instrument_key=k),
                                  ",".join(c.instrument_key for c in batch))
            except Exception as exc:                          # noqa: BLE001
                log.error("%s: option-greek quote fallback failed: %s", symbol, exc)
                return None
            by_key = {c.instrument_key: c for c in batch}
            for _, q in (_g(resp, "data") or {}).items():
                c = by_key.get(_g(q, "instrument_token"))
                if c is None or _f(_g(q, "last_price")) <= 0:
                    continue
                oq = OptionQuote(c.instrument_key, c.trading_symbol, c.strike, c.kind, expiry, c.lot_size,
                                 ltp=_f(_g(q, "last_price")), volume=_f(_g(q, "volume")), oi=_f(_g(q, "oi")),
                                 delta=_opt(_g(q, "delta")), gamma=_opt(_g(q, "gamma")),
                                 theta=_opt(_g(q, "theta")), vega=_opt(_g(q, "vega")), iv=_opt(_g(q, "iv")))
                chain.side(c.kind)[c.strike] = oq
        return chain if (chain.calls or chain.puts) else None


# =========================================================================== #
# Pure selection helpers (unit-testable, no network)
# =========================================================================== #
def select_directional_option(chain: OptionChain, direction: int, target_price: Optional[float] = None
                              ) -> Optional[OptionQuote]:
    """Pick the liquid CE (direction=+1) / PE (-1) with |delta| nearest 0.45 (inside 0.40-0.50).

    If Greeks are unavailable the nearest-ATM liquid strike is used. When several strikes
    qualify, the one with the best payoff-per-rupee at ``target_price`` wins.
    """
    kind = "CE" if direction > 0 else "PE"
    liquid = [q for q in chain.side(kind).values() if q.is_liquid()]
    if not liquid:
        return None
    with_delta = [q for q in liquid if q.delta is not None and q.delta != 0]
    if with_delta:
        lo, hi = cfg.TARGET_DELTA_RANGE
        in_band = [q for q in with_delta if lo <= abs(q.delta) <= hi]
        if in_band and target_price:
            def payoff(q: OptionQuote) -> float:
                intrinsic = max(0.0, target_price - q.strike) if kind == "CE" else max(0.0, q.strike - target_price)
                return (intrinsic - q.ltp) / q.ltp
            return max(in_band, key=payoff)
        pool = in_band or with_delta
        return min(pool, key=lambda q: abs(abs(q.delta) - cfg.TARGET_DELTA))
    return min(liquid, key=lambda q: abs(q.strike - chain.spot))


def select_credit_spread(chain: OptionChain, kind: str, boundary: float,
                         width_strikes: int = cfg.SPREAD_WIDTH_STRIKES
                         ) -> Optional[tuple[OptionQuote, OptionQuote]]:
    """Credit spread whose short strike lies *outside* the predicted boundary.

    kind="PE": bull put spread, short strike = highest liquid strike <= boundary (q10).
    kind="CE": bear call spread, short strike = lowest liquid strike >= boundary (q90).
    Returns (short_leg, long_leg); the long leg is ``width_strikes`` further out of the money.
    """
    side = chain.side(kind)
    strikes = sorted(side)
    if len(strikes) < width_strikes + 1:
        return None
    if kind == "PE":
        cand = [k for k in strikes if k <= boundary and side[k].is_liquid()]
        if not cand:
            return None
        short_k = max(cand)
        direction = -1
    else:
        cand = [k for k in strikes if k >= boundary and side[k].is_liquid()]
        if not cand:
            return None
        short_k = min(cand)
        direction = 1
    # Protective wing: `width_strikes` away, or further out (up to 2 extra strikes) until it is liquid.
    base = strikes.index(short_k)
    long_k = None
    for extra in range(WING_SEARCH_EXTRA + 1):
        idx = base + direction * (width_strikes + extra)
        if idx < 0 or idx >= len(strikes):
            break
        if side[strikes[idx]].is_liquid(min_premium=WING_MIN_PREMIUM):
            long_k = strikes[idx]
            break
    if long_k is None:
        return None
    short_q, long_q = side[short_k], side[long_k]
    if long_q.ltp <= 0 or short_q.ltp <= long_q.ltp:
        return None
    return short_q, long_q


# =========================================================================== #
# Message parsing helpers
# =========================================================================== #
def _opt(x: Any) -> Optional[float]:
    if x is None:
        return None
    v = _f(x, float("nan"))
    return None if math.isnan(v) else v


def _extract_ltpc(feed: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[float]]:
    """Return (ltpc dict, cumulative traded volume or None) from any V3 feed flavour."""
    if "ltpc" in feed:
        return feed["ltpc"], None
    full = feed.get("fullFeed") or {}
    for flavour in ("marketFF", "indexFF"):
        ff = full.get(flavour)
        if ff:
            vtt = ff.get("vtt")
            return ff.get("ltpc"), (_f(vtt) if vtt is not None else None)
    first = feed.get("firstLevelWithGreeks")
    if first:
        return first.get("ltpc"), None
    return None, None


def _tick_time(ltt: Any) -> dt.datetime:
    """``ltt`` is epoch milliseconds (string/int); fall back to now if absent/implausible."""
    try:
        ts = dt.datetime.fromtimestamp(float(ltt) / 1000.0, cfg.IST)
        if abs((cfg.now_ist() - ts).total_seconds()) < 3600:
            return ts
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return cfg.now_ist()
