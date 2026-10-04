"""Offline tests: no network, no GPU, no model weights."""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from features import (ExpiryCalendar, anchored_log_prices, atr, build_features, parkinson_vol,  # noqa: E402
                      session_covariates)
from model_worker import InferenceWorker, MockBackend, Prediction, make_prediction  # noqa: E402
from strategy_engine import (SETUP_CONDOR, SETUP_MOMENTUM, StrategyEngine, TradeDB, evaluate_signal,  # noqa: E402
                             size_condor, size_long_option)
from upstox_feed import (Candle, CandleBuilder, OptionChain, OptionQuote, Underlying,  # noqa: E402
                         bucket_start, parse_candle_rows, select_credit_spread, select_directional_option)

IST = cfg.IST


def at(h: int, m: int = 0, day: int = 6) -> dt.datetime:       # 2026-10-06 is a Tuesday
    return dt.datetime(2026, 10, day, h, m, tzinfo=IST)


# ------------------------------------------------------------------ config
def test_token_loader(tmp_path):
    with pytest.raises(cfg.ConfigError, match="not found"):
        cfg.load_access_token(tmp_path / "nope.txt")
    f = tmp_path / "upstox.txt"
    f.write_text("  \n")
    with pytest.raises(cfg.ConfigError, match="empty"):
        cfg.load_access_token(f)
    f.write_text("abc.def.ghi \r\n\n")
    assert cfg.load_access_token(f) == "abc.def.ghi"
    assert cfg.build_configuration("tok").access_token == "tok"


# ------------------------------------------------------------------ candles
def test_candle_builder_buckets_and_flush():
    b = CandleBuilder("X")
    assert bucket_start(at(9, 17)) == at(9, 15)
    assert b.on_tick(at(9, 15), 100, 10) == []
    assert b.on_tick(at(9, 17), 103, 5) == []
    assert b.on_tick(at(9, 19), 99, 5) == []
    closed = b.on_tick(at(9, 20), 101, 1)
    assert len(closed) == 1
    c = closed[0]
    assert (c.start, c.open, c.high, c.low, c.close, c.volume) == (at(9, 15), 100, 103, 99, 99, 20)
    assert b.on_tick(at(9, 16), 500, 1) == []                      # late tick ignored
    assert b.flush(at(9, 24)) == []
    flushed = b.flush(at(9, 25, ) + dt.timedelta(seconds=3))
    assert len(flushed) == 1 and flushed[0].start == at(9, 20)
    assert b.flush(at(9, 40)) == []


def test_parse_candle_rows():
    rows = [["2026-10-06T09:20:00+05:30", 2, 3, 1, 2.5, 100, 0], ["2026-10-06T09:15:00+05:30", 1, 2, 1, 2, 50, 0]]
    out = parse_candle_rows(rows)
    assert [c.start for c in out] == [at(9, 15), at(9, 20)]


# ------------------------------------------------------------------ features
def make_candles(n_sessions=3, per=30, gap=0.05):
    out, price = [], 100.0
    for s in range(n_sessions):
        day = at(9, 15, day=1 + s)
        price *= 1 + gap                                           # overnight gap
        for i in range(per):
            o = price
            price *= 1 + 0.0005 * (1 if i % 3 else -1)
            out.append(Candle(day + dt.timedelta(minutes=5 * i), o, max(o, price) * 1.0005,
                              min(o, price) * 0.9995, price, 1000))
    return out


def test_gap_shielding_and_anchor():
    cs = make_candles(gap=0.05)
    z = anchored_log_prices(cs)
    assert z[-1] == pytest.approx(0.0)
    # 5 % overnight gaps must not appear: total path range stays far below ln(1.05)*2
    assert np.abs(np.diff(z)).max() < 0.01
    raw = np.log(np.array([c.close for c in cs]))
    assert np.abs(np.diff(raw)).max() > 0.04                       # sanity: raw series does have the gaps


def test_parkinson_and_atr():
    hi, lo = np.full(30, 101.0), np.full(30, 99.0)
    pv = parkinson_vol(hi, lo, 12)
    expected = np.log(101 / 99) / np.sqrt(4 * np.log(2))
    assert pv[-1] == pytest.approx(expected)
    assert atr(make_candles(), 20) > 0


def test_covariates():
    t = [at(9, 15), at(15, 30), at(18, 30), at(23, 30)]
    nse = session_covariates(t[:2], "NSE", lambda d: False)
    assert nse[0, 0] == 0.0 and nse[0, 1] == 1.0 and nse[1].sum() == 0
    mcx = session_covariates(t[1:], "MCX", lambda d: d == t[0].date())
    assert mcx[1, 1] == 1.0 and mcx[1, 0] == 0.0 and 0.0 <= mcx[0].min() and mcx[0].max() <= 1.0
    assert mcx[2].all()


def test_build_features_shapes():
    cs = make_candles(per=75)
    cal = ExpiryCalendar(frozenset({cs[-1].start.date()}), 1, today=cs[-1].start.date())
    fb = build_features("NIFTY", "NSE", cs, cal, horizon=12)
    n = len(fb.target)
    assert fb.past_only.shape == (1, n) and fb.past_future.shape == (3, n + 12)
    assert fb.target[-1] == 0 and fb.target.dtype == np.float32
    assert fb.spot_is_expiry and fb.past_future[2, -1] == 1.0
    assert build_features("NIFTY", "NSE", cs[:10], cal) is None


# ------------------------------------------------------------------ chains
def make_chain(spot=100.0, strikes=range(80, 125, 5), expiry=dt.date(2026, 10, 8), lot=100) -> OptionChain:
    ch = OptionChain("TEST", expiry, spot)
    for k in strikes:
        for kind, store in (("CE", ch.calls), ("PE", ch.puts)):
            intrinsic = max(0, spot - k) if kind == "CE" else max(0, k - spot)
            time_val = 6.0 * np.exp(-((k - spot) / 12.0) ** 2) + 0.3
            ltp = round(intrinsic + time_val, 2)
            delta = float(np.clip(0.5 + (spot - k) / 40.0, 0.02, 0.98)) * (1 if kind == "CE" else -1)
            if kind == "PE":
                delta = float(-np.clip(0.5 + (k - spot) / 40.0, 0.02, 0.98))
            store[float(k)] = OptionQuote(f"{kind}{k}", f"TEST {k} {kind}", float(k), kind, expiry, lot, ltp=ltp,
                                          bid=ltp * 0.995, ask=ltp * 1.005, volume=1000, oi=5000, delta=delta)
    return ch


def test_select_directional_option():
    ch = make_chain()
    call = select_directional_option(ch, +1, target_price=110)
    put = select_directional_option(ch, -1, target_price=90)
    assert call.kind == "CE" and 0.40 <= call.delta <= 0.50
    assert put.kind == "PE" and 0.40 <= abs(put.delta) <= 0.50
    for q in ch.calls.values():
        q.volume = 0
    assert select_directional_option(ch, +1) is None               # illiquid -> nothing


def test_select_credit_spread_outside_boundary():
    ch = make_chain()
    short, long_ = select_credit_spread(ch, "PE", boundary=92.0, width_strikes=2)
    assert short.strike == 90 and long_.strike == 80 and short.strike <= 92
    short, long_ = select_credit_spread(ch, "CE", boundary=108.0, width_strikes=2)
    assert short.strike == 110 and long_.strike == 120 and short.strike >= 108
    assert select_credit_spread(ch, "PE", boundary=70.0) is None


# ------------------------------------------------------------------ sizing / signals
def test_long_option_sizing_respects_risk_and_margin():
    q = OptionQuote("k", "s", 100, "CE", dt.date(2026, 10, 8), 100, ltp=5.0, bid=4.98, ask=5.02, volume=500, delta=0.45)
    s = size_long_option(q, spot=100, stop_underlying=95, risk_budget=3000, margin_available=60000)
    assert s and s.lots >= 1 and s.risk_at_stop <= 3000 and s.margin <= 60000
    assert size_long_option(q, 100, 95, 3000, margin_available=100) is None       # margin too small
    big = OptionQuote("k", "s", 100, "CE", dt.date(2026, 10, 8), 5000, ltp=50.0, volume=500, delta=0.45)
    assert size_long_option(big, 100, 80, 3000, 60000) is None                    # 1 lot > Rs 3,000 risk


def test_condor_sizing_bounds_max_loss():
    ch = make_chain()
    ps, pl = select_credit_spread(ch, "PE", 92.0)
    cs, cl = select_credit_spread(ch, "CE", 108.0)
    s = size_condor(ps, pl, cs, cl, risk_budget=30000, margin_available=60000)
    assert s and s.risk_at_stop <= 30000 and s.margin <= 60000
    assert size_condor(ps, pl, cs, cl, risk_budget=10, margin_available=60000) is None


def pred(**kw) -> Prediction:
    base = dict(symbol="TEST", exchange="NSE", bar_start=at(10, 0), anchor_price=100.0, horizon=12, q10=96.0,
                q50=103.0, q90=108.0, expected_drift=3.0, expected_drift_pct=0.03, vol_spread=12.0, spread_pct=.12,
                atr20=1.0, parkinson_now=0.001, is_expiry_day=False, price_quantiles=np.zeros((12, 9)))
    base.update(kw)
    return Prediction(**base)


def test_signal_rules():
    s = evaluate_signal(pred(), spread_history=[8.0, 9.0])
    assert s and s.setup == SETUP_MOMENTUM and s.direction == 1 and s.stop <= 96.0
    assert evaluate_signal(pred(), spread_history=[12.0, 12.0]) is None               # envelope not expanding
    assert evaluate_signal(pred(), spread_history=[]) is None
    bear = evaluate_signal(pred(q10=92, q50=97, q90=104, expected_drift=-3.0), [8.0])
    assert bear and bear.direction == -1 and bear.stop >= 104
    tight = pred(q10=99.5, q50=100.1, q90=101.0, expected_drift=0.1, vol_spread=1.5, atr20=1.0)
    c = evaluate_signal(tight, [1.5])
    assert c and c.setup == SETUP_CONDOR and (c.stop, c.target) == (99.5, 101.0)


# ------------------------------------------------------------------ engine (fake feed + clock)
class FakeFeed:
    def __init__(self, chain: OptionChain, lot=100):
        self.chain = chain
        spec = cfg.InstrumentSpec("TEST", cfg.CATEGORY_STOCK, "NSE", ("TEST",), None, lot)
        self.underlyings = {"TEST": Underlying(spec, "NSE_EQ|TEST", lot, {chain.expiry: []})}
        self.spot = 100.0
        self.prices: dict[str, float] = {}
        self.subscribed: set[str] = set()

    def underlying_price(self, symbol): return self.spot
    def last_price(self, key, max_age=30.0): return self.prices.get(key)
    def pick_expiry(self, symbol, today=None): return self.chain.expiry
    def get_option_chain(self, symbol, expiry): return self.chain
    def subscribe(self, keys): self.subscribed.update(keys)

    def reprice(self, chain_spot: float):
        self.spot = chain_spot
        for q in list(self.chain.calls.values()) + list(self.chain.puts.values()):
            intrinsic = max(0, chain_spot - q.strike) if q.kind == "CE" else max(0, q.strike - chain_spot)
            self.prices[q.instrument_key] = round(intrinsic + 0.3 + 6.0 * np.exp(-((q.strike - chain_spot) / 12.0) ** 2), 2)


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


@pytest.fixture()
def rig(tmp_path):
    feed = FakeFeed(make_chain())
    feed.reprice(100.0)
    clock = Clock(at(10, 5))
    db = TradeDB(tmp_path / "t.db")
    eng = StrategyEngine(feed, db, clock)
    yield feed, clock, db, eng
    db.close()


def test_momentum_trade_lifecycle_and_sizing(rig):
    feed, clock, db, eng = rig
    eng.handle_prediction(pred(vol_spread=8.0, q10=98.0, q90=106.0, q50=102.0, expected_drift=2.0))   # seeds history
    assert not eng.positions
    eng.handle_prediction(pred(bar_start=at(10, 0)))
    pos = eng.positions["TEST"]
    assert pos.setup == SETUP_MOMENTUM and pos.legs[0].kind == "CE"
    assert pos.risk_at_stop <= cfg.MAX_RISK_PER_TRADE * cfg.CAPITAL
    assert pos.margin <= cfg.MAX_MARGIN_PER_CATEGORY * cfg.CAPITAL
    assert pos.legs[0].entry_px == pytest.approx(pos.legs[0].last_px * (1 + cfg.SLIPPAGE), abs=0.01)
    assert pos.legs[0].instrument_key in feed.subscribed
    # price runs to the stop -> exits at a loss no worse than the cap (+ slippage/brokerage)
    clock.t = at(10, 20)
    feed.reprice(pos.stop_underlying - 0.1)
    eng.manage_positions()
    assert not eng.positions
    row = db.query("SELECT status, exit_reason, net_pnl FROM trades")[0]
    assert row[0] == "CLOSED" and row[1] in ("STOP_UNDERLYING", "RISK_CAP") and row[2] < 0
    assert abs(row[2]) <= cfg.MAX_RISK_PER_TRADE * cfg.CAPITAL * 1.1
    assert db.query("SELECT COUNT(*) FROM fills")[0][0] == 2
    assert eng.realized == pytest.approx(row[2])


def test_momentum_target_and_trailing(rig):
    feed, clock, db, eng = rig
    eng.handle_prediction(pred(vol_spread=8.0, q10=98.0, q90=106.0, q50=102.0, expected_drift=2.0))
    eng.handle_prediction(pred())
    pos = eng.positions["TEST"]
    initial_stop = pos.stop_underlying
    clock.t = at(10, 15)
    feed.reprice(pos.entry_underlying + 0.6 * (pos.target_underlying - pos.entry_underlying))
    eng.manage_positions()
    assert "TEST" in eng.positions and pos.stop_underlying > initial_stop                    # open, stop ratcheted up
    feed.reprice(pos.entry_underlying + 0.3)                                                 # pullback
    eng.manage_positions()
    assert "TEST" not in eng.positions
    reason = db.query("SELECT exit_reason FROM trades")[0][0]
    assert reason in ("TRAILING_STOP", "STOP_UNDERLYING")


def test_cold_start_and_entry_window(rig):
    feed, clock, db, eng = rig
    clock.t = at(9, 20)
    eng.handle_prediction(pred(bar_start=at(9, 15), vol_spread=8.0))
    eng.handle_prediction(pred(bar_start=at(9, 15)))
    assert not eng.positions                                                                  # before 09:30
    clock.t = at(14, 50)
    eng.handle_prediction(pred(bar_start=at(14, 45)))
    assert not eng.positions                                                                  # after last-entry cut-off


def test_eod_square_off_nse(rig):
    feed, clock, db, eng = rig
    eng.handle_prediction(pred(vol_spread=8.0))
    eng.handle_prediction(pred())
    assert eng.positions
    eng.positions["TEST"].max_hold = dt.timedelta(days=1)          # isolate the square-off rule from the time stop
    clock.t = at(15, 14)
    eng.manage_positions()
    assert eng.positions
    clock.t = at(15, 15)
    eng.manage_positions()
    assert not eng.positions
    assert db.query("SELECT exit_reason FROM trades")[0][0] == "EOD_SQUARE_OFF"
    assert db.query("SELECT COUNT(*) FROM equity")[0][0] >= 0


def test_condor_entry_and_boundary_stop(rig):
    feed, clock, db, eng = rig
    tight = dict(q10=92.0, q50=100.1, q90=108.0, expected_drift=0.1, vol_spread=2.0, atr20=1.0)
    eng.handle_prediction(pred(**tight))
    eng.handle_prediction(pred(**tight))
    pos = eng.positions["TEST"]
    assert pos.setup == SETUP_CONDOR and len(pos.legs) == 4
    shorts = [l for l in pos.legs if l.side < 0]
    assert any(l.kind == "PE" and l.strike <= 92 for l in shorts) and any(l.kind == "CE" and l.strike >= 108 for l in shorts)
    assert pos.risk_at_stop <= cfg.MAX_RISK_PER_TRADE * cfg.CAPITAL
    clock.t = at(11, 0)
    feed.reprice(91.0)
    eng.manage_positions()
    assert not eng.positions
    assert db.query("SELECT exit_reason FROM trades")[0][0] in ("STOP_BOUNDARY", "RISK_CAP")


def test_mcx_square_off_time(rig):
    feed, clock, db, eng = rig
    eng.positions.clear()
    spec = cfg.InstrumentSpec("TEST", cfg.CATEGORY_MCX, "MCX", ("TEST",), None, 100)
    eng.specs["TEST"] = spec
    clock.t = at(22, 0)
    eng.handle_prediction(pred(exchange="MCX", bar_start=at(21, 55), vol_spread=8.0))
    eng.handle_prediction(pred(exchange="MCX", bar_start=at(21, 55)))
    assert eng.positions
    eng.positions["TEST"].max_hold = dt.timedelta(days=1)
    clock.t = at(23, 14)
    eng.manage_positions()
    assert eng.positions
    clock.t = at(23, 15)
    eng.manage_positions()
    assert not eng.positions


# ------------------------------------------------------------------ worker thread
def test_inference_worker_async_roundtrip():
    cs = make_candles(per=75)
    cal = ExpiryCalendar(frozenset(), None)
    got: list[Prediction] = []
    w = InferenceWorker(MockBackend(), got.append, collect_s=0.05)
    w.start()
    t0 = time.perf_counter()
    for sym in ("A", "B", "C"):
        w.submit(build_features(sym, "NSE", cs, cal))
    assert time.perf_counter() - t0 < 0.5                          # submit never blocks on inference
    for _ in range(100):
        if len(got) == 3:
            break
        time.sleep(0.05)
    w.stop()
    assert {p.symbol for p in got} == {"A", "B", "C"}
    p = got[0]
    assert p.q10 <= p.q50 <= p.q90 and p.price_quantiles.shape == (12, 9)
    assert p.vol_spread == pytest.approx(p.q90 - p.q10)


def test_make_prediction_maps_z_to_price():
    cs = make_candles(per=75)
    fb = build_features("A", "NSE", cs, ExpiryCalendar(frozenset(), None))
    z = np.tile(np.linspace(-0.01, 0.01, 9), (12, 1))
    p = make_prediction(fb, z, 0.1, "t")
    assert p.q50 == pytest.approx(fb.anchor_price)
    assert p.q90 == pytest.approx(fb.anchor_price * np.exp(0.01))


# ------------------------------------------------------------------ dashboard
def test_dashboard_state_and_http(rig):
    import json
    import urllib.request
    from dashboard import DashboardServer, build_state

    feed, clock, db, eng = rig
    feed.health = lambda: {"stream_open": True, "poll_only": False, "msg_age_s": 1.0,
                           "symbols": [{"symbol": "TEST", "price": 100.0, "candles": 80, "tick_age_s": 1.0}]}
    eng.handle_prediction(pred(vol_spread=8.0))
    eng.handle_prediction(pred())
    clock.t = at(10, 20)
    feed.reprice(eng.positions["TEST"].stop_underlying - 0.1)
    eng.manage_positions()                                          # one closed trade
    eng.handle_prediction(pred(bar_start=at(10, 15)))
    st = build_state(eng, feed, db, "mock")
    assert st["trades"][0]["status"] == "CLOSED" and st["predictions"][0]["symbol"] == "TEST"
    assert st["sessions"]["NSE"]["state"] == "trading" and st["today"]["closed"] == 1
    srv = DashboardServer(eng, feed, db, "mock", port=0)
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv._httpd.server_address[1]}"
        assert b"Paper Trading Monitor" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/state").read())["backend"] == "mock"
    finally:
        srv.stop()
