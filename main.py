"""Orchestrator: Upstox feed -> features -> TimesFM-3 worker thread -> strategy engine (paper trades)."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import datetime as dt
from typing import Optional

import config as cfg
from dashboard import DashboardServer
from features import ExpiryCalendar, build_features
from model_worker import Backend, InferenceWorker, MockBackend, TimesFMBackend
from strategy_engine import StrategyEngine, TradeDB
from upstox_feed import Candle, InstrumentMaster, UpstoxFeed

log = logging.getLogger("main")


class Bot:
    def __init__(self, opts: cfg.RuntimeOptions) -> None:
        self.opts = opts
        self.stop_evt = threading.Event()

        token = cfg.load_access_token()                       # fails fast with a clear message
        api = cfg.build_api_client(token)

        master = InstrumentMaster()
        underlyings = []
        for spec in cfg.UNIVERSE:
            if spec.symbol not in opts.symbols:
                continue
            try:
                underlyings.append(master.resolve(spec))
            except Exception as exc:                          # noqa: BLE001
                log.error("Skipping %s: %s", spec.symbol, exc)
        if not underlyings:
            raise SystemExit("No tradable instruments could be resolved from the instrument master.")
        self.calendars = {u.spec.symbol: ExpiryCalendar(frozenset(u.expiries), u.spec.weekly_expiry_weekday)
                          for u in underlyings}

        self.db = TradeDB()
        n = self.db.abandon_open_trades()
        if n:
            log.warning("%d trade(s) from a previous run were still OPEN in the DB; marked ABANDONED", n)

        self.feed = UpstoxFeed(api, underlyings, self.on_candle_close, poll_only=opts.poll_only)
        self.engine = StrategyEngine(self.feed, self.db)
        backend: Backend = MockBackend() if opts.mock_model else TimesFMBackend()
        if opts.mock_model:
            log.warning("MOCK model backend in use - signals are NOT from TimesFM-3")
        self.worker = InferenceWorker(backend, self.engine.on_prediction)
        self.dashboard: Optional[DashboardServer] = None
        if opts.dashboard:
            self.dashboard = DashboardServer(self.engine, self.feed, self.db, backend.name,
                                             opts.dashboard_host, opts.dashboard_port)

    # Called on the WebSocket / watchdog thread: must stay cheap and non-blocking.
    def on_candle_close(self, symbol: str, candle: Candle) -> None:
        if cfg.now_ist() - candle.end > dt.timedelta(minutes=cfg.CANDLE_MINUTES * 3):
            return                                            # backfilled bar, not a live close
        self.submit_features(symbol)

    def submit_features(self, symbol: str) -> None:
        u = self.feed.underlyings[symbol]
        bundle = build_features(symbol, u.spec.exchange, self.feed.snapshot(symbol), self.calendars[symbol])
        if bundle is None:
            log.debug("%s: not enough history for inference yet", symbol)
            return
        self.worker.submit(bundle)

    def run(self) -> None:
        if self.dashboard:
            self.dashboard.start()
        self.worker.start()
        self.engine.start()
        self.feed.bootstrap_history()
        for sym in self.feed.underlyings:                     # warm the envelope history
            self.submit_features(sym)
        self.feed.start()
        log.info("Bot running. Universe: %s", ", ".join(self.feed.underlyings))
        while not self.stop_evt.wait(60.0):
            log.info(self.engine.status())
            if cfg.now_ist().time() >= cfg.BOT_SHUTDOWN_TIME and not self.engine.positions:
                log.info("Session over; shutting down")
                break
        self.shutdown()

    def shutdown(self) -> None:
        self.feed.stop()
        self.engine.stop()
        self.engine.join(timeout=5)
        if self.opts.flatten_on_exit and self.engine.positions:
            log.warning("Flattening open paper positions before exit (no overnight holding)")
            self.engine.flatten_all("SHUTDOWN")
        self.worker.stop()
        if self.dashboard:
            self.dashboard.stop()
        log.info("Final: %s", self.engine.status())
        self.db.close()


def parse_args(argv: Optional[list[str]] = None) -> cfg.RuntimeOptions:
    p = argparse.ArgumentParser(description="TimesFM-3 intraday paper-trading bot (Upstox, NSE/MCX)")
    p.add_argument("--mock-model", action="store_true", help="use a dummy forecaster instead of TimesFM-3 (wiring tests)")
    p.add_argument("--poll-only", action="store_true", help="REST-poll candles instead of the WebSocket")
    p.add_argument("--symbols", nargs="+", default=[s.symbol for s in cfg.UNIVERSE],
                   help="subset of: " + " ".join(s.symbol for s in cfg.UNIVERSE))
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--no-dashboard", action="store_true", help="do not start the monitoring web UI")
    p.add_argument("--dashboard-host", default="127.0.0.1",
                   help="bind address (default localhost; the UI has no authentication)")
    p.add_argument("--dashboard-port", type=int, default=8050)
    a = p.parse_args(argv)
    unknown = set(a.symbols) - set(cfg.SPEC_BY_SYMBOL)
    if unknown:
        p.error(f"unknown symbols: {sorted(unknown)}")
    return cfg.RuntimeOptions(mock_model=a.mock_model, poll_only=a.poll_only,
                              symbols=tuple(a.symbols), log_level=a.log_level.upper(),
                              dashboard=not a.no_dashboard, dashboard_host=a.dashboard_host,
                              dashboard_port=a.dashboard_port)


def main(argv: Optional[list[str]] = None) -> int:
    opts = parse_args(argv)
    logging.basicConfig(level=opts.log_level, format="%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s")
    try:
        bot = Bot(opts)
    except cfg.ConfigError as exc:
        log.error("%s", exc)
        return 2
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: bot.stop_evt.set())
    bot.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
