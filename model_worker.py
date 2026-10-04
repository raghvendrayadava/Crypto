"""TimesFM-3 inference, run in a dedicated worker thread.

``InferenceWorker.submit()`` is non-blocking (called from the WebSocket candle callback); the
worker batches whatever has queued up for the same bar boundary, runs one fp16 forward pass on the
GPU and hands structured :class:`Prediction` objects to a callback.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

import config as cfg
from features import FeatureBundle

log = logging.getLogger(__name__)

Q10, Q50, Q90 = 0, 4, 8                      # indices into TimesFM-3's 9 deciles (0.1 .. 0.9)


@dataclass(frozen=True)
class Prediction:
    symbol: str
    exchange: str
    bar_start: dt.datetime                   # the closed bar the forecast is anchored on
    anchor_price: float
    horizon: int
    q10: float                               # horizon-end quantile prices
    q50: float
    q90: float
    expected_drift: float                    # q50 - anchor (rupees)
    expected_drift_pct: float
    vol_spread: float                        # q90 - q10 (rupees)
    spread_pct: float
    atr20: float
    parkinson_now: float
    is_expiry_day: bool
    price_quantiles: np.ndarray = field(repr=False, compare=False)   # (H, 9) rupee quantiles
    latency_s: float = 0.0
    backend: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "symbol": self.symbol, "bar_start": self.bar_start.isoformat(), "anchor": self.anchor_price,
            "q10": self.q10, "q50": self.q50, "q90": self.q90, "drift": self.expected_drift,
            "spread": self.vol_spread, "atr20": self.atr20, "latency_s": self.latency_s,
            "backend": self.backend,
        }


def make_prediction(bundle: FeatureBundle, z_quantiles: np.ndarray, latency: float, backend: str) -> Prediction:
    """z_quantiles: (H, 9) log-return deciles relative to the anchor."""
    zq = np.sort(np.asarray(z_quantiles, dtype=np.float64), axis=-1)
    pq = bundle.prices(zq)
    anchor = bundle.anchor_price
    q10, q50, q90 = (float(pq[-1, i]) for i in (Q10, Q50, Q90))
    return Prediction(
        symbol=bundle.symbol, exchange=bundle.exchange, bar_start=bundle.bar_start, anchor_price=anchor,
        horizon=bundle.horizon, q10=q10, q50=q50, q90=q90,
        expected_drift=q50 - anchor, expected_drift_pct=(q50 - anchor) / anchor,
        vol_spread=q90 - q10, spread_pct=(q90 - q10) / anchor, atr20=bundle.atr20,
        parkinson_now=bundle.parkinson_now, is_expiry_day=bundle.spot_is_expiry,
        price_quantiles=pq, latency_s=latency, backend=backend,
    )


# =========================================================================== #
# Backends
# =========================================================================== #
class Backend(Protocol):
    name: str

    def predict(self, bundles: Sequence[FeatureBundle]) -> list[np.ndarray]:
        """Return one (H, 9) array of z-quantiles per bundle."""


class TimesFMBackend:
    """google/timesfm-3.0-pytorch on CUDA in float16 (weights + autocast).

    NOTE: the TimesFM-3 weights are released under a non-commercial licence; this bot is
    paper-trading only.
    """

    def __init__(self, model_id: str = cfg.MODEL_ID, batch_size: int = cfg.MODEL_BATCH_SIZE) -> None:
        import torch
        from timesfm3 import TimesFM3Forecaster

        self._torch = torch
        if not torch.cuda.is_available():
            if not cfg.ALLOW_CPU_FALLBACK:
                raise RuntimeError("CUDA GPU not available; fp16 inference requires the GTX 1660 Ti "
                                   "(set ALLOW_CPU_FALLBACK=1 to run on CPU in fp32 for debugging).")
            log.warning("CUDA unavailable: running TimesFM-3 on CPU in fp32 (debug only)")
            self.device, self._fp16 = "cpu", False
        else:
            self.device, self._fp16 = "cuda", True
        self.name = f"timesfm3-{'fp16' if self._fp16 else 'fp32-cpu'}"
        log.info("Loading %s on %s ...", model_id, self.device)
        self.forecaster = TimesFM3Forecaster.from_pretrained(
            model_id, device=self.device, per_core_batch_size=batch_size)
        if self._fp16:
            self._enable_fp16()
        self._warm_up()

    # -- precision -------------------------------------------------------- #
    def _enable_fp16(self) -> None:
        """Half-precision weights; decode() is wrapped so inputs are fp16 and matmuls run under autocast."""
        torch = self._torch
        model = self.forecaster.model
        model.half()
        original = getattr(model, "_decode_original", None) or model.decode
        model._decode_original = original

        def decode_fp16(*args, **kwargs):
            def cast(x):
                return x.to(torch.float16) if torch.is_tensor(x) and x.is_floating_point() else x
            args = tuple(cast(a) for a in args)
            kwargs = {k: cast(v) for k, v in kwargs.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = original(*args, **kwargs)
            return out.float() if torch.is_tensor(out) else out

        model.decode = decode_fp16                       # instance attr shadows the method
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    def _warm_up(self) -> None:
        """Run a dummy forecast; if fp16 weights yield NaN/inf, keep fp32 weights + fp16 autocast."""
        from features import FeatureBundle
        n, h = 256, cfg.HORIZON
        rng = np.random.default_rng(0)
        z = np.cumsum(rng.normal(0, 1e-3, n)).astype(np.float32)
        bundle = FeatureBundle("WARM", "NSE", cfg.now_ist(), 100.0, z - z[-1],
                               np.abs(rng.normal(0.1, 0.02, (1, n))).astype(np.float32),
                               rng.random((3, n + h)).astype(np.float32), h, 0.1, 0.001, False)
        try:
            out = self.predict([bundle])[0]
            ok = bool(np.all(np.isfinite(out)))
        except Exception as exc:                                          # noqa: BLE001
            log.warning("fp16-weight warm-up failed (%s)", exc)
            ok = False
        if not ok and self._fp16:
            log.warning("fp16 weights unstable; reloading fp32 weights with fp16 autocast compute")
            model = self.forecaster.model
            model.float()
            self.name = "timesfm3-fp16-autocast"
            out = self.predict([bundle])[0]
            if not np.all(np.isfinite(out)):
                raise RuntimeError("TimesFM-3 produced non-finite output during warm-up")
        log.info("TimesFM-3 ready (%s); VRAM used %.2f GB", self.name, self._vram_gb())

    def _vram_gb(self) -> float:
        return self._torch.cuda.memory_allocated() / 1e9 if self.device == "cuda" else 0.0

    # -- inference -------------------------------------------------------- #
    def predict(self, bundles: Sequence[FeatureBundle]) -> list[np.ndarray]:
        horizon = bundles[0].horizon
        try:
            outs = list(self.forecaster.predict_batch(
                contexts=[b.target for b in bundles],
                horizon=horizon,
                past_only_covariates=[b.past_only for b in bundles],
                past_future_covariates=[b.past_future for b in bundles],
                return_quantiles=True,
                use_symmetric_averaging=False,
                make_positive=False,      # z can be negative; default clamping would corrupt it
                sort_quantiles=True,
                use_znorm=False,
                padding_mode="none",
            ))
            res = [np.asarray(o.quantiles, dtype=np.float32) for o in outs]
            if all(np.all(np.isfinite(r)) for r in res):
                return res
            log.warning("non-finite covariate forecast; retrying univariate")
        except Exception as exc:                                          # noqa: BLE001
            if "out of memory" in str(exc).lower():
                self._torch.cuda.empty_cache()
            log.warning("covariate forecast failed (%s); retrying univariate", exc)
        outs = list(self.forecaster.predict_batch(
            contexts=[b.target for b in bundles], horizon=horizon, return_quantiles=True,
            use_symmetric_averaging=False, make_positive=False, sort_quantiles=True))
        return [np.asarray(o.quantiles, dtype=np.float32) for o in outs]


class MockBackend:
    """Offline stand-in (no GPU / weights): momentum + vol-scaled quantiles. For wiring tests only."""
    name = "mock"

    def predict(self, bundles: Sequence[FeatureBundle]) -> list[np.ndarray]:
        out = []
        norm = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
        for b in bundles:
            sigma = max(b.parkinson_now, 1e-5)
            steps = np.arange(1, b.horizon + 1)[:, None]
            drift_per_bar = float(b.target[-1] - b.target[-7]) / 6.0 * 0.5   # damped trailing momentum
            out.append((drift_per_bar * steps + sigma * np.sqrt(steps) * norm[None, :]).astype(np.float32))
        return out


# =========================================================================== #
# Worker thread
# =========================================================================== #
PredictionCallback = Callable[[Prediction], None]


class InferenceWorker(threading.Thread):
    """Coalescing, batching inference thread.

    ``submit`` keeps only the newest bundle per symbol, so a slow GPU never builds a backlog of
    stale forecasts.
    """

    def __init__(self, backend: Backend, on_prediction: PredictionCallback,
                 batch_size: int = cfg.MODEL_BATCH_SIZE, collect_s: float = cfg.BATCH_COLLECT_SECONDS) -> None:
        super().__init__(name="timesfm-worker", daemon=True)
        self.backend = backend
        self.on_prediction = on_prediction
        self.batch_size = batch_size
        self.collect_s = collect_s
        self._pending: "OrderedDict[str, FeatureBundle]" = OrderedDict()
        self._cv = threading.Condition()
        self._stop_evt = threading.Event()

    def submit(self, bundle: FeatureBundle) -> None:
        with self._cv:
            self._pending[bundle.symbol] = bundle
            self._pending.move_to_end(bundle.symbol)
            self._cv.notify()

    def stop(self) -> None:
        self._stop_evt.set()
        with self._cv:
            self._cv.notify_all()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            with self._cv:
                while not self._pending and not self._stop_evt.is_set():
                    self._cv.wait()
                if self._stop_evt.is_set():
                    return
            time.sleep(self.collect_s)                    # let sibling symbols' bars arrive
            with self._cv:
                jobs = list(self._pending.values())
                self._pending.clear()
            for i in range(0, len(jobs), self.batch_size):
                self._run_chunk(jobs[i:i + self.batch_size])

    def _run_chunk(self, chunk: list[FeatureBundle]) -> None:
        t0 = time.perf_counter()
        try:
            results = self.backend.predict(chunk)
        except Exception:                                                  # noqa: BLE001
            log.exception("batch inference failed; retrying symbols one by one")
            results = []
            for b in chunk:
                try:
                    results.append(self.backend.predict([b])[0])
                except Exception:                                          # noqa: BLE001
                    log.exception("inference failed for %s", b.symbol)
                    results.append(None)
        dt_s = time.perf_counter() - t0
        for b, zq in zip(chunk, results):
            if zq is None or zq.shape[0] < b.horizon or not np.all(np.isfinite(zq)):
                continue
            try:
                self.on_prediction(make_prediction(b, zq[: b.horizon], dt_s, self.backend.name))
            except Exception:                                              # noqa: BLE001
                log.exception("prediction callback failed for %s", b.symbol)
        log.debug("inference of %d series took %.2fs", len(chunk), dt_s)
