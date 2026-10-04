# TimesFM-3 intraday paper-trading bot (Upstox · NSE/MCX)

Paper-trades NIFTY / liquid stock options / MCX CRUDEOIL options intraday with quantile forecasts from
`google/timesfm-3.0-pytorch`. **No real orders are ever placed.**

```
upstox_feed.py     -> ticks -> 5m candles ──┐ (WebSocket thread, never blocked)
features.py        <- candles: z=ln(P/P_anchor), gap-shielded, covariates
model_worker.py    -> dedicated GPU thread, fp16, batched TimesFM-3 inference -> Prediction(q10,q50,q90,...)
strategy_engine.py -> signals, sizing, paper fills, trailing, square-off, SQLite (paper_trades.db)
dashboard.py       -> live monitoring web UI (stdlib http server thread)
main.py            -> wires it all together
```

## Setup
```bash
pip install -r requirements.txt          # install a CUDA build of torch first for the GTX 1660 Ti
echo "<your Upstox access token>" > upstox.txt     # git-ignored; stripped of whitespace on load
python main.py                           # live data + TimesFM-3 (needs GPU + HF access to the weights)
python main.py --mock-model              # wiring test without GPU/weights (signals are NOT TimesFM)
python main.py --symbols NIFTY CRUDEOIL --poll-only
python main.py --dashboard-port 8050     # UI at http://127.0.0.1:8050 (default; --no-dashboard to disable)
pytest tests                             # offline unit/integration tests
```
Upstox tokens expire daily — refresh `upstox.txt` each morning. TimesFM-3 weights are licensed
non-commercial/non-production; paper trading only.

## Rules implemented
| Item | Behaviour |
|---|---|
| Capital / risk | ₹2,00,000; ≤1.5 % of *current* equity risked per trade; ≤30 % of equity margin per category (NIFTY / STOCK / MCX) |
| Entry window | no entries before 09:30 IST; last entry 14:45 (NSE) / 22:45 (MCX) |
| Square-off | everything flat at 15:15 (NSE) / 23:15 (MCX); also flattened on shutdown |
| Setup 1 | \|q50−px\| ≥ 1.5·ATR20 and ≥ 15 % of envelope, envelope ≥ 5 % wider than recent forecasts → buy ~0.45-Δ CE/PE; stop on underlying at q10 (long) / q90 (short) |
| Setup 2 | q90−q10 < 1.6·ATR20·√H and small drift → iron condor with short strikes outside q10 / q90; stop if underlying crosses q10 / q90 |
| Sizing | long: loss to stop (delta-estimated, ≤ premium) ≤ risk budget. Condor: *max loss* (width − credit) ≤ risk budget |
| Fills | LTP ± 0.1 % slippage, ₹20 per leg-order brokerage (`config.BROKERAGE_PER_ORDER`) |
| Trailing | underlying stop ratchets after 1× stop-distance of progress; premium floor: break-even at +1R, lock 50 % of peak at +2R; time stop after one horizon |
| Hard cap | mark-to-market loss ≥ per-trade risk limit → immediate exit (`RISK_CAP`) |

Design notes: the target series is the gap-shielded `ln(P_t/P_anchor)` (overnight jump removed);
`intraday_progress`, `is_us_overlap`, `is_expiry_day` are passed as past-and-future covariates and
Parkinson volatility as a past-only covariate (native TimesFM-3 covariate support). `make_positive`
is disabled (log-returns are signed). The q90−q10 envelope is a horizon-wide quantity, so it is compared with
the per-bar ATR20 scaled by √H.

## Tables in `paper_trades.db`
`predictions`, `trades` (entry/exit, risk, P&L, reason), `fills` (every leg fill with slippage & charges), `equity`.

## Things to check before trusting it
* Lot sizes / expiry weekdays change; lot sizes come from the live instrument master (`config.LOT_SIZES` is only a fallback).
* Exchange holidays and special sessions (Muhurat) are not modelled.
* Option-chain endpoint support for MCX underlyings is assumed; a master+quote fallback is built in.
* Paper fills at LTP±0.1 % understate real-world spread/impact in illiquid strikes.

## Dashboard
Starts with the bot at `http://127.0.0.1:8050` (auto-refresh every 3 s; light/dark follows the OS, with a toggle).
Shows equity / P&L / closed-trade KPIs, today's equity curve, margin used vs. the 30 % cap per category,
open positions (legs, stop, target, trail floor), the latest q10/q50/q90 forecasts, recent trades,
feed health (stale-tick warnings), recent skipped signals and session state (cold start / trading / squared-off).
JSON is available at `/api/state`. It has **no authentication**, so it binds to localhost; only change
`--dashboard-host` on a network you trust.
