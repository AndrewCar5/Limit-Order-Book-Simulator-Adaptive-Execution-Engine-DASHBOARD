# Limit Order Book Execution Engine

Event-driven limit order book engine with price-time priority, partial fills, cancels, and standard order types (GTC/IOC/FOK/DAY/GTD/Market/Post-only). Includes L2 replay (Bybit OB200 snapshots + trade feed) and a Streamlit dashboard for execution quality.

## One-command run

```bash
./scripts/run_replay_bybit.sh
```

Generates CSV logs and a basic metrics report in `outputs_exec_single_2026_01_01/`.

## Execution demos

- **Single order demo (1-day):**
  ```bash
  ./scripts/run_replay_bybit.sh
  ```

- **TWAP demo (full dataset):**
  ```bash
  ./scripts/run_replay_bybit_full.sh
  ```

## Key capabilities

- Matching engine + order book with FIFO queues and cancel/partial-fill logic
- Order types: **GTC**, **IOC**, **FOK**, **DAY**, **GTD**, **Market**, **Post-only**
- L2 replay (Bybit OB200) + external trade feed
- Execution metrics: fill rate, time-to-fill, slippage, fee impact
- Streamlit dashboard for execution diagnostics
- Execution policies: fixed and adaptive TWAP (schedule-based aggression control)
- Online markout model (lightweight ML) for aggression bias in adaptive mode

## Repo layout

```
lob-sim/
  src/engine/        # C++ matching engine + simulator
  configs/           # run configs (key=value)
  analysis/          # python metrics + plots
  outputs/           # logs
  docs/              # write-up notes
```

## Order type semantics

- **GTC:** rests on book until filled or canceled.
- **IOC:** fills what is immediately available, cancels remainder.
- **FOK:** fills entirely immediately or cancels (no partials).
- **DAY:** rests until end-of-day boundary in simulator event time.
- **GTD:** rests until explicit expiry timestamp in simulator event time.
- **Market:** crosses the spread immediately, best available price.
- **Post-only:** cancels if it would cross at submission time.

## Execution metric semantics

- **MTM (cash+inventory@mid):** mark-to-market value, not realized cash PnL.
- **Spread capture / per-maker edge:** meaningful only when maker fills exist.
- **Markout:** use horizon markout for taker execution quality (50/250/1000/5000 ms).
- **Cost decomposition:** evaluate slippage + fees in bps for execution runs.

## Execution time window config

- `exec_start_ms` / `exec_end_ms` support two modes:
  - **Absolute epoch ms** (e.g. `1767225600000`).
  - **Replay-relative offsets** from first replay timestamp (e.g. `0` to `604800000`).
- Replay-relative mode is auto-selected for values before year-2000 epoch cutoff.

## Adaptive execution policy

- `exec_policy=fixed`: legacy behavior (single style from `exec_tif`/flags).
- `exec_policy=adaptive`: slice-by-slice decision among IOC / passive / post-only / skip.
- Adaptive decision inputs:
  - schedule error (`time_progress - fill_progress`)
  - spread, imbalance, volatility guards
  - online ML predicted short-horizon mid move (markout proxy)
- Main knobs:
  - `exec_behind_threshold`, `exec_ahead_threshold`, `exec_panic_threshold`
  - `exec_spread_max_ticks`, `exec_imbalance_max`, `exec_vol_max`, `exec_max_passive_slices`
  - `exec_min_slice_qty`, `exec_max_slice_qty`, `exec_passive_offset_ticks`
  - `exec_final_sweep_slices` / `exec_final_sweep_ms` (force IOC in final window)
  - `exec_ml_*` for horizon, learning rate, regularization, warmup, aggression threshold

## Notes

- Single instrument, single venue, fixed tick size.
- Notional is reported in scaled integer units (`price_ticks × qty_units`). We report trader-only fill notional separately
  from all trade-feed prints to avoid mixing distributions.
