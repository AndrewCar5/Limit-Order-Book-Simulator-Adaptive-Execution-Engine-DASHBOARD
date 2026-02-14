#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j

(
  cd "$ROOT_DIR"
  mkdir -p "$ROOT_DIR/outputs_exec_twap_full"
  "$BUILD_DIR/lob_sim" --config "$ROOT_DIR/configs/exec_bybit_twap_full.cfg" | tee "$ROOT_DIR/outputs_exec_twap_full/run_breakdown.txt"
)

if command -v python3 >/dev/null 2>&1; then
  python3 "$ROOT_DIR/analysis/metrics.py" \
    --trades "$ROOT_DIR/outputs_exec_twap_full/execution_trades.csv" \
    --book "$ROOT_DIR/outputs_exec_twap_full/execution_book.csv" \
    --orders "$ROOT_DIR/outputs_exec_twap_full/execution_orders.csv" || true
fi
