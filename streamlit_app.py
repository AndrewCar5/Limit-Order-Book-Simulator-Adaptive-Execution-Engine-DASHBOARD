from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "TWAP (Full)": ROOT_DIR / "outputs_exec_twap_full",
}

DEFAULT_HORIZONS_MS = (50, 250, 1000, 5000)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, usecols=None) -> pd.DataFrame:
    if usecols is None:
        return pd.read_csv(path)
    try:
        return pd.read_csv(path, usecols=usecols)
    except ValueError:
        return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_run_breakdown(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    adverse = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Adverse selection (by horizon ms):") or line.startswith(
            "Markout (by horizon ms):"
        ):
            payload = line.split(":", 1)[1].strip()
            for item in payload.split(","):
                token = item.strip()
                if "=" not in token:
                    continue
                h, v = token.split("=", 1)
                try:
                    adverse[int(h.strip())] = float(v.strip())
                except ValueError:
                    continue
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    if adverse:
        out["adverse_by_horizon"] = adverse
    return out


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def load_dataset(out_dir: Path):
    trades_path = out_dir / "execution_trades.csv"
    book_path = out_dir / "execution_book.csv"
    orders_path = out_dir / "execution_orders.csv"
    decisions_path = out_dir / "execution_decisions.csv"
    run_breakdown_path = out_dir / "run_breakdown.txt"

    if not trades_path.exists() or not book_path.exists() or not orders_path.exists():
        return None, None, None, None, None

    trades = load_csv(trades_path)
    book = load_csv(book_path)
    orders = load_csv(orders_path)
    decisions = load_csv(decisions_path) if decisions_path.exists() else pd.DataFrame()
    run_breakdown = load_run_breakdown(run_breakdown_path)
    return trades, book, orders, decisions, run_breakdown


def extract_exec_orders(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders
    orders = orders.copy()
    if "action" in orders.columns:
        orders["action"] = orders["action"].astype(str)
        exec_mask = orders["action"].str.startswith("exec_")
        exec_orders = orders[exec_mask].copy()
    else:
        exec_orders = orders[orders.get("owner") == "MM"].copy()
    exec_orders["time"] = pd.to_numeric(exec_orders["time"], errors="coerce")
    exec_orders["qty"] = pd.to_numeric(exec_orders["qty"], errors="coerce").fillna(0)
    exec_orders["order_id"] = pd.to_numeric(exec_orders["order_id"], errors="coerce")
    exec_orders = exec_orders.dropna(subset=["time", "order_id"])
    return exec_orders


def infer_trader_side(trades: pd.DataFrame) -> pd.Series:
    maker_side = trades["maker_side"].astype(str)
    is_maker = trades["maker_is_mm"] == 1
    is_taker = trades["taker_is_mm"] == 1

    side = pd.Series("", index=trades.index)
    side[is_maker] = maker_side[is_maker]
    opposite = maker_side.map({"BUY": "SELL", "SELL": "BUY"})
    side[is_taker] = opposite[is_taker]
    return side


def build_trader_fills(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["time"] = pd.to_numeric(trades["time"], errors="coerce")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce")
    trades["mid"] = pd.to_numeric(trades["mid"], errors="coerce")
    trades["maker_id"] = pd.to_numeric(trades["maker_id"], errors="coerce")
    trades["taker_id"] = pd.to_numeric(trades["taker_id"], errors="coerce")
    trades = trades.dropna(subset=["time", "price", "qty", "mid"])

    trader_mask = (trades["maker_is_mm"] == 1) | (trades["taker_is_mm"] == 1)
    trades = trades[trader_mask].copy()
    if trades.empty:
        return trades

    trades["trader_side"] = infer_trader_side(trades)
    trades["trader_order_id"] = np.where(
        trades["maker_is_mm"] == 1, trades["maker_id"], trades["taker_id"]
    )
    trades["slippage_ticks"] = np.where(
        trades["trader_side"] == "BUY",
        trades["price"] - trades["mid"],
        trades["mid"] - trades["price"],
    )
    trades["slippage_bps"] = np.where(
        trades["mid"] > 0, 10000.0 * trades["slippage_ticks"] / trades["mid"], np.nan
    )
    return trades


def compute_execution_metrics(
    trades: pd.DataFrame, orders: pd.DataFrame, run_breakdown: dict
) -> dict:
    metrics = {
        "total_trades": len(trades),
        "trader_fills": 0,
        "maker_fills": 0,
        "taker_fills": 0,
        "fill_qty": 0.0,
        "submitted_qty": 0.0,
        "fill_rate": 0.0,
        "avg_time_to_fill_ms": None,
        "avg_slippage_ticks": None,
        "avg_slippage_bps": None,
        "fill_notional": 0.0,
        "mid_notional": 0.0,
        "slippage_value": 0.0,
        "slippage_bps_total": None,
        "fees_total": None,
        "fees_bps": None,
        "total_cost_bps": None,
        "avg_fee_per_fill": None,
        "action_counts": {},
        "inventory": parse_number(run_breakdown.get("Inventory")) if run_breakdown else None,
        "max_abs_inventory": (
            parse_number(run_breakdown.get("Max abs inventory (observed)"))
            if run_breakdown
            else None
        ),
        "mtm_value": (
            parse_number(run_breakdown.get("MTM (cash+inv@mid)"))
            if run_breakdown
            else None
        ),
        "exec_target_qty": (
            parse_number(run_breakdown.get("Exec qty (units)")) if run_breakdown else None
        ),
        "exec_filled_qty": (
            parse_number(run_breakdown.get("Exec filled (units)")) if run_breakdown else None
        ),
        "exec_completion_pct": (
            parse_number(run_breakdown.get("Completion (%)")) if run_breakdown else None
        ),
        "exec_shortfall": (
            parse_number(run_breakdown.get("End shortfall (units)")) if run_breakdown else None
        ),
        "exec_forced_ioc": (
            parse_number(run_breakdown.get("Exec forced IOC")) if run_breakdown else None
        ),
        "exec_ml_ioc": (
            parse_number(run_breakdown.get("Exec ML IOC")) if run_breakdown else None
        ),
        "exec_ml_ioc_pct": None,
        "exec_skips": (
            parse_number(run_breakdown.get("Exec skips")) if run_breakdown else None
        ),
        "exec_max_no_fill": (
            parse_number(run_breakdown.get("Exec max no-fill streak"))
            if run_breakdown
            else None
        ),
        "exec_start_ts": (
            parse_number(run_breakdown.get("Exec start (ms)")) if run_breakdown else None
        ),
        "exec_end_ts": (
            parse_number(run_breakdown.get("Exec end (ms)")) if run_breakdown else None
        ),
        "exec_slice_ms": (
            parse_number(run_breakdown.get("Exec slice (ms)")) if run_breakdown else None
        ),
    }
    if metrics["mtm_value"] is None and run_breakdown:
        metrics["mtm_value"] = parse_number(run_breakdown.get("PnL"))

    if trades.empty:
        return metrics

    trades = trades.copy()
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce").fillna(0)
    maker_mask = trades["maker_is_mm"] == 1
    taker_mask = trades["taker_is_mm"] == 1
    metrics["maker_fills"] = int(maker_mask.sum())
    metrics["taker_fills"] = int(taker_mask.sum())

    trader_trades = build_trader_fills(trades)
    metrics["trader_fills"] = len(trader_trades)
    if trader_trades.empty:
        return metrics

    metrics["fill_qty"] = float(trader_trades["qty"].sum())
    metrics["fill_notional"] = float((trader_trades["price"] * trader_trades["qty"]).sum())
    metrics["mid_notional"] = float((trader_trades["mid"] * trader_trades["qty"]).sum())
    metrics["slippage_value"] = float(
        (trader_trades["slippage_ticks"] * trader_trades["qty"]).sum()
    )
    metrics["avg_slippage_ticks"] = float(trader_trades["slippage_ticks"].mean())
    metrics["avg_slippage_bps"] = float(trader_trades["slippage_bps"].mean())
    if metrics["mid_notional"] > 0:
        metrics["slippage_bps_total"] = (
            10000.0 * metrics["slippage_value"] / metrics["mid_notional"]
        )

    if not orders.empty:
        exec_orders = extract_exec_orders(orders)
        submitted_qty = float(exec_orders["qty"].sum()) if not exec_orders.empty else 0.0
        metrics["submitted_qty"] = submitted_qty
        if submitted_qty > 0:
            metrics["fill_rate"] = metrics["fill_qty"] / submitted_qty

        if not exec_orders.empty:
            if "action" in exec_orders.columns:
                metrics["action_counts"] = exec_orders["action"].value_counts().to_dict()
                ioc_total = (
                    metrics["action_counts"].get("exec_ioc_touch", 0)
                    + metrics["action_counts"].get("exec_forced_ioc", 0)
                )
                if metrics.get("exec_ml_ioc") is not None and ioc_total > 0:
                    metrics["exec_ml_ioc_pct"] = (
                        100.0 * float(metrics["exec_ml_ioc"]) / float(ioc_total)
                    )
            first_fill = (
                trader_trades.groupby("trader_order_id", as_index=True)["time"].min().to_dict()
            )
            exec_orders["first_fill_time"] = exec_orders["order_id"].map(first_fill)
            filled = exec_orders.dropna(subset=["time", "first_fill_time"]).copy()
            if not filled.empty:
                filled["time_to_fill"] = filled["first_fill_time"] - filled["time"]
                metrics["avg_time_to_fill_ms"] = float(filled["time_to_fill"].mean())

    fees_total = parse_number(run_breakdown.get("Fees")) if run_breakdown else None
    metrics["fees_total"] = fees_total
    if fees_total is not None and metrics["fill_notional"] > 0:
        metrics["fees_bps"] = 10000.0 * fees_total / metrics["fill_notional"]
    if fees_total is not None and metrics["trader_fills"] > 0:
        metrics["avg_fee_per_fill"] = fees_total / float(metrics["trader_fills"])
    if metrics["fees_bps"] is not None and metrics["slippage_bps_total"] is not None:
        metrics["total_cost_bps"] = metrics["fees_bps"] + metrics["slippage_bps_total"]

    return metrics


def build_cumulative_execution(trader_trades: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    parts = []

    if not trader_trades.empty:
        fills = (
            trader_trades.groupby("time", as_index=False)["qty"]
            .sum()
            .sort_values("time")
            .reset_index(drop=True)
        )
        fills["cum_qty"] = fills["qty"].cumsum()
        fills["series"] = "Filled Qty"
        parts.append(fills[["time", "cum_qty", "series"]])

    if not orders.empty:
        ords = orders.copy()
        ords["time"] = pd.to_numeric(ords["time"], errors="coerce")
        ords["qty"] = pd.to_numeric(ords["qty"], errors="coerce").fillna(0)
        mm_orders = ords[ords["owner"] == "MM"].dropna(subset=["time"])
        if not mm_orders.empty:
            submitted = (
                mm_orders.groupby("time", as_index=False)["qty"]
                .sum()
                .sort_values("time")
                .reset_index(drop=True)
            )
            submitted["cum_qty"] = submitted["qty"].cumsum()
            submitted["series"] = "Submitted Qty"
            parts.append(submitted[["time", "cum_qty", "series"]])

    if not parts:
        return pd.DataFrame(columns=["time", "cum_qty", "series"])
    return pd.concat(parts, ignore_index=True)


def build_schedule_line(
    exec_orders: pd.DataFrame,
    exec_target_qty: float,
    exec_start_ts: float | None,
    exec_end_ts: float | None,
    exec_slice_ms: float | None,
) -> pd.DataFrame:
    if exec_target_qty is None or exec_target_qty <= 0:
        return pd.DataFrame(columns=["time", "cum_qty", "series"])
    if exec_orders.empty:
        return pd.DataFrame(columns=["time", "cum_qty", "series"])
    if exec_start_ts and exec_end_ts and exec_slice_ms:
        if exec_end_ts <= exec_start_ts or exec_slice_ms <= 0:
            return pd.DataFrame(columns=["time", "cum_qty", "series"])
        times = np.arange(exec_start_ts, exec_end_ts + 1, exec_slice_ms, dtype=np.int64)
    else:
        times = np.sort(exec_orders["time"].unique())
    if len(times) < 2:
        return pd.DataFrame(columns=["time", "cum_qty", "series"])
    start = float(times[0])
    end = float(times[-1])
    if end <= start:
        return pd.DataFrame(columns=["time", "cum_qty", "series"])
    schedule_qty = exec_target_qty * (times - start) / (end - start)
    df = pd.DataFrame({"time": times, "cum_qty": schedule_qty})
    df["series"] = "Target Schedule"
    return df


def build_action_timeline(
    decisions: pd.DataFrame,
    exec_orders: pd.DataFrame,
    exec_start_ts: float | None,
    exec_end_ts: float | None,
    exec_slice_ms: float | None,
) -> pd.DataFrame:
    if not decisions.empty and "final_action" in decisions.columns:
        out = decisions.copy()
        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["final_action"] = out["final_action"].astype(str)
        out["ml_triggered"] = pd.to_numeric(out.get("ml_triggered"), errors="coerce").fillna(0)
        out = out.dropna(subset=["time"])
        return out.rename(columns={"final_action": "action"})[["time", "action", "ml_triggered"]]
    if exec_orders.empty:
        return pd.DataFrame(columns=["time", "action", "ml_triggered"])
    exec_orders = exec_orders.copy()
    exec_orders["action"] = exec_orders["action"].astype(str)
    exec_orders["time"] = pd.to_numeric(exec_orders["time"], errors="coerce")
    action_by_time = exec_orders.groupby("time")["action"].agg(lambda s: s.iloc[0])
    if exec_start_ts and exec_end_ts and exec_slice_ms and exec_slice_ms > 0:
        times = np.arange(exec_start_ts, exec_end_ts + 1, exec_slice_ms, dtype=np.int64)
        actions = []
        for t in times:
            actions.append(action_by_time.get(t, "exec_skip"))
        return pd.DataFrame({"time": times, "action": actions, "ml_triggered": 0})
    return pd.DataFrame(
        {"time": action_by_time.index.to_numpy(), "action": action_by_time.values, "ml_triggered": 0}
    )


def compute_ml_scatter(decisions: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or book.empty:
        return pd.DataFrame(columns=["predicted_move", "actual_move", "ml_triggered"])
    decisions = decisions.copy()
    decisions["time"] = pd.to_numeric(decisions["time"], errors="coerce")
    decisions["mid_now"] = pd.to_numeric(decisions.get("mid_now"), errors="coerce")
    decisions["ml_ready"] = pd.to_numeric(decisions.get("ml_ready"), errors="coerce").fillna(0)
    decisions["ml_predicted_move"] = pd.to_numeric(
        decisions.get("ml_predicted_move"), errors="coerce"
    )
    decisions["ml_triggered"] = pd.to_numeric(decisions.get("ml_triggered"), errors="coerce").fillna(0)
    decisions["ml_horizon_ms"] = pd.to_numeric(
        decisions.get("ml_horizon_ms"), errors="coerce"
    )
    decisions = decisions.dropna(subset=["time", "mid_now", "ml_predicted_move", "ml_horizon_ms"])
    decisions = decisions[decisions["ml_ready"] == 1]
    if decisions.empty:
        return pd.DataFrame(columns=["predicted_move", "actual_move", "ml_triggered"])

    horizon = int(decisions["ml_horizon_ms"].iloc[0])

    mids = book.copy()
    mids["time"] = pd.to_numeric(mids["time"], errors="coerce")
    mids["mid"] = pd.to_numeric(mids["mid"], errors="coerce")
    mids = mids.dropna(subset=["time", "mid"]).sort_values("time")
    if mids.empty:
        return pd.DataFrame(columns=["predicted_move", "actual_move", "ml_triggered"])

    book_times = mids["time"].to_numpy(dtype=np.int64)
    book_mids = mids["mid"].to_numpy(dtype=float)

    target_times = decisions["time"].to_numpy(dtype=np.int64) + horizon
    idx = np.searchsorted(book_times, target_times, side="left")
    valid = idx < len(book_times)
    if not np.any(valid):
        return pd.DataFrame(columns=["predicted_move", "actual_move", "ml_triggered"])

    future_mid = book_mids[idx[valid]]
    base_mid = decisions["mid_now"].to_numpy(dtype=float)[valid]
    actual_move = future_mid - base_mid
    predicted = decisions["ml_predicted_move"].to_numpy(dtype=float)[valid]
    triggered = decisions["ml_triggered"].to_numpy(dtype=int)[valid]

    return pd.DataFrame(
        {
            "predicted_move": predicted,
            "actual_move": actual_move,
            "ml_triggered": triggered,
        }
    )


def build_slice_fill_ratio(
    trader_trades: pd.DataFrame, exec_orders: pd.DataFrame
) -> pd.DataFrame:
    if trader_trades.empty or exec_orders.empty:
        return pd.DataFrame(columns=["time", "fill_ratio"])
    fills_by_order = (
        trader_trades.groupby("trader_order_id", as_index=True)["qty"].sum().to_dict()
    )
    exec_orders = exec_orders.copy()
    exec_orders["filled_qty"] = exec_orders["order_id"].map(fills_by_order).fillna(0)
    exec_orders["fill_ratio"] = np.where(
        exec_orders["qty"] > 0, exec_orders["filled_qty"] / exec_orders["qty"], np.nan
    )
    return exec_orders[["time", "fill_ratio"]].dropna(subset=["time"])


def build_cumulative_prices(trader_trades: pd.DataFrame) -> pd.DataFrame:
    if trader_trades.empty:
        return pd.DataFrame(columns=["time", "label", "price"])
    fills = trader_trades.sort_values("time").copy()
    fills["cum_qty"] = fills["qty"].cumsum()
    fills["cum_fill_notional"] = (fills["price"] * fills["qty"]).cumsum()
    fills["cum_mid_notional"] = (fills["mid"] * fills["qty"]).cumsum()
    fills["cum_vwap"] = fills["cum_fill_notional"] / fills["cum_qty"]
    fills["cum_mid"] = fills["cum_mid_notional"] / fills["cum_qty"]
    out = fills[["time", "cum_vwap", "cum_mid"]].copy()
    out = out.rename(columns={"cum_vwap": "Cumulative VWAP", "cum_mid": "Cumulative Mid"})
    return out.melt(id_vars=["time"], var_name="label", value_name="price")


def compute_markout_curve(
    trader_trades: pd.DataFrame, book: pd.DataFrame, horizons_ms=DEFAULT_HORIZONS_MS
) -> pd.DataFrame:
    if trader_trades.empty or book.empty:
        return pd.DataFrame(columns=["horizon_ms", "markout_value", "markout_bps"])

    mids = book.copy()
    mids["time"] = pd.to_numeric(mids["time"], errors="coerce")
    mids["mid"] = pd.to_numeric(mids["mid"], errors="coerce")
    mids = mids.dropna(subset=["time", "mid"])
    mids = mids.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    if mids.empty:
        return pd.DataFrame(columns=["horizon_ms", "markout_value", "markout_bps"])

    fills = trader_trades.copy()
    fills = fills.dropna(subset=["time", "mid", "qty"])
    if fills.empty:
        return pd.DataFrame(columns=["horizon_ms", "markout_value", "markout_bps"])

    fill_times = fills["time"].to_numpy(dtype=np.int64)
    fill_mids = fills["mid"].to_numpy(dtype=float)
    fill_qty = fills["qty"].to_numpy(dtype=float)
    fill_sign = np.where(fills["trader_side"] == "BUY", 1.0, -1.0)

    book_times = mids["time"].to_numpy(dtype=np.int64)
    book_mids = mids["mid"].to_numpy(dtype=float)

    rows = []
    for horizon in horizons_ms:
        target_times = fill_times + int(horizon)
        idx = np.searchsorted(book_times, target_times, side="left")
        valid = idx < len(book_times)
        if not np.any(valid):
            rows.append(
                {"horizon_ms": int(horizon), "markout_value": np.nan, "markout_bps": np.nan}
            )
            continue

        future_mid = book_mids[idx[valid]]
        base_mid = fill_mids[valid]
        qty = fill_qty[valid]
        sign = fill_sign[valid]

        markout_value = float(np.sum((future_mid - base_mid) * sign * qty))
        base_notional = float(np.sum(base_mid * qty))
        markout_bps = (10000.0 * markout_value / base_notional) if base_notional > 0 else np.nan

        rows.append(
            {
                "horizon_ms": int(horizon),
                "markout_value": markout_value,
                "markout_bps": markout_bps,
            }
        )

    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Execution Engine Dashboard", layout="wide")
    st.title("Execution Engine Dashboard")

    dataset_name = "TWAP (Full)"
    max_points = 50000

    out_dir = DATASETS[dataset_name]
    trades, book, orders, decisions, run_breakdown = load_dataset(out_dir)

    if trades is None:
        st.error("Missing output files. Run the execution scripts first.")
        st.stop()

    def fmt_int(value):
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{int(round(float(value))):,}"

    def fmt_float(value, digits=2):
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{float(value):,.{digits}f}"

    metrics = compute_execution_metrics(trades, orders, run_breakdown or {})
    trader_trades = build_trader_fills(trades)
    exec_orders = extract_exec_orders(orders)
    markout_df = compute_markout_curve(trader_trades, book, DEFAULT_HORIZONS_MS)
    ml_scatter = compute_ml_scatter(decisions, book)
    markout_1s = None
    markout_5s = None
    if not markout_df.empty:
        markout_1s = markout_df.loc[markout_df["horizon_ms"] == 1000, "markout_bps"]
        if not markout_1s.empty:
            markout_1s = float(markout_1s.iloc[0])
        else:
            markout_1s = None
        markout_5s = markout_df.loc[markout_df["horizon_ms"] == 5000, "markout_bps"]
        if not markout_5s.empty:
            markout_5s = float(markout_5s.iloc[0])
        else:
            markout_5s = None

    st.caption(
        "Limitation: snapshot-rebuild replay does not model persistent queue position, "
        "so passive-fill realism is limited."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Qty", fmt_int(metrics["exec_target_qty"]))
    c2.metric("Filled Qty", fmt_int(metrics["fill_qty"]))
    c3.metric("Completion (%)", fmt_float(metrics["exec_completion_pct"], 2))
    c4.metric("End Shortfall (units)", fmt_int(metrics["exec_shortfall"]))

    c5, c6, c7, c8 = st.columns(4)
    ml_ioc_pct = metrics.get("exec_ml_ioc_pct")
    ml_ioc_pct_text = "n/a" if ml_ioc_pct is None or not np.isfinite(ml_ioc_pct) else f"{ml_ioc_pct:.1f}%"
    c5.metric("Child Orders", fmt_int(len(exec_orders)))
    c6.metric("Submitted Qty", fmt_int(metrics["submitted_qty"]))
    c7.metric("Forced IOC", fmt_int(metrics["exec_forced_ioc"]))
    c8.metric("ML IOC (%)", ml_ioc_pct_text)

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Total Cost (bps)", fmt_float(metrics["total_cost_bps"], 2))
    c10.metric("Fees (bps)", fmt_float(metrics["fees_bps"], 2))
    c11.metric("Slippage (bps)", fmt_float(metrics["slippage_bps_total"], 3))
    c12.metric("Fill Rate (%)", fmt_float(100.0 * metrics.get("fill_rate", 0.0), 2))

    c13, c14, c15, c16 = st.columns(4)
    c13.metric("Markout 1s (bps)", fmt_float(markout_1s, 3))
    c14.metric("Markout 5s (bps)", fmt_float(markout_5s, 3))
    c15.metric("Total Fees", fmt_float(metrics["fees_total"], 0))
    c16.metric("Max No-Fill Streak", fmt_int(metrics["exec_max_no_fill"]))

    st.subheader("1) Completion vs Schedule")
    cumulative = build_cumulative_execution(trader_trades, orders)
    schedule = build_schedule_line(
        exec_orders,
        metrics.get("exec_target_qty"),
        metrics.get("exec_start_ts"),
        metrics.get("exec_end_ts"),
        metrics.get("exec_slice_ms"),
    )
    if not schedule.empty:
        cumulative = pd.concat([cumulative, schedule], ignore_index=True)
    if cumulative.empty:
        st.info("No trader orders/fills found for cumulative execution chart.")
    else:
        cumulative["ts"] = pd.to_datetime(cumulative["time"], unit="ms")
        cumulative = downsample(cumulative, max_points)
        qty_chart = (
            alt.Chart(cumulative)
            .mark_line()
            .encode(
                x="ts:T",
                y=alt.Y("cum_qty:Q", title="Quantity"),
                color=alt.Color("series:N", title="Series"),
            )
            .properties(height=280)
        )
        st.altair_chart(qty_chart, use_container_width=True)

    st.subheader("2) Action Timeline (IOC / Passive / Skip)")
    timeline = build_action_timeline(
        decisions,
        exec_orders,
        metrics.get("exec_start_ts"),
        metrics.get("exec_end_ts"),
        metrics.get("exec_slice_ms"),
    )
    if timeline.empty:
        st.info("No exec action timeline available.")
    else:
        timeline["ts"] = pd.to_datetime(timeline["time"], unit="ms")
        timeline = downsample(timeline, max_points)
        action_order = [
            "exec_ioc_touch",
            "exec_forced_ioc",
            "exec_passive",
            "exec_post_only",
            "exec_skip",
        ]
        base = alt.Chart(timeline)
        action_chart = (
            base.mark_tick(thickness=2, size=60)
            .encode(
                x="ts:T",
                y=alt.Y("action:N", sort=action_order, title="Action"),
                color=alt.Color("action:N", legend=None),
            )
            .properties(height=260)
        )
        ml_points = timeline[timeline["ml_triggered"] == 1]
        if not ml_points.empty:
            overlay = (
                alt.Chart(ml_points)
                .mark_circle(size=60, color="#d62728")
                .encode(x="ts:T", y=alt.Y("action:N", sort=action_order))
            )
            action_chart = action_chart + overlay
        st.altair_chart(action_chart, use_container_width=True)

        action_counts = metrics.get("action_counts", {})
        if metrics.get("exec_skips") is not None:
            action_counts = dict(action_counts)
            action_counts.setdefault("exec_skip", int(metrics["exec_skips"]))
        if action_counts:
            total_actions = float(sum(action_counts.values()))
            action_df = pd.DataFrame(
                [
                    {
                        "action": action,
                        "count": int(count),
                        "pct": (100.0 * count / total_actions) if total_actions > 0 else 0.0,
                    }
                    for action, count in sorted(
                        action_counts.items(), key=lambda kv: kv[1], reverse=True
                    )
                ]
            )
            action_df["pct"] = action_df["pct"].map(lambda x: round(float(x), 2))
            st.dataframe(action_df, use_container_width=True)

    st.subheader("3) Cost Decomposition (bps)")
    cost_rows = [
        {"component": "Fees", "bps": metrics["fees_bps"]},
        {"component": "Slippage", "bps": metrics["slippage_bps_total"]},
        {"component": "Total Cost", "bps": metrics["total_cost_bps"]},
    ]
    cost_df = pd.DataFrame(cost_rows).dropna(subset=["bps"])
    if cost_df.empty:
        st.info("Insufficient data for cost decomposition.")
    else:
        cost_chart = (
            alt.Chart(cost_df)
            .mark_bar()
            .encode(
                x=alt.X("component:N", title="Component"),
                y=alt.Y("bps:Q", title="Cost (bps)"),
                color=alt.Color("component:N", legend=None),
            )
            .properties(height=280)
        )
        st.altair_chart(cost_chart, use_container_width=True)

    st.subheader("4) Markout Curve by Horizon")
    markout_df = markout_df.dropna(subset=["markout_bps"]) if not markout_df.empty else markout_df
    if markout_df.empty:
        st.info("Insufficient data for markout curve.")
    else:
        markout_chart = (
            alt.Chart(markout_df)
            .mark_bar()
            .encode(
                x=alt.X("horizon_ms:O", title="Horizon (ms)"),
                y=alt.Y("markout_bps:Q", title="Markout (bps)"),
            )
            .properties(height=280)
        )
        st.altair_chart(markout_chart, use_container_width=True)

    with st.expander("Advanced Diagnostics", expanded=False):
        st.subheader("Mid + Spread (Market Context)")
        book_plot = book.copy()
        book_plot["time"] = pd.to_numeric(book_plot["time"], errors="coerce")
        book_plot["mid"] = pd.to_numeric(book_plot["mid"], errors="coerce")
        book_plot["spread"] = pd.to_numeric(book_plot["spread"], errors="coerce")
        book_plot = book_plot.dropna(subset=["time", "mid", "spread"])
        book_ds = downsample(book_plot, max_points)
        book_ds["ts"] = pd.to_datetime(book_ds["time"], unit="ms")
        base = alt.Chart(book_ds).encode(x="ts:T")
        mid_line = base.mark_line(color="#1f77b4").encode(
            y=alt.Y("mid:Q", title="Mid Price")
        )
        spread_line = base.mark_line(color="#ff7f0e").encode(
            y=alt.Y("spread:Q", title="Spread")
        )
        context_chart = alt.layer(mid_line, spread_line).resolve_scale(y="independent").properties(
            height=240
        )
        st.altair_chart(context_chart, use_container_width=True)

        st.subheader("Cumulative VWAP vs Cumulative Mid")
        cumulative_prices = build_cumulative_prices(trader_trades)
        if cumulative_prices.empty:
            st.info("No trader fills found for cumulative VWAP chart.")
        else:
            cumulative_prices["ts"] = pd.to_datetime(cumulative_prices["time"], unit="ms")
            cumulative_prices = downsample(cumulative_prices, max_points)
            price_chart = (
                alt.Chart(cumulative_prices)
                .mark_line()
                .encode(
                    x="ts:T",
                    y=alt.Y("price:Q", title="Price (ticks)"),
                    color=alt.Color("label:N", title="Series"),
                )
                .properties(height=260)
            )
            st.altair_chart(price_chart, use_container_width=True)

        st.subheader("ML Diagnostics")
        if decisions.empty:
            st.info("No ML decision data available.")
        else:
            ml_ready_df = decisions.copy()
            ml_ready_df["ml_ready"] = pd.to_numeric(
                ml_ready_df.get("ml_ready"), errors="coerce"
            ).fillna(0)
            ml_ready_df = ml_ready_df[ml_ready_df["ml_ready"] == 1]
            if not ml_scatter.empty:
                ml_scatter = ml_scatter.dropna(subset=["predicted_move", "actual_move"]).copy()
                corr_value = None
                if len(ml_scatter) > 1:
                    corr_value = ml_scatter["predicted_move"].corr(ml_scatter["actual_move"])
                sign_pred = np.sign(ml_scatter["predicted_move"])
                sign_act = np.sign(ml_scatter["actual_move"])
                sign_mask = (sign_pred != 0) & (sign_act != 0)
                hit_rate = None
                if sign_mask.any():
                    hit_rate = float((sign_pred[sign_mask] == sign_act[sign_mask]).mean() * 100.0)
                triggered_mean = None
                not_triggered_mean = None
                if "ml_triggered" in ml_scatter.columns:
                    triggered_mean = ml_scatter.loc[
                        ml_scatter["ml_triggered"] == 1, "actual_move"
                    ].mean()
                    not_triggered_mean = ml_scatter.loc[
                        ml_scatter["ml_triggered"] == 0, "actual_move"
                    ].mean()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "ML Corr (pred vs realized)",
                    f"{corr_value:.3f}" if corr_value is not None else "n/a",
                )
                m2.metric(
                    "ML Hit Rate (%)",
                    f"{hit_rate:.1f}" if hit_rate is not None else "n/a",
                )
                m3.metric(
                    "Mean Realized (ML Triggered)",
                    f"{triggered_mean:.2f}" if triggered_mean is not None else "n/a",
                )
                m4.metric(
                    "Mean Realized (No Trigger)",
                    f"{not_triggered_mean:.2f}" if not_triggered_mean is not None else "n/a",
                )

                st.caption(
                    f"Samples: {len(ml_scatter)} total, "
                    f"{int((ml_scatter['ml_triggered'] == 1).sum())} triggered, "
                    f"{int((ml_scatter['ml_triggered'] == 0).sum())} not triggered"
                )
            if ml_scatter.empty:
                st.info("Insufficient data for ML prediction scatter.")
            else:
                scatter = (
                    alt.Chart(ml_scatter)
                    .mark_circle(size=40, opacity=0.6)
                    .encode(
                        x=alt.X("predicted_move:Q", title="Predicted Move (ticks)"),
                        y=alt.Y("actual_move:Q", title="Realized Move (ticks)"),
                        color=alt.Color("ml_triggered:N", title="ML Triggered"),
                    )
                    .properties(height=240)
                )
                st.altair_chart(scatter, use_container_width=True)
            if ml_ready_df.empty:
                st.info("No ML-ready decisions for urgency distribution.")
            else:
                ml_ready_df["ml_urgency"] = pd.to_numeric(
                    ml_ready_df.get("ml_urgency"), errors="coerce"
                )
                hist = (
                    alt.Chart(ml_ready_df.dropna(subset=["ml_urgency"]))
                    .mark_bar()
                    .encode(
                        x=alt.X("ml_urgency:Q", bin=alt.Bin(maxbins=40), title="ML Urgency"),
                        y=alt.Y("count()", title="Count"),
                    )
                    .properties(height=240)
                )
                threshold = None
                if "ml_threshold" in ml_ready_df.columns:
                    thr_series = pd.to_numeric(
                        ml_ready_df["ml_threshold"], errors="coerce"
                    ).dropna()
                    if not thr_series.empty:
                        threshold = float(thr_series.iloc[0])
                if threshold is not None:
                    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(
                        color="red"
                    ).encode(x="x:Q")
                    hist = hist + rule
                st.altair_chart(hist, use_container_width=True)

        st.subheader("Slippage Distribution (bps)")
        if trader_trades.empty:
            st.info("No trader fills found for slippage distribution.")
        else:
            slippage = trader_trades.dropna(subset=["slippage_bps"])
            slippage = downsample(slippage, max_points)
            hist = (
                alt.Chart(slippage)
                .mark_bar()
                .encode(
                    x=alt.X("slippage_bps:Q", bin=alt.Bin(maxbins=40), title="Slippage (bps)"),
                    y=alt.Y("count()", title="Fill Count"),
                )
                .properties(height=260)
            )
            st.altair_chart(hist, use_container_width=True)

        st.subheader("Fill Ratio per Slice")
        fill_ratio = build_slice_fill_ratio(trader_trades, exec_orders)
        if fill_ratio.empty:
            st.info("No slice fill ratios available.")
        else:
            fill_ratio["ts"] = pd.to_datetime(fill_ratio["time"], unit="ms")
            fill_ratio = downsample(fill_ratio, max_points)
            ratio_chart = (
                alt.Chart(fill_ratio)
                .mark_circle(size=30, opacity=0.6)
                .encode(
                    x="ts:T",
                    y=alt.Y("fill_ratio:Q", title="Fill Ratio"),
                )
                .properties(height=220)
            )
            st.altair_chart(ratio_chart, use_container_width=True)


if __name__ == "__main__":
    main()
