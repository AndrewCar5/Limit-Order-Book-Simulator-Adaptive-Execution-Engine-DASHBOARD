#!/usr/bin/env python3
import argparse
import csv
import os

import pandas as pd


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(trades, book):
    total_trades = len(trades)
    mm_maker = sum(1 for t in trades if t["maker_is_mm"] == "1")
    mm_taker = sum(1 for t in trades if t["taker_is_mm"] == "1")

    spreads = [int(row["spread"]) for row in book if row["spread"]]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0.0

    notionals = []
    mm_notionals = []
    external_notionals = []
    maker_notional = 0.0
    taker_notional = 0.0
    for t in trades:
        try:
            price = float(t["price"])
            qty = float(t["qty"])
        except (KeyError, ValueError):
            continue
        notional = price * qty
        notionals.append(notional)
        maker_is_mm = t.get("maker_is_mm") == "1"
        taker_is_mm = t.get("taker_is_mm") == "1"
        if maker_is_mm or taker_is_mm:
            mm_notionals.append(notional)
        else:
            external_notionals.append(notional)
        if maker_is_mm:
            maker_notional += notional
        elif taker_is_mm:
            taker_notional += notional

    avg_notional = sum(notionals) / len(notionals) if notionals else 0.0
    avg_mm_notional = sum(mm_notionals) / len(mm_notionals) if mm_notionals else 0.0
    avg_external_notional = (
        sum(external_notionals) / len(external_notionals) if external_notionals else 0.0
    )
    median_notional = 0.0
    if notionals:
        notionals_sorted = sorted(notionals)
        mid = len(notionals_sorted) // 2
        if len(notionals_sorted) % 2 == 0:
            median_notional = (notionals_sorted[mid - 1] + notionals_sorted[mid]) / 2.0
        else:
            median_notional = notionals_sorted[mid]

    return {
        "total_trades": total_trades,
        "mm_maker_trades": mm_maker,
        "mm_taker_trades": mm_taker,
        "avg_spread": avg_spread,
        "avg_notional": avg_notional,
        "avg_notional_all": avg_notional,
        "avg_notional_mm": avg_mm_notional,
        "avg_notional_external": avg_external_notional,
        "median_notional": median_notional,
        "maker_notional": maker_notional,
        "taker_notional": taker_notional,
    }

def compute_maker_edge(trades):
    if not trades:
        return None
    required = {"maker_is_mm", "maker_side", "mid", "price", "qty", "time"}
    if not required.issubset(trades[0].keys()):
        return None

    maker_rows = []
    for row in trades:
        if row["maker_is_mm"] != "1":
            continue
        side = row["maker_side"]
        if side not in ("BUY", "SELL"):
            continue
        mid = float(row["mid"])
        price = float(row["price"])
        qty = float(row["qty"])
        edge = (mid - price) if side == "BUY" else (price - mid)
        maker_rows.append((int(row["time"]), edge * qty))

    if not maker_rows:
        return None
    maker_rows.sort(key=lambda x: x[0])
    times = [t for t, _ in maker_rows]
    cum_edge = []
    running = 0.0
    for _, edge in maker_rows:
        running += edge
        cum_edge.append(running)
    return times, cum_edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--orders", required=True)
    args = parser.parse_args()

    trades = read_csv(args.trades)
    book = read_csv(args.book)

    summary = summarize(trades, book)
    print("Metrics")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    try:
        import matplotlib.pyplot as plt

        spreads = [int(row["spread"]) for row in book if row["spread"]]
        times = [pd.to_datetime(int(row["time"]), unit="ms") for row in book if row["time"]]
        plt.figure(figsize=(8, 4))
        plt.plot(times, spreads, linewidth=1.0)
        plt.title("Spread Over Time")
        plt.xlabel("Time")
        plt.ylabel("Spread")
        out_path = os.path.join(os.path.dirname(args.book), "spread_plot.png")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved plot: {out_path}")

        maker_edge = compute_maker_edge(trades)
        if maker_edge:
            times, cum_edge = maker_edge
            times_dt = [pd.to_datetime(int(t), unit="ms") for t in times]
            plt.figure(figsize=(8, 4))
            plt.plot(times_dt, cum_edge, linewidth=1.0)
            plt.title("Cumulative Maker Edge (Ergodic Convergence)")
            plt.xlabel("Time")
            plt.ylabel("Cumulative Edge")
            out_path = os.path.join(os.path.dirname(args.trades), "maker_edge_cum.png")
            plt.tight_layout()
            plt.savefig(out_path)
            print(f"Saved plot: {out_path}")
    except Exception as exc:
        print(f"Plotting skipped ({exc})")


if __name__ == "__main__":
    main()
