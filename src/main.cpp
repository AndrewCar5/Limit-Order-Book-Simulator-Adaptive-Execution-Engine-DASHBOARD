#include <iostream>

#include "engine/simulator.h"

int main(int argc, char** argv) {
  std::string config_path = "configs/exec_bybit_single.cfg";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    }
  }

  auto config = lob::loadConfig(config_path);
  lob::Simulator simulator(config);

  if (config.replay_path.empty() || config.replay_type != "bybit_ob") {
    std::cerr << "Error: replay_path must be set and replay_type must be 'bybit_ob'.\n";
    return 1;
  }

  auto result = simulator.runReplayBybit("execution", config.replay_path);

  std::cout << "Run: execution" << "\n";
  std::cout << "Trades: " << result.trades << "\n";
  std::cout << "Maker/Taker fills: " << result.metrics.makerTrades()
            << " / " << result.metrics.takerTrades() << "\n";
  std::cout << "Exec qty (units): " << config.exec_qty << "\n";
  if (result.exec_target_qty > 0) {
    double completion =
        static_cast<double>(result.exec_filled_qty) /
        static_cast<double>(result.exec_target_qty);
    int shortfall = result.exec_target_qty - result.exec_filled_qty;
    std::cout << "Exec filled (units): " << result.exec_filled_qty << "\n";
    std::cout << "Completion (%): " << completion * 100.0 << "\n";
    std::cout << "End shortfall (units): " << shortfall << "\n";
    if (result.exec_submitted_qty > 0) {
      std::cout << "Exec submitted qty: " << result.exec_submitted_qty << "\n";
    }
    std::cout << "Exec forced IOC: " << result.exec_forced_ioc << "\n";
    std::cout << "Exec ML IOC: " << result.exec_ml_ioc << "\n";
    std::cout << "Exec skips: " << result.exec_skip_count << "\n";
    std::cout << "Exec max no-fill streak: " << result.exec_max_no_fill << "\n";
    if (result.exec_start_ts > 0) {
      std::cout << "Exec start (ms): " << result.exec_start_ts << "\n";
    }
    if (result.exec_end_ts > 0) {
      std::cout << "Exec end (ms): " << result.exec_end_ts << "\n";
    }
    if (result.exec_slice_ms > 0) {
      std::cout << "Exec slice (ms): " << result.exec_slice_ms << "\n";
    }
  }
  std::cout << "Position limit (units): " << config.max_inventory << "\n";
  std::cout << "Inventory: " << result.metrics.inventory() << "\n";
  std::cout << "Max abs inventory (observed): " << result.metrics.maxAbsInventory() << "\n";
  double markout = result.metrics.spreadCapture() + result.metrics.adverseSelection() -
                   result.metrics.fees();
  int mid = result.last_mid != 0 ? result.last_mid : config.initial_mid;
  std::cout << "MTM (cash+inv@mid): " << result.metrics.pnl(mid) << "\n";
  std::cout << "Spread capture: " << result.metrics.spreadCapture() << "\n";
  std::cout << "Fees: " << result.metrics.fees() << "\n";
  std::cout << "Markout PnL (@ horizon): " << markout << "\n\n";
  const auto& horizons = result.metrics.adverseHorizons();
  if (!horizons.empty()) {
    std::string label =
        (config.exec_mode != "none") ? "Markout (by horizon ms): "
                                     : "Adverse selection (by horizon ms): ";
    std::cout << label;
    for (size_t i = 0; i < horizons.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << horizons[i] << "=" << result.metrics.adverseSelectionAt(i);
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  uint64_t maker_trades = result.metrics.makerTrades();
  uint64_t taker_trades = result.metrics.takerTrades();
  double total_notional = result.metrics.totalNotional();
  double maker_notional = result.metrics.makerNotional();
  double taker_notional = result.metrics.takerNotional();
  double maker_fees = result.metrics.makerFees();
  double taker_fees = result.metrics.takerFees();
  double fees_per_trade =
      (maker_trades + taker_trades) > 0
          ? result.metrics.fees() / static_cast<double>(maker_trades + taker_trades)
          : 0.0;
  double fees_per_maker =
      maker_trades > 0 ? maker_fees / static_cast<double>(maker_trades) : 0.0;
  double fees_per_taker =
      taker_trades > 0 ? taker_fees / static_cast<double>(taker_trades) : 0.0;
  double avg_notional =
      (maker_trades + taker_trades) > 0
          ? total_notional / static_cast<double>(maker_trades + taker_trades)
          : 0.0;
  double avg_notional_maker =
      maker_trades > 0 ? maker_notional / static_cast<double>(maker_trades) : 0.0;
  double avg_notional_taker =
      taker_trades > 0 ? taker_notional / static_cast<double>(taker_trades) : 0.0;
  double adverse_per_maker =
      maker_trades > 0
          ? result.metrics.adverseSelection() / static_cast<double>(maker_trades)
          : 0.0;
  double pnl_per_maker =
      maker_trades > 0 ? result.metrics.pnl(mid) / static_cast<double>(maker_trades)
                       : 0.0;
  double edge_per_maker =
      maker_trades > 0
          ? result.metrics.spreadCapture() / static_cast<double>(maker_trades)
          : 0.0;

  std::cout << "Per maker trade (edge/adverse/fees/pnl): " << edge_per_maker << " / "
            << adverse_per_maker << " / " << fees_per_maker << " / " << pnl_per_maker
            << "\n\n";
  std::cout << "Fees (maker/taker/total): " << maker_fees << " / " << taker_fees
            << " / " << result.metrics.fees() << "\n";
  std::cout << "Fee per fill (maker/taker/total): " << fees_per_maker << " / "
            << fees_per_taker << " / " << fees_per_trade << "\n";
  std::cout << "Avg notional (maker/taker/total): " << avg_notional_maker << " / "
            << avg_notional_taker << " / " << avg_notional << "\n";

  return 0;
}
