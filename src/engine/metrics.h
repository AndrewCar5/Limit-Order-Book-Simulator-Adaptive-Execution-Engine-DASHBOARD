#pragma once

#include <cstdint>
#include <vector>

#include "engine/order.h"

namespace lob {

struct PendingAdverse {
  uint64_t time = 0;
  int qty_signed = 0;
  int mid_at_trade = 0;
  int bias = 0;
  size_t next_horizon_idx = 0;
};

class Metrics {
 public:
  void onFill(uint64_t time,
              Side side,
              int qty,
              int price,
              int mid,
              bool maker,
              double fee_bps,
              double rebate_bps,
              int markout_bias = 0);

  void setAdverseHorizons(const std::vector<uint64_t>& horizons);
  void updateAdverseSelection(uint64_t time, int mid);

  double cash() const { return cash_; }
  int inventory() const { return inventory_; }
  double fees() const { return fees_; }
  double spreadCapture() const { return spread_capture_; }
  double adverseSelection() const { return adverse_selection_; }
  double adverseSelectionAt(size_t idx) const { return adverse_by_horizon_.at(idx); }
  const std::vector<uint64_t>& adverseHorizons() const { return adverse_horizons_; }
  double pnl(int mid) const { return cash_ + static_cast<double>(inventory_) * mid; }
  uint64_t makerTrades() const { return maker_trades_; }
  uint64_t takerTrades() const { return taker_trades_; }
  double makerFees() const { return maker_fees_; }
  double takerFees() const { return taker_fees_; }
  double totalNotional() const { return total_notional_; }
  double makerNotional() const { return maker_notional_; }
  double takerNotional() const { return taker_notional_; }
  uint64_t totalFills() const { return maker_trades_ + taker_trades_; }
  int maxAbsInventory() const { return max_abs_inventory_; }

 private:
  double cash_ = 0.0;
  int inventory_ = 0;
  int max_abs_inventory_ = 0;
  double fees_ = 0.0;
  double spread_capture_ = 0.0;
  double adverse_selection_ = 0.0;
  std::vector<uint64_t> adverse_horizons_;
  std::vector<double> adverse_by_horizon_;
  uint64_t maker_trades_ = 0;
  uint64_t taker_trades_ = 0;
  double maker_fees_ = 0.0;
  double taker_fees_ = 0.0;
  double total_notional_ = 0.0;
  double maker_notional_ = 0.0;
  double taker_notional_ = 0.0;
  std::vector<PendingAdverse> pending_;
};

}  // namespace lob
