#include "engine/metrics.h"

#include <algorithm>
#include <cmath>

namespace lob {

void Metrics::setAdverseHorizons(const std::vector<uint64_t>& horizons) {
  adverse_horizons_ = horizons;
  if (adverse_horizons_.empty()) {
    adverse_horizons_.push_back(50);
  }
  std::sort(adverse_horizons_.begin(), adverse_horizons_.end());
  adverse_horizons_.erase(std::unique(adverse_horizons_.begin(), adverse_horizons_.end()),
                          adverse_horizons_.end());
  adverse_by_horizon_.assign(adverse_horizons_.size(), 0.0);
}

void Metrics::onFill(uint64_t time,
                     Side side,
                     int qty,
                     int price,
                     int mid,
                     bool maker,
                     double fee_bps,
                     double rebate_bps,
                     int markout_bias) {
  int signed_qty = (side == Side::Buy) ? qty : -qty;
  double notional = static_cast<double>(price) * qty;
  total_notional_ += notional;

  if (side == Side::Buy) {
    inventory_ += qty;
    cash_ -= notional;
  } else {
    inventory_ -= qty;
    cash_ += notional;
  }
  max_abs_inventory_ = std::max(max_abs_inventory_, std::abs(inventory_));

  double fee_rate = maker ? -rebate_bps : fee_bps;
  double fee = notional * (fee_rate / 10000.0);
  fees_ += fee;
  if (maker) {
    maker_fees_ += fee;
    maker_notional_ += notional;
  } else {
    taker_fees_ += fee;
    taker_notional_ += notional;
  }
  cash_ -= fee;

  if (maker) {
    maker_trades_++;
    double capture = (side == Side::Buy)
                         ? (static_cast<double>(mid - price) * qty)
                         : (static_cast<double>(price - mid) * qty);
    spread_capture_ += capture;
  } else {
    taker_trades_++;
  }

  pending_.push_back({time, signed_qty, mid, markout_bias, 0});
}

void Metrics::updateAdverseSelection(uint64_t time, int mid) {
  if (adverse_horizons_.empty()) {
    adverse_horizons_.push_back(50);
    adverse_by_horizon_.assign(1, 0.0);
  }
  auto it = std::remove_if(pending_.begin(), pending_.end(),
                           [&](PendingAdverse& pending) {
                             while (pending.next_horizon_idx < adverse_horizons_.size()) {
                               uint64_t horizon = adverse_horizons_[pending.next_horizon_idx];
                               if (time < pending.time + horizon) {
                                 break;
                               }
                               double move = static_cast<double>(
                                   mid - pending.mid_at_trade + pending.bias);
                               adverse_by_horizon_[pending.next_horizon_idx] +=
                                   move * pending.qty_signed;
                               pending.next_horizon_idx++;
                             }
                             return pending.next_horizon_idx >= adverse_horizons_.size();
                           });
  pending_.erase(it, pending_.end());
  adverse_selection_ = adverse_by_horizon_.front();
}

}  // namespace lob
