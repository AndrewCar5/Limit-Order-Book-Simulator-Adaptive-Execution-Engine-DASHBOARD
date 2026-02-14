#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/metrics.h"
#include "engine/matching_engine.h"
namespace lob {

struct SimConfig {
  int initial_mid = 10000;
  int max_inventory = 200;
  double fee_bps = 2.0;
  double rebate_bps = 0.0;
  int price_scale = 100;
  int qty_scale = 1000;
  int replay_skip = 1;
  int replay_max_steps = 0;
  std::string output_dir = "outputs";
  int use_trade_feed = 0;
  std::string trade_path;
  std::string replay_path;
  std::string replay_type;
  uint64_t adverse_horizon_ms = 50;
  std::vector<uint64_t> adverse_horizons_ms;
  std::string exec_mode = "none";
  std::string exec_side = "BUY";
  int exec_qty = 0;
  int exec_limit_offset_ticks = 0;
  uint64_t exec_start_ms = 0;
  uint64_t exec_end_ms = 0;
  int exec_slice_ms = 1000;
  std::string exec_tif = "GTC";
  int exec_post_only = 0;
  int exec_is_market = 0;
  std::string exec_price_mode = "touch";
  std::string exec_policy = "fixed";
  int exec_min_slice_qty = 1;
  int exec_max_slice_qty = 0;
  double exec_behind_threshold = 0.02;
  double exec_ahead_threshold = 0.02;
  double exec_panic_threshold = 0.15;
  int exec_spread_max_ticks = 0;
  double exec_imbalance_max = 1.0;
  double exec_vol_max = 0.0;
  int exec_passive_offset_ticks = 0;
  int exec_skip_when_toxic = 1;
  int exec_max_passive_slices = 3;
  int exec_final_sweep_slices = 0;
  uint64_t exec_final_sweep_ms = 0;
  int exec_ml_enabled = 1;
  uint64_t exec_ml_horizon_ms = 1000;
  double exec_ml_learning_rate = 0.01;
  double exec_ml_l2 = 0.0001;
  int exec_ml_warmup_samples = 20;
  double exec_ml_aggression_threshold = 0.0;
  uint64_t day_duration_ms = 86400000;
  uint64_t day_anchor_ms = 0;
  uint64_t exec_gtd_expire_ms = 0;
};

struct SimResult {
  Metrics metrics;
  uint64_t trades = 0;
  int last_mid = 0;
  int exec_target_qty = 0;
  int exec_filled_qty = 0;
  int exec_submitted_qty = 0;
  int exec_forced_ioc = 0;
  int exec_ml_ioc = 0;
  int exec_skip_count = 0;
  int exec_max_no_fill = 0;
  uint64_t exec_start_ts = 0;
  uint64_t exec_end_ts = 0;
  int exec_slice_ms = 0;
  std::unordered_map<std::string, int> exec_action_counts;
};

class Simulator {
 public:
  explicit Simulator(SimConfig config);

  SimResult runReplayBybit(const std::string& output_prefix, const std::string& data_path);

 private:
  SimConfig config_;
  OrderBook book_;
  MatchingEngine engine_;
  uint64_t next_order_id_ = 1;

  uint64_t nextId() { return next_order_id_++; }

  int midPrice() const;
  int spread() const;
  uint64_t dayExpiry(uint64_t time) const;
  void expireBookOrders(std::ofstream& order_log, uint64_t now);

  void logBook(std::ofstream& out, uint64_t time, uint64_t mm_bid_id = 0, uint64_t mm_ask_id = 0);
  void logOrder(std::ofstream& out, const Order& order, const std::string& action);
  void logTrade(std::ofstream& out, const Trade& trade, int mid, int inventory);
};

SimConfig loadConfig(const std::string& path);

}  // namespace lob
