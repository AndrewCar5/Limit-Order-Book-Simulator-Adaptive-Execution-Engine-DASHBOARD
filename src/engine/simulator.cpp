#include "engine/simulator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>

namespace lob {

namespace {

int clampPrice(int price) {
  return std::max(1, price);
}

Side parseSide(const std::string& value) {
  if (value == "SELL" || value == "sell") {
    return Side::Sell;
  }
  return Side::Buy;
}

TimeInForce parseTif(const std::string& value) {
  if (value == "IOC" || value == "ioc") {
    return TimeInForce::IOC;
  }
  if (value == "FOK" || value == "fok") {
    return TimeInForce::FOK;
  }
  if (value == "DAY" || value == "day") {
    return TimeInForce::DAY;
  }
  if (value == "GTD" || value == "gtd") {
    return TimeInForce::GTD;
  }
  return TimeInForce::GTC;
}


std::string extractArray(const std::string& line, const std::string& key) {
  std::string token = "\"" + key + "\":[";
  auto start = line.find(token);
  if (start == std::string::npos) {
    return {};
  }
  start = line.find('[', start);
  if (start == std::string::npos) {
    return {};
  }

  int depth = 0;
  for (size_t i = start; i < line.size(); ++i) {
    if (line[i] == '[') {
      depth++;
    } else if (line[i] == ']') {
      depth--;
      if (depth == 0) {
        return line.substr(start, i - start + 1);
      }
    }
  }

  return {};
}

void parseLevels(const std::string& array_text,
                 std::vector<std::pair<double, double>>& out) {
  out.clear();
  size_t pos = 0;
  while (true) {
    auto p1 = array_text.find("[\"", pos);
    if (p1 == std::string::npos) {
      break;
    }
    auto p2 = array_text.find("\",\"", p1 + 2);
    if (p2 == std::string::npos) {
      break;
    }
    auto p3 = array_text.find("\"]", p2 + 3);
    if (p3 == std::string::npos) {
      break;
    }
    std::string price_str = array_text.substr(p1 + 2, p2 - (p1 + 2));
    std::string qty_str = array_text.substr(p2 + 3, p3 - (p2 + 3));
    out.emplace_back(std::stod(price_str), std::stod(qty_str));
    pos = p3 + 2;
  }
}

uint64_t parseTimestamp(const std::string& line) {
  auto pos = line.find("\"ts\":");
  if (pos == std::string::npos) {
    return 0;
  }
  pos += 5;
  while (pos < line.size() && (line[pos] < '0' || line[pos] > '9')) {
    ++pos;
  }
  uint64_t value = 0;
  while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
    value = value * 10 + static_cast<uint64_t>(line[pos] - '0');
    ++pos;
  }
  return value;
}

bool isSnapshot(const std::string& line) {
  return line.find("\"type\":\"snapshot\"") != std::string::npos;
}

enum class ExecAction { Skip, IocTouch, PassiveLimit, PostOnlyPassive };

double clampUnit(double x) {
  if (x < 0.0) {
    return 0.0;
  }
  if (x > 1.0) {
    return 1.0;
  }
  return x;
}

bool isAdaptivePolicy(const std::string& value) {
  return value == "adaptive" || value == "ADAPTIVE";
}

std::array<double, 5> makeExecFeatures(const OrderBook& book, int mid, double vol_proxy) {
  double spread = 0.0;
  auto bid = book.bestBid();
  auto ask = book.bestAsk();
  if (bid && ask) {
    spread = static_cast<double>(*ask - *bid);
  }

  double bid_depth = static_cast<double>(book.depth(Side::Buy, 5));
  double ask_depth = static_cast<double>(book.depth(Side::Sell, 5));
  double denom = bid_depth + ask_depth;
  double imbalance = (denom > 0.0) ? (bid_depth - ask_depth) / denom : 0.0;
  double micro_delta = book.microPrice(5) - static_cast<double>(mid);
  return {1.0, spread, imbalance, micro_delta, vol_proxy};
}

std::string actionToLog(ExecAction action) {
  switch (action) {
    case ExecAction::IocTouch:
      return "exec_ioc_touch";
    case ExecAction::PassiveLimit:
      return "exec_passive";
    case ExecAction::PostOnlyPassive:
      return "exec_post_only";
    default:
      return "exec_skip";
  }
}

struct PendingMarkoutSample {
  uint64_t ts = 0;
  int mid = 0;
  std::array<double, 5> x{};
};

class OnlineMarkoutModel {
 public:
  explicit OnlineMarkoutModel(const SimConfig& cfg)
      : horizon_ms_(std::max<uint64_t>(1, cfg.exec_ml_horizon_ms)),
        lr_(cfg.exec_ml_learning_rate),
        l2_(cfg.exec_ml_l2),
        warmup_samples_(std::max(0, cfg.exec_ml_warmup_samples)) {
    weights_.fill(0.0);
  }

  void addSample(uint64_t ts, int mid, const std::array<double, 5>& x) {
    pending_.push_back({ts, mid, x});
  }

  void observe(uint64_t ts, int mid) {
    while (!pending_.empty()) {
      const auto& sample = pending_.front();
      if (ts < sample.ts + horizon_ms_) {
        break;
      }
      train(sample, mid);
      pending_.pop_front();
    }
  }

  bool ready() const { return trained_samples_ >= warmup_samples_; }

  double predict(const std::array<double, 5>& x) const {
    double out = 0.0;
    for (size_t i = 0; i < weights_.size(); ++i) {
      out += weights_[i] * x[i];
    }
    return out;
  }

 private:
  void train(const PendingMarkoutSample& sample, int current_mid) {
    if (lr_ <= 0.0) {
      trained_samples_++;
      return;
    }
    double target = static_cast<double>(current_mid - sample.mid);
    double pred = predict(sample.x);
    double err = pred - target;
    for (size_t i = 0; i < weights_.size(); ++i) {
      weights_[i] -= lr_ * (err * sample.x[i] + l2_ * weights_[i]);
    }
    trained_samples_++;
  }

  uint64_t horizon_ms_ = 1000;
  double lr_ = 0.01;
  double l2_ = 0.0001;
  int warmup_samples_ = 20;
  int trained_samples_ = 0;
  std::array<double, 5> weights_{};
  std::deque<PendingMarkoutSample> pending_;
};

}  // namespace

Simulator::Simulator(SimConfig config)
    : config_(config), engine_(book_) {}

SimResult Simulator::runReplayBybit(const std::string& output_prefix,
                                    const std::string& data_path) {
  std::filesystem::create_directories(config_.output_dir);

  std::ofstream book_log(config_.output_dir + "/" + output_prefix + "_book.csv");
  std::ofstream trade_log(config_.output_dir + "/" + output_prefix + "_trades.csv");
  std::ofstream order_log(config_.output_dir + "/" + output_prefix + "_orders.csv");
  std::ofstream decisions_log(config_.output_dir + "/" + output_prefix + "_decisions.csv");

  book_log << "time,bid,ask,mid,spread,microprice,bid_depth,ask_depth,imbalance,mm_bid_queue,mm_ask_queue\n";
  trade_log << "time,price,qty,maker_id,taker_id,maker_is_mm,taker_is_mm,maker_side,mid,mm_inventory\n";
  order_log << "time,order_id,side,price,qty,remaining,owner,tif,post_only,action\n";
  decisions_log << "time,remaining_qty,slice_qty,mid_now,best_bid,best_ask,spread_ticks,"
                   "imbalance,vol_ema,time_ratio,fill_ratio,schedule_error,ml_ready,"
                   "ml_predicted_move,ml_urgency,ml_triggered,forced_ioc,skip,raw_action,"
                   "final_action,ml_horizon_ms,ml_threshold\n";

  Metrics metrics;
  if (!config_.adverse_horizons_ms.empty()) {
    metrics.setAdverseHorizons(config_.adverse_horizons_ms);
  } else {
    metrics.setAdverseHorizons({config_.adverse_horizon_ms});
  }
  uint64_t trades = 0;
  int last_mid = config_.initial_mid;
  bool exec_enabled = config_.exec_mode != "none" && config_.exec_qty > 0;
  bool exec_initialized = false;
  Side exec_side = parseSide(config_.exec_side);
  TimeInForce exec_tif = parseTif(config_.exec_tif);
  bool adaptive_policy = isAdaptivePolicy(config_.exec_policy);
  int exec_target_qty = std::max(0, config_.exec_qty);
  int exec_filled_qty = 0;
  uint64_t exec_next_ts = config_.exec_start_ms;
  uint64_t exec_start_ts = 0;
  uint64_t exec_end_ts = config_.exec_end_ms;
  bool exec_window_resolved = false;
  int exec_prev_mid = 0;
  double exec_vol_ema = 0.0;
  int exec_slices_without_fill = 0;
  int exec_submitted_qty = 0;
  int exec_forced_ioc = 0;
  int exec_ml_ioc = 0;
  int exec_skip_count = 0;
  int exec_max_no_fill = 0;
  std::unordered_map<std::string, int> exec_action_counts;
  OnlineMarkoutModel ml_model(config_);

  std::function<void(const std::vector<Trade>&, int)> handleTrades;
  handleTrades = [&](const std::vector<Trade>& trades_out, int mid) {
    for (const auto& trade : trades_out) {
      trades++;
      if (trade.maker_is_mm || trade.taker_is_mm) {
        int filled_before = exec_filled_qty;
        Side mm_side = trade.maker_is_mm ? trade.maker_side
                                         : (trade.maker_side == Side::Buy ? Side::Sell : Side::Buy);
        bool mm_is_maker = trade.maker_is_mm;
        metrics.onFill(trade.time,
                       mm_side,
                       trade.qty,
                       trade.price,
                       mid,
                       mm_is_maker,
                       config_.fee_bps,
                       config_.rebate_bps);
        if (exec_enabled) {
          if (mm_side == exec_side) {
            exec_filled_qty += trade.qty;
          } else {
            exec_filled_qty -= trade.qty;
          }
          exec_filled_qty = std::clamp(exec_filled_qty, 0, exec_target_qty);
          if (exec_filled_qty > filled_before) {
            exec_slices_without_fill = 0;
          }
        }
      }
      logTrade(trade_log, trade, mid, metrics.inventory());
    }
  };

  auto submitExecution = [&](uint64_t ts, int mid) {
    if (!exec_enabled || exec_target_qty <= 0) {
      return;
    }

    if (mid > 0) {
      if (exec_prev_mid > 0) {
        double ret = static_cast<double>(mid - exec_prev_mid);
        exec_vol_ema = 0.95 * exec_vol_ema + 0.05 * std::abs(ret);
      }
      exec_prev_mid = mid;
      if (config_.exec_ml_enabled != 0) {
        ml_model.observe(ts, mid);
      }
    }

    if (!exec_window_resolved) {
      constexpr uint64_t kEpochMsCutoff = 946684800000ULL;  // 2000-01-01 UTC.
      if (exec_next_ts < kEpochMsCutoff) {
        exec_next_ts = ts + exec_next_ts;
      }
      if (exec_end_ts > 0 && exec_end_ts < kEpochMsCutoff) {
        exec_end_ts = ts + exec_end_ts;
      }
      exec_window_resolved = true;
    }
    if (!exec_initialized) {
      if (exec_next_ts == 0) {
        exec_next_ts = ts;
      }
      exec_start_ts = exec_next_ts;
      if (config_.exec_mode == "twap") {
        if (exec_end_ts == 0 && config_.exec_slice_ms > 0) {
          exec_end_ts = exec_next_ts + static_cast<uint64_t>(config_.exec_slice_ms * 10);
        }
        if (exec_end_ts <= exec_next_ts || config_.exec_slice_ms <= 0) {
          exec_enabled = false;
          return;
        }
      }
      exec_initialized = true;
    }

    auto placeOrder = [&](ExecAction action, int qty, bool forced_ioc) {
      if (qty <= 0) {
        return;
      }
      int filled_before = exec_filled_qty;
      auto best_bid = book_.bestBid();
      auto best_ask = book_.bestAsk();
      int inv = metrics.inventory();
      if (config_.max_inventory > 0) {
        if (exec_side == Side::Buy && inv + qty > config_.max_inventory) {
          qty = std::max(0, config_.max_inventory - inv);
        } else if (exec_side == Side::Sell && inv - qty < -config_.max_inventory) {
          qty = std::max(0, inv + config_.max_inventory);
        }
      }
      if (qty <= 0) {
        return;
      }
      exec_submitted_qty += qty;
      auto order = std::make_shared<Order>();
      order->id = nextId();
      order->owner = Owner::MarketMaker;
      order->time = ts;
      order->qty = qty;
      order->remaining = qty;
      order->side = exec_side;
      order->is_market = false;
      order->post_only = false;

      if (action == ExecAction::IocTouch) {
        order->tif = TimeInForce::IOC;
        order->is_market = config_.exec_is_market != 0;
      } else {
        order->tif = exec_tif;
        if (order->tif == TimeInForce::IOC || order->tif == TimeInForce::FOK) {
          order->tif = TimeInForce::GTC;
        }
        order->post_only = (action == ExecAction::PostOnlyPassive);
      }

      if (order->tif == TimeInForce::DAY) {
        order->expire_time = dayExpiry(ts);
      } else if (order->tif == TimeInForce::GTD) {
        order->expire_time = (config_.exec_gtd_expire_ms > ts) ? config_.exec_gtd_expire_ms : ts;
      }

      int price = mid;
      if (!order->is_market) {
        if (action == ExecAction::IocTouch && config_.exec_price_mode == "touch") {
          if (exec_side == Side::Buy) {
            price = best_ask ? *best_ask : mid;
            price += config_.exec_limit_offset_ticks;
          } else {
            price = best_bid ? *best_bid : mid;
            price -= config_.exec_limit_offset_ticks;
          }
        } else if (action == ExecAction::PassiveLimit ||
                   action == ExecAction::PostOnlyPassive) {
          int offset = std::max(0, config_.exec_passive_offset_ticks);
          if (exec_side == Side::Buy) {
            int base = best_bid.value_or(mid);
            price = base - offset;
          } else {
            int base = best_ask.value_or(mid);
            price = base + offset;
          }
        } else {
          price = mid + (exec_side == Side::Buy ? config_.exec_limit_offset_ticks
                                                : -config_.exec_limit_offset_ticks);
        }
        price = clampPrice(price);
      }
      order->price = price;
      auto order_trades = engine_.process(order);
      std::string action_label =
          forced_ioc ? "exec_forced_ioc" : actionToLog(action);
      exec_action_counts[action_label]++;
      logOrder(order_log, *order, action_label);
      handleTrades(order_trades, mid);
      if (exec_filled_qty > filled_before) {
        exec_slices_without_fill = 0;
      } else {
        exec_slices_without_fill++;
        exec_max_no_fill = std::max(exec_max_no_fill, exec_slices_without_fill);
      }
    };

    struct ExecDecision {
      ExecAction raw_action = ExecAction::PassiveLimit;
      double time_ratio = 0.0;
      double fill_ratio = 0.0;
      double schedule_error = 0.0;
      int spread_ticks = 0;
      double imbalance = 0.0;
      double vol_ema = 0.0;
      int best_bid = 0;
      int best_ask = 0;
      int ml_ready = 0;
      double ml_predicted_move = 0.0;
      double ml_urgency = 0.0;
      int ml_triggered = 0;
    };

    auto decideAction = [&](uint64_t now_ts, bool allow_skip) {
      ExecDecision decision;
      decision.vol_ema = exec_vol_ema;

      auto best_bid = book_.bestBid();
      auto best_ask = book_.bestAsk();
      if (best_bid) {
        decision.best_bid = *best_bid;
      }
      if (best_ask) {
        decision.best_ask = *best_ask;
      }
      if (best_bid && best_ask) {
        decision.spread_ticks = *best_ask - *best_bid;
      }

      double bid_depth = static_cast<double>(book_.depth(Side::Buy, 5));
      double ask_depth = static_cast<double>(book_.depth(Side::Sell, 5));
      double depth_denom = bid_depth + ask_depth;
      decision.imbalance =
          (depth_denom > 0.0) ? (bid_depth - ask_depth) / depth_denom : 0.0;

      if (exec_end_ts > exec_start_ts) {
        uint64_t elapsed = (now_ts > exec_start_ts) ? (now_ts - exec_start_ts) : 0;
        uint64_t duration = exec_end_ts - exec_start_ts;
        decision.time_ratio =
            clampUnit(static_cast<double>(elapsed) / static_cast<double>(duration));
      }
      decision.fill_ratio =
          (exec_target_qty > 0)
              ? clampUnit(static_cast<double>(exec_filled_qty) /
                          static_cast<double>(exec_target_qty))
              : 1.0;
      decision.schedule_error = decision.time_ratio - decision.fill_ratio;

      bool toxic = false;
      if (config_.exec_spread_max_ticks > 0 &&
          decision.spread_ticks > config_.exec_spread_max_ticks) {
        toxic = true;
      }
      if (config_.exec_imbalance_max < 1.0 &&
          std::abs(decision.imbalance) > config_.exec_imbalance_max) {
        toxic = true;
      }
      if (config_.exec_vol_max > 0.0 && exec_vol_ema > config_.exec_vol_max) {
        toxic = true;
      }

      if (!adaptive_policy) {
        if (config_.exec_is_market != 0 || exec_tif == TimeInForce::IOC ||
            exec_tif == TimeInForce::FOK) {
          decision.raw_action = ExecAction::IocTouch;
        } else if (config_.exec_post_only != 0) {
          decision.raw_action = ExecAction::PostOnlyPassive;
        } else {
          decision.raw_action = ExecAction::PassiveLimit;
        }
        return decision;
      }

      auto features = makeExecFeatures(book_, mid, exec_vol_ema);
      if (config_.exec_ml_enabled != 0) {
        ml_model.addSample(now_ts, mid, features);
      }

      if (config_.exec_ml_enabled != 0 && ml_model.ready()) {
        decision.ml_ready = 1;
        decision.ml_predicted_move = ml_model.predict(features);
        decision.ml_urgency =
            (exec_side == Side::Buy) ? decision.ml_predicted_move
                                     : -decision.ml_predicted_move;
      }

      int max_passive_slices = std::max(0, config_.exec_max_passive_slices);
      if (max_passive_slices > 0 &&
          exec_slices_without_fill >= max_passive_slices) {
        decision.raw_action = ExecAction::IocTouch;
        return decision;
      }

      if (toxic) {
        if (decision.schedule_error > config_.exec_panic_threshold) {
          decision.raw_action = ExecAction::IocTouch;
          return decision;
        }
        if (allow_skip && config_.exec_skip_when_toxic != 0) {
          decision.raw_action = ExecAction::Skip;
          return decision;
        }
        decision.raw_action = ExecAction::PostOnlyPassive;
        return decision;
      }
      if (decision.schedule_error > config_.exec_behind_threshold) {
        decision.raw_action = ExecAction::IocTouch;
        return decision;
      }
      if (decision.schedule_error < -config_.exec_ahead_threshold) {
        decision.raw_action = ExecAction::PostOnlyPassive;
        return decision;
      }
      if (config_.exec_ml_enabled != 0 && decision.ml_ready != 0 &&
          decision.ml_urgency > config_.exec_ml_aggression_threshold) {
        exec_ml_ioc++;
        decision.ml_triggered = 1;
        decision.raw_action = ExecAction::IocTouch;
        return decision;
      }
      decision.raw_action = ExecAction::PassiveLimit;
      return decision;
    };

    auto clampSliceQty = [&](int qty) {
      if (qty <= 0) {
        return 0;
      }
      int min_qty = std::max(1, config_.exec_min_slice_qty);
      int max_qty =
          (config_.exec_max_slice_qty > 0) ? config_.exec_max_slice_qty : qty;
      qty = std::max(min_qty, qty);
      qty = std::min(max_qty, qty);
      return qty;
    };

    auto remainingQty = [&]() { return std::max(0, exec_target_qty - exec_filled_qty); };

    if (config_.exec_mode == "single") {
      if (ts < exec_next_ts) {
        return;
      }
      int qty = clampSliceQty(remainingQty());
      ExecDecision decision = decideAction(ts, false);
      ExecAction final_action = decision.raw_action;
      if (final_action == ExecAction::Skip) {
        final_action = ExecAction::IocTouch;
      }
      std::string raw_label = actionToLog(decision.raw_action);
      std::string final_label = actionToLog(final_action);
      decisions_log << ts << "," << remainingQty() << "," << qty << "," << mid << ","
                    << decision.best_bid << "," << decision.best_ask << ","
                    << decision.spread_ticks << "," << decision.imbalance << ","
                    << decision.vol_ema << "," << decision.time_ratio << ","
                    << decision.fill_ratio << "," << decision.schedule_error << ","
                    << decision.ml_ready << "," << decision.ml_predicted_move << ","
                    << decision.ml_urgency << "," << decision.ml_triggered << ",0,0,"
                    << raw_label << "," << final_label << ","
                    << config_.exec_ml_horizon_ms << ","
                    << config_.exec_ml_aggression_threshold << "\n";
      placeOrder(final_action, qty, false);
      exec_enabled = false;
      return;
    }

    if (config_.exec_mode == "twap") {
      if (ts < exec_next_ts || ts > exec_end_ts) {
        return;
      }
      while (remainingQty() > 0 && ts >= exec_next_ts && ts <= exec_end_ts) {
        uint64_t slices_left_u64 =
            (exec_end_ts >= exec_next_ts)
                ? ((exec_end_ts - exec_next_ts +
                    static_cast<uint64_t>(config_.exec_slice_ms) - 1) /
                   static_cast<uint64_t>(config_.exec_slice_ms) +
                   1)
                : 1;
        int slices_left = std::max(1, static_cast<int>(slices_left_u64));
        int qty = (remainingQty() + slices_left - 1) / slices_left;
        qty = clampSliceQty(qty);
        uint64_t time_remaining =
            (exec_end_ts > exec_next_ts) ? (exec_end_ts - exec_next_ts) : 0;
        bool in_sweep = false;
        if (config_.exec_final_sweep_slices > 0 &&
            slices_left <= config_.exec_final_sweep_slices) {
          in_sweep = true;
        }
        if (config_.exec_final_sweep_ms > 0 &&
            time_remaining <= config_.exec_final_sweep_ms) {
          in_sweep = true;
        }
        ExecDecision decision = decideAction(exec_next_ts, !in_sweep);
        ExecAction final_action = decision.raw_action;
        bool forced_ioc = false;
        if (in_sweep && final_action != ExecAction::IocTouch) {
          exec_forced_ioc++;
          forced_ioc = true;
          final_action = ExecAction::IocTouch;
        }
        bool skipped = final_action == ExecAction::Skip;
        std::string raw_label = actionToLog(decision.raw_action);
        std::string final_label =
            forced_ioc ? "exec_forced_ioc" : actionToLog(final_action);
        decisions_log << exec_next_ts << "," << remainingQty() << "," << qty << ","
                      << mid << "," << decision.best_bid << "," << decision.best_ask
                      << "," << decision.spread_ticks << "," << decision.imbalance
                      << "," << decision.vol_ema << "," << decision.time_ratio << ","
                      << decision.fill_ratio << "," << decision.schedule_error << ","
                      << decision.ml_ready << "," << decision.ml_predicted_move << ","
                      << decision.ml_urgency << "," << decision.ml_triggered << ","
                      << (forced_ioc ? 1 : 0) << "," << (skipped ? 1 : 0) << ","
                      << raw_label << "," << final_label << ","
                      << config_.exec_ml_horizon_ms << ","
                      << config_.exec_ml_aggression_threshold << "\n";
        if (!skipped) {
          placeOrder(final_action, qty, forced_ioc);
        } else {
          exec_slices_without_fill++;
          exec_skip_count++;
          exec_max_no_fill = std::max(exec_max_no_fill, exec_slices_without_fill);
        }
        exec_next_ts += static_cast<uint64_t>(config_.exec_slice_ms);
      }
    }
  };

  struct TradeRow {
    uint64_t ts = 0;
    double qty = 0.0;
    Side side = Side::Buy;
  };

  class TradeFeed {
   public:
    explicit TradeFeed(const std::string& path) {
      std::error_code ec;
      if (std::filesystem::is_directory(path, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          if (!entry.is_regular_file()) {
            continue;
          }
          if (entry.path().extension() == ".csv") {
            files_.push_back(entry.path().string());
          }
        }
        std::sort(files_.begin(), files_.end());
      } else {
        files_.push_back(path);
      }
      openNextFile();
    }

    bool hasNext() const { return has_next_; }
    const TradeRow& peek() const { return current_; }
    void advance() { readNext(); }

   private:
    std::vector<std::string> files_;
    size_t file_index_ = 0;
    std::ifstream file_;
    TradeRow current_;
    bool has_next_ = false;

    void openNextFile() {
      file_.close();
      while (file_index_ < files_.size()) {
        file_.open(files_[file_index_++]);
        if (file_) {
          std::string header;
          std::getline(file_, header);
          break;
        }
      }
      readNext();
    }

    void readNext() {
      std::string line;
      while (true) {
        if (!file_) {
          has_next_ = false;
          return;
        }
        if (!std::getline(file_, line)) {
          openNextFile();
          continue;
        }
        if (line.empty()) {
          continue;
        }
        if (line[0] < '0' || line[0] > '9') {
          continue;
        }
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> cols;
        while (std::getline(ss, token, ',')) {
          cols.push_back(token);
        }
        if (cols.size() < 5) {
          continue;
        }
        uint64_t ts = 0;
        double qty = 0.0;
        try {
          ts = static_cast<uint64_t>(std::stoull(cols[1]));
          qty = std::stod(cols[3]);
        } catch (...) {
          continue;
        }
        std::string side_str = cols[4];
        Side side = (side_str == "buy") ? Side::Buy : Side::Sell;
        current_ = {ts, qty, side};
        has_next_ = true;
        return;
      }
    }
  };

  std::unique_ptr<TradeFeed> trade_feed;
  if (config_.use_trade_feed != 0 && !config_.trade_path.empty()) {
    trade_feed = std::make_unique<TradeFeed>(config_.trade_path);
  }

  std::vector<std::string> files;
  std::error_code ec;
  if (std::filesystem::is_directory(data_path, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(data_path)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().extension() == ".data") {
        files.push_back(entry.path().string());
      }
    }
    std::sort(files.begin(), files.end());
  } else {
    files.push_back(data_path);
  }

  if (files.empty()) {
    std::cerr << "No replay files found in: " << data_path << "\n";
    SimResult result;
    result.metrics = metrics;
    result.trades = 0;
    result.last_mid = last_mid;
    return result;
  }

  std::unordered_map<int, int> bids;
  std::unordered_map<int, int> asks;
  uint64_t event_index = 0;
  uint64_t processed = 0;

  for (const auto& path : files) {
    std::ifstream file(path);
    if (!file) {
      std::cerr << "Failed to open replay file: " << path << "\n";
      continue;
    }
    bids.clear();
    asks.clear();
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }
      uint64_t ts = parseTimestamp(line);
      std::vector<std::pair<double, double>> bid_updates;
      std::vector<std::pair<double, double>> ask_updates;

      auto bid_array = extractArray(line, "b");
      if (!bid_array.empty()) {
        parseLevels(bid_array, bid_updates);
      }
      auto ask_array = extractArray(line, "a");
      if (!ask_array.empty()) {
        parseLevels(ask_array, ask_updates);
      }

      if (isSnapshot(line)) {
        bids.clear();
        asks.clear();
      }

      for (const auto& [price, qty] : bid_updates) {
        int price_int = static_cast<int>(std::lround(price * config_.price_scale));
        int qty_int = static_cast<int>(std::lround(qty * config_.qty_scale));
        if (qty_int <= 0) {
          bids.erase(price_int);
        } else {
          bids[price_int] = qty_int;
        }
      }
      for (const auto& [price, qty] : ask_updates) {
        int price_int = static_cast<int>(std::lround(price * config_.price_scale));
        int qty_int = static_cast<int>(std::lround(qty * config_.qty_scale));
        if (qty_int <= 0) {
          asks.erase(price_int);
        } else {
          asks[price_int] = qty_int;
        }
      }

      event_index++;
      if (config_.replay_skip > 1 && (event_index % config_.replay_skip) != 0) {
        continue;
      }

      if (config_.replay_max_steps > 0 &&
          processed >= static_cast<uint64_t>(config_.replay_max_steps)) {
        break;
      }

      expireBookOrders(order_log, ts);
      book_.clear();
      for (const auto& [price, qty] : bids) {
        auto order = std::make_shared<Order>();
        order->id = nextId();
        order->owner = Owner::External;
        order->side = Side::Buy;
        order->priority_front = true;
        order->time = ts;
        order->price = clampPrice(price);
        order->qty = std::max(1, qty);
        order->remaining = order->qty;
        book_.addOrder(order);
      }
      for (const auto& [price, qty] : asks) {
        auto order = std::make_shared<Order>();
        order->id = nextId();
        order->owner = Owner::External;
        order->side = Side::Sell;
        order->priority_front = true;
        order->time = ts;
        order->price = clampPrice(price);
        order->qty = std::max(1, qty);
        order->remaining = order->qty;
        book_.addOrder(order);
      }

      int mid = midPrice();
      if (mid == 0) {
        mid = last_mid;
      }
      last_mid = mid;
      metrics.updateAdverseSelection(ts, mid);
      submitExecution(ts, mid);

      if (trade_feed) {
        while (trade_feed->hasNext() && trade_feed->peek().ts <= ts) {
          const auto& tr = trade_feed->peek();
          int qty_int = static_cast<int>(std::lround(tr.qty * config_.qty_scale));
          if (qty_int > 0) {
            auto trade_order = std::make_shared<Order>();
            trade_order->id = nextId();
            trade_order->owner = Owner::External;
            trade_order->time = tr.ts;
            trade_order->qty = qty_int;
            trade_order->remaining = trade_order->qty;
            trade_order->is_market = true;
            trade_order->side = tr.side;
            trade_order->price = mid;
            auto trade_trades = engine_.process(trade_order);
            logOrder(order_log, *trade_order, "trade_feed");
            handleTrades(trade_trades, mid);
          }
          trade_feed->advance();
        }
      }

      logBook(book_log, ts, 0, 0);
      processed++;
    }

    if (config_.replay_max_steps > 0 &&
        processed >= static_cast<uint64_t>(config_.replay_max_steps)) {
      break;
    }
  }

  SimResult result;
  result.metrics = metrics;
  result.trades = trades;
  result.last_mid = last_mid;
  result.exec_target_qty = exec_target_qty;
  result.exec_filled_qty = exec_filled_qty;
  result.exec_submitted_qty = exec_submitted_qty;
  result.exec_forced_ioc = exec_forced_ioc;
  result.exec_ml_ioc = exec_ml_ioc;
  result.exec_skip_count = exec_skip_count;
  result.exec_max_no_fill = exec_max_no_fill;
  result.exec_action_counts = std::move(exec_action_counts);
  return result;
}

int Simulator::midPrice() const {
  auto bid = book_.bestBid();
  auto ask = book_.bestAsk();
  if (bid && ask) {
    return (*bid + *ask) / 2;
  }
  if (bid) {
    return *bid;
  }
  if (ask) {
    return *ask;
  }
  return 0;
}

int Simulator::spread() const {
  auto bid = book_.bestBid();
  auto ask = book_.bestAsk();
  if (!bid || !ask) {
    return 0;
  }
  return *ask - *bid;
}

uint64_t Simulator::dayExpiry(uint64_t time) const {
  uint64_t day_ms = std::max<uint64_t>(1, config_.day_duration_ms);
  uint64_t anchor = config_.day_anchor_ms;
  if (time < anchor) {
    return anchor + day_ms;
  }
  uint64_t elapsed = time - anchor;
  uint64_t periods = elapsed / day_ms;
  return anchor + (periods + 1) * day_ms;
}

void Simulator::expireBookOrders(std::ofstream& order_log, uint64_t now) {
  std::vector<OrderPtr> expired;
  book_.expireOrders(now, &expired);
  for (const auto& order : expired) {
    if (!order) {
      continue;
    }
    order->time = now;
    logOrder(order_log, *order, "expire");
  }
}

void Simulator::logBook(std::ofstream& out, uint64_t time, uint64_t mm_bid_id, uint64_t mm_ask_id) {
  int bid = book_.bestBid().value_or(0);
  int ask = book_.bestAsk().value_or(0);
  int mid = midPrice();
  int spr = spread();
  int bid_depth = book_.depth(Side::Buy, 5);
  int ask_depth = book_.depth(Side::Sell, 5);
  int mm_bid_queue = mm_bid_id ? book_.queuePosition(mm_bid_id) : -1;
  int mm_ask_queue = mm_ask_id ? book_.queuePosition(mm_ask_id) : -1;
  double imbalance = 0.0;
  if (bid_depth + ask_depth > 0) {
    imbalance = static_cast<double>(bid_depth - ask_depth) /
                static_cast<double>(bid_depth + ask_depth);
  }
  double micro = book_.microPrice(5);

  out << time << ',' << bid << ',' << ask << ',' << mid << ',' << spr << ','
      << micro << ',' << bid_depth << ',' << ask_depth << ',' << imbalance << ','
      << mm_bid_queue << ',' << mm_ask_queue << '\n';
}

void Simulator::logOrder(std::ofstream& out, const Order& order, const std::string& action) {
  out << order.time << ',' << order.id << ',' << sideToString(order.side) << ','
      << order.price << ',' << order.qty << ',' << order.remaining << ','
      << ownerToString(order.owner) << ',' << tifToString(order.tif) << ','
      << (order.post_only ? 1 : 0) << ',' << action << '\n';
}

void Simulator::logTrade(std::ofstream& out,
                         const Trade& trade,
                         int mid,
                         int inventory) {
  out << trade.time << ',' << trade.price << ',' << trade.qty << ','
      << trade.maker_id << ',' << trade.taker_id << ','
      << (trade.maker_is_mm ? 1 : 0) << ',' << (trade.taker_is_mm ? 1 : 0)
      << ',' << sideToString(trade.maker_side) << ',' << mid << ','
      << inventory << '\n';
}

SimConfig loadConfig(const std::string& path) {
  SimConfig config;
  std::ifstream file(path);
  if (!file) {
    return config;
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, pos);
    std::string value = line.substr(pos + 1);
    values[key] = value;
  }

  auto getInt = [&](const std::string& key, int& out) {
    auto it = values.find(key);
    if (it != values.end()) {
      out = std::stoi(it->second);
    }
  };
  auto getDouble = [&](const std::string& key, double& out) {
    auto it = values.find(key);
    if (it != values.end()) {
      out = std::stod(it->second);
    }
  };
  auto getString = [&](const std::string& key, std::string& out) {
    auto it = values.find(key);
    if (it != values.end()) {
      out = it->second;
    }
  };
  auto getU64 = [&](const std::string& key, uint64_t& out) {
    auto it = values.find(key);
    if (it != values.end()) {
      out = static_cast<uint64_t>(std::stoull(it->second));
    }
  };

  getInt("initial_mid", config.initial_mid);
  getInt("max_inventory", config.max_inventory);
  getDouble("fee_bps", config.fee_bps);
  getDouble("rebate_bps", config.rebate_bps);
  getInt("price_scale", config.price_scale);
  getInt("qty_scale", config.qty_scale);
  getInt("replay_skip", config.replay_skip);
  getInt("replay_max_steps", config.replay_max_steps);
  getString("output_dir", config.output_dir);
  getInt("use_trade_feed", config.use_trade_feed);
  getString("trade_path", config.trade_path);
  getString("replay_path", config.replay_path);
  getString("replay_type", config.replay_type);
  getU64("adverse_horizon_ms", config.adverse_horizon_ms);
  {
    std::string horizons_csv;
    getString("adverse_horizons_ms", horizons_csv);
    if (!horizons_csv.empty()) {
      std::stringstream ss(horizons_csv);
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (token.empty()) {
          continue;
        }
        try {
          config.adverse_horizons_ms.push_back(
              static_cast<uint64_t>(std::stoull(token)));
        } catch (...) {
        }
      }
    }
  }
  getString("exec_mode", config.exec_mode);
  getString("exec_side", config.exec_side);
  getInt("exec_qty", config.exec_qty);
  getInt("exec_limit_offset_ticks", config.exec_limit_offset_ticks);
  getU64("exec_start_ms", config.exec_start_ms);
  getU64("exec_end_ms", config.exec_end_ms);
  getInt("exec_slice_ms", config.exec_slice_ms);
  getString("exec_tif", config.exec_tif);
  getInt("exec_post_only", config.exec_post_only);
  getInt("exec_is_market", config.exec_is_market);
  getString("exec_price_mode", config.exec_price_mode);
  getString("exec_policy", config.exec_policy);
  getInt("exec_min_slice_qty", config.exec_min_slice_qty);
  getInt("exec_max_slice_qty", config.exec_max_slice_qty);
  getDouble("exec_behind_threshold", config.exec_behind_threshold);
  getDouble("exec_ahead_threshold", config.exec_ahead_threshold);
  getDouble("exec_panic_threshold", config.exec_panic_threshold);
  getInt("exec_spread_max_ticks", config.exec_spread_max_ticks);
  getDouble("exec_imbalance_max", config.exec_imbalance_max);
  getDouble("exec_vol_max", config.exec_vol_max);
  getInt("exec_passive_offset_ticks", config.exec_passive_offset_ticks);
  getInt("exec_skip_when_toxic", config.exec_skip_when_toxic);
  getInt("exec_max_passive_slices", config.exec_max_passive_slices);
  getInt("exec_final_sweep_slices", config.exec_final_sweep_slices);
  getU64("exec_final_sweep_ms", config.exec_final_sweep_ms);
  getInt("exec_ml_enabled", config.exec_ml_enabled);
  getU64("exec_ml_horizon_ms", config.exec_ml_horizon_ms);
  getDouble("exec_ml_learning_rate", config.exec_ml_learning_rate);
  getDouble("exec_ml_l2", config.exec_ml_l2);
  getInt("exec_ml_warmup_samples", config.exec_ml_warmup_samples);
  getDouble("exec_ml_aggression_threshold", config.exec_ml_aggression_threshold);
  getU64("day_duration_ms", config.day_duration_ms);
  getU64("day_anchor_ms", config.day_anchor_ms);
  getU64("exec_gtd_expire_ms", config.exec_gtd_expire_ms);

  return config;
}

}  // namespace lob
