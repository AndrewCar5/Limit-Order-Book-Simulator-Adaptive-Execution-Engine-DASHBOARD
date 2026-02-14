#include "engine/order_book.h"

#include <algorithm>

namespace lob {

bool OrderBook::addOrder(const OrderPtr& order) {
  if (!order || order->qty <= 0) {
    return false;
  }
  if (orders_.count(order->id)) {
    return false;
  }
  if (order->remaining <= 0) {
    order->remaining = order->qty;
  } else if (order->remaining > order->qty) {
    order->remaining = order->qty;
  }
  if (order->remaining <= 0) {
    return false;
  }
  order->status = (order->remaining < order->qty) ? OrderStatus::Partial
                                                   : OrderStatus::Open;

  auto& queue = (order->side == Side::Buy) ? bids_[order->price] : asks_[order->price];
  if (order->priority_front) {
    queue.push_front(order);
    orders_[order->id] = {order, queue.begin()};
  } else {
    queue.push_back(order);
    auto it = queue.end();
    --it;
    orders_[order->id] = {order, it};
  }

  updateLevel(order->side, order->price, order->remaining, 1);
  return true;
}

bool OrderBook::cancelOrder(uint64_t id) {
  auto it = orders_.find(id);
  if (it == orders_.end()) {
    return false;
  }
  auto order = it->second.order;
  auto* queue = queueAt(order->side, order->price);
  if (!queue) {
    return false;
  }

  order->status = OrderStatus::Canceled;
  queue->erase(it->second.location);
  updateLevel(order->side, order->price, -order->remaining, -1);
  orders_.erase(it);
  pruneLevel(order->side, order->price);
  return true;
}

size_t OrderBook::cancelOrders(const std::vector<uint64_t>& ids,
                               std::vector<OrderPtr>* canceled_orders) {
  size_t canceled = 0;
  if (canceled_orders) {
    canceled_orders->clear();
    canceled_orders->reserve(ids.size());
  }
  for (uint64_t id : ids) {
    auto order = find(id);
    if (!order) {
      continue;
    }
    if (cancelOrder(id)) {
      ++canceled;
      if (canceled_orders) {
        canceled_orders->push_back(order);
      }
    }
  }
  return canceled;
}

bool OrderBook::modifyOrder(const OrderPtr& order) {
  if (!order) {
    return false;
  }
  auto it = orders_.find(order->id);
  if (it == orders_.end()) {
    return false;
  }
  if (!cancelOrder(order->id)) {
    return false;
  }
  return addOrder(order);
}

bool OrderBook::removeOrder(uint64_t id) {
  auto it = orders_.find(id);
  if (it == orders_.end()) {
    return false;
  }
  auto order = it->second.order;
  auto* queue = queueAt(order->side, order->price);
  if (!queue) {
    return false;
  }

  queue->erase(it->second.location);
  updateLevel(order->side, order->price, -order->remaining, -1);
  orders_.erase(it);
  pruneLevel(order->side, order->price);
  return true;
}

size_t OrderBook::expireOrders(uint64_t now, std::vector<OrderPtr>* expired_orders) {
  std::vector<uint64_t> ids;
  ids.reserve(orders_.size());
  for (const auto& [id, entry] : orders_) {
    if (entry.order->expire_time == 0 || entry.order->expire_time > now) {
      continue;
    }
    if (entry.order->status == OrderStatus::Canceled ||
        entry.order->status == OrderStatus::Filled) {
      continue;
    }
    ids.push_back(id);
  }
  return cancelOrders(ids, expired_orders);
}

void OrderBook::applyFill(const OrderPtr& order, int fill_qty) {
  if (!order || fill_qty <= 0) {
    return;
  }
  if (fill_qty > order->remaining) {
    fill_qty = order->remaining;
  }
  if (fill_qty <= 0) {
    return;
  }

  order->remaining -= fill_qty;
  updateLevel(order->side, order->price, -fill_qty, 0);

  if (order->remaining == 0) {
    updateLevel(order->side, order->price, 0, -1);
    auto it = orders_.find(order->id);
    if (it == orders_.end()) {
      return;
    }
    auto* queue = queueAt(order->side, order->price);
    if (queue) {
      queue->erase(it->second.location);
    }
    orders_.erase(it);
    pruneLevel(order->side, order->price);
  }
}

void OrderBook::clear() {
  bids_.clear();
  asks_.clear();
  orders_.clear();
  bid_levels_.clear();
  ask_levels_.clear();
}

std::optional<int> OrderBook::bestBid() const {
  if (bids_.empty()) {
    return std::nullopt;
  }
  return bids_.begin()->first;
}

std::optional<int> OrderBook::bestAsk() const {
  if (asks_.empty()) {
    return std::nullopt;
  }
  return asks_.begin()->first;
}

int OrderBook::depth(Side side, int levels) const {
  int total = 0;
  int count = 0;

  if (side == Side::Buy) {
    for (const auto& [price, queue] : bids_) {
      if (count++ >= levels) {
        break;
      }
      auto it = bid_levels_.find(price);
      if (it != bid_levels_.end()) {
        total += it->second.qty;
      }
    }
  } else {
    for (const auto& [price, queue] : asks_) {
      if (count++ >= levels) {
        break;
      }
      auto it = ask_levels_.find(price);
      if (it != ask_levels_.end()) {
        total += it->second.qty;
      }
    }
  }

  return total;
}

double OrderBook::microPrice(int levels) const {
  auto bid = bestBid();
  auto ask = bestAsk();
  if (!bid || !ask) {
    return 0.0;
  }

  double bid_depth = static_cast<double>(depth(Side::Buy, levels));
  double ask_depth = static_cast<double>(depth(Side::Sell, levels));
  double denom = bid_depth + ask_depth;
  if (denom <= 0.0) {
    return 0.5 * (static_cast<double>(*bid) + static_cast<double>(*ask));
  }

  return (static_cast<double>(*ask) * bid_depth +
          static_cast<double>(*bid) * ask_depth) /
         denom;
}

int OrderBook::queuePosition(uint64_t id) const {
  auto it = orders_.find(id);
  if (it == orders_.end()) {
    return -1;
  }
  const auto& order = it->second.order;
  int ahead = 0;
  if (order->side == Side::Buy) {
    auto level = bids_.find(order->price);
    if (level == bids_.end()) {
      return -1;
    }
    for (const auto& ptr : level->second) {
      if (ptr->id == id) {
        break;
      }
      ahead += ptr->remaining;
    }
  } else {
    auto level = asks_.find(order->price);
    if (level == asks_.end()) {
      return -1;
    }
    for (const auto& ptr : level->second) {
      if (ptr->id == id) {
        break;
      }
      ahead += ptr->remaining;
    }
  }
  return ahead;
}

int OrderBook::availableLiquidity(Side side, int limit_price) const {
  int total = 0;
  if (side == Side::Buy) {
    for (const auto& [price, queue] : asks_) {
      if (price > limit_price) {
        break;
      }
      auto it = ask_levels_.find(price);
      if (it != ask_levels_.end()) {
        total += it->second.qty;
      }
    }
  } else {
    for (const auto& [price, queue] : bids_) {
      if (price < limit_price) {
        break;
      }
      auto it = bid_levels_.find(price);
      if (it != bid_levels_.end()) {
        total += it->second.qty;
      }
    }
  }
  return total;
}

std::vector<OrderBook::LevelSnapshot> OrderBook::levels(Side side, int max_levels) const {
  std::vector<LevelSnapshot> out;
  if (max_levels <= 0) {
    return out;
  }
  out.reserve(static_cast<size_t>(max_levels));
  if (side == Side::Buy) {
    for (const auto& [price, queue] : bids_) {
      if (static_cast<int>(out.size()) >= max_levels) {
        break;
      }
      auto it = bid_levels_.find(price);
      if (it == bid_levels_.end()) {
        continue;
      }
      out.push_back(LevelSnapshot{price, it->second.qty, it->second.count});
    }
    return out;
  }
  for (const auto& [price, queue] : asks_) {
    if (static_cast<int>(out.size()) >= max_levels) {
      break;
    }
    auto it = ask_levels_.find(price);
    if (it == ask_levels_.end()) {
      continue;
    }
    out.push_back(LevelSnapshot{price, it->second.qty, it->second.count});
  }
  return out;
}

OrderPtr OrderBook::find(uint64_t id) const {
  auto it = orders_.find(id);
  if (it == orders_.end()) {
    return nullptr;
  }
  return it->second.order;
}

std::list<OrderPtr>* OrderBook::queueAt(Side side, int price) {
  if (side == Side::Buy) {
    auto it = bids_.find(price);
    if (it == bids_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  auto it = asks_.find(price);
  if (it == asks_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<uint64_t> OrderBook::externalOrderIds() const {
  std::vector<uint64_t> ids;
  ids.reserve(orders_.size());
  for (const auto& [id, entry] : orders_) {
    if (entry.order->owner == Owner::External &&
        entry.order->remaining > 0 &&
        entry.order->status != OrderStatus::Canceled &&
        entry.order->status != OrderStatus::Filled) {
      ids.push_back(id);
    }
  }
  return ids;
}

void OrderBook::pruneLevel(Side side, int price) {
  if (side == Side::Buy) {
    auto it = bids_.find(price);
    if (it != bids_.end() && it->second.empty()) {
      bids_.erase(it);
    }
    return;
  }

  auto it = asks_.find(price);
  if (it != asks_.end() && it->second.empty()) {
    asks_.erase(it);
  }
}

void OrderBook::updateLevel(Side side, int price, int qty_delta, int count_delta) {
  auto& levels = (side == Side::Buy) ? bid_levels_ : ask_levels_;
  auto& level = levels[price];
  level.qty += qty_delta;
  level.count += count_delta;
  if (level.count <= 0 || level.qty <= 0) {
    levels.erase(price);
  }
}

}  // namespace lob
