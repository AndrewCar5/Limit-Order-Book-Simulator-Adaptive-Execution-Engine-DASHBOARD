#include "engine/matching_engine.h"

#include <limits>

namespace lob {

MatchingEngine::MatchingEngine(OrderBook& book) : book_(book) {}

bool MatchingEngine::isCrossing(const OrderPtr& order) const {
  if (order->is_market) {
    return true;
  }
  if (order->side == Side::Buy) {
    auto best_ask = book_.bestAsk();
    return best_ask && order->price >= *best_ask;
  }
  auto best_bid = book_.bestBid();
  return best_bid && order->price <= *best_bid;
}

bool MatchingEngine::canFillAll(const OrderPtr& order) const {
  if (!order || order->remaining <= 0) {
    return false;
  }
  int limit_price = order->price;
  if (order->is_market) {
    limit_price = (order->side == Side::Buy) ? std::numeric_limits<int>::max()
                                             : std::numeric_limits<int>::min();
  }
  int available = book_.availableLiquidity(order->side, limit_price);
  return available >= order->remaining;
}

std::vector<Trade> MatchingEngine::process(const OrderPtr& order) {
  std::vector<Trade> trades;
  if (!order || order->remaining <= 0) {
    return trades;
  }
  if ((order->tif == TimeInForce::DAY || order->tif == TimeInForce::GTD) &&
      order->expire_time > 0 &&
      order->expire_time <= order->time) {
    order->status = OrderStatus::Canceled;
    return trades;
  }

  if (order->post_only) {
    if (order->is_market || isCrossing(order)) {
      order->status = OrderStatus::Canceled;
      return trades;
    }
  }

  if (order->tif == TimeInForce::FOK && !canFillAll(order)) {
    order->status = OrderStatus::Canceled;
    return trades;
  }

  bool any_fill = false;
  while (order->remaining > 0 && isCrossing(order)) {
    Side opp_side = order->side == Side::Buy ? Side::Sell : Side::Buy;
    auto best_price = (opp_side == Side::Buy) ? book_.bestBid() : book_.bestAsk();
    if (!best_price) {
      break;
    }

    auto* queue = book_.queueAt(opp_side, *best_price);
    if (!queue || queue->empty()) {
      break;
    }

    auto maker = queue->front();
    int fill_qty = std::min(order->remaining, maker->remaining);
    int trade_price = maker->price;

    order->remaining -= fill_qty;
    book_.applyFill(maker, fill_qty);
    any_fill = true;

    Trade trade;
    trade.time = order->time;
    trade.price = trade_price;
    trade.qty = fill_qty;
    trade.maker_id = maker->id;
    trade.taker_id = order->id;
    trade.maker_side = maker->side;
    trade.maker_is_mm = maker->owner == Owner::MarketMaker;
    trade.taker_is_mm = order->owner == Owner::MarketMaker;
    trades.push_back(trade);

    if (maker->remaining == 0) {
      maker->status = OrderStatus::Filled;
    } else {
      maker->status = OrderStatus::Partial;
    }
  }

  if (order->remaining > 0 && order->tif == TimeInForce::IOC) {
    order->status = any_fill ? OrderStatus::Partial : OrderStatus::Canceled;
  } else if (order->remaining > 0 && order->is_market) {
    order->status = any_fill ? OrderStatus::Partial : OrderStatus::Canceled;
  } else if (order->remaining > 0 && !order->is_market) {
    if (!book_.addOrder(order)) {
      order->status = OrderStatus::Canceled;
    }
  } else if (order->remaining == 0) {
    order->status = OrderStatus::Filled;
  }

  return trades;
}

}  // namespace lob
