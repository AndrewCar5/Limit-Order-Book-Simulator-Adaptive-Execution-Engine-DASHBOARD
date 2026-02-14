#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "engine/order.h"

namespace lob {

using OrderPtr = std::shared_ptr<Order>;

class OrderBook {
 public:
  struct LevelSnapshot {
    int price = 0;
    int qty = 0;
    int count = 0;
  };

  bool addOrder(const OrderPtr& order);
  bool cancelOrder(uint64_t id);
  size_t cancelOrders(const std::vector<uint64_t>& ids,
                      std::vector<OrderPtr>* canceled_orders = nullptr);
  bool modifyOrder(const OrderPtr& order);
  bool removeOrder(uint64_t id);
  size_t expireOrders(uint64_t now, std::vector<OrderPtr>* expired_orders = nullptr);
  void applyFill(const OrderPtr& order, int fill_qty);
  void clear();

  std::optional<int> bestBid() const;
  std::optional<int> bestAsk() const;

  int depth(Side side, int levels) const;
  double microPrice(int levels) const;
  int queuePosition(uint64_t id) const;
  int availableLiquidity(Side side, int limit_price) const;
  std::vector<LevelSnapshot> levels(Side side, int max_levels) const;

  OrderPtr find(uint64_t id) const;

  std::list<OrderPtr>* queueAt(Side side, int price);

  std::vector<uint64_t> externalOrderIds() const;

 private:
  struct OrderEntry {
    OrderPtr order;
    std::list<OrderPtr>::iterator location;
  };

  struct LevelData {
    int qty = 0;
    int count = 0;
  };

  using LevelMap = std::unordered_map<int, LevelData>;

  using BidMap = std::map<int, std::list<OrderPtr>, std::greater<int>>;
  using AskMap = std::map<int, std::list<OrderPtr>, std::less<int>>;

  BidMap bids_;
  AskMap asks_;
  std::unordered_map<uint64_t, OrderEntry> orders_;
  LevelMap bid_levels_;
  LevelMap ask_levels_;

  void pruneLevel(Side side, int price);
  void updateLevel(Side side, int price, int qty_delta, int count_delta);
};

}  // namespace lob
