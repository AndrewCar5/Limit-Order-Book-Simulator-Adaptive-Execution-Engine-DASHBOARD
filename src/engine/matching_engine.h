#pragma once

#include <vector>

#include "engine/order.h"
#include "engine/order_book.h"

namespace lob {

class MatchingEngine {
 public:
  explicit MatchingEngine(OrderBook& book);

  std::vector<Trade> process(const OrderPtr& order);

 private:
  OrderBook& book_;

  bool isCrossing(const OrderPtr& order) const;
  bool canFillAll(const OrderPtr& order) const;
};

}  // namespace lob
