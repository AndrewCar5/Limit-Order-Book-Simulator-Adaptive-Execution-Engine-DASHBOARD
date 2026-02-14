#pragma once

#include <cstdint>
#include <string>

namespace lob {

enum class Side { Buy, Sell };

enum class Owner { External, MarketMaker };

enum class OrderStatus { Open, Partial, Filled, Canceled };

enum class TimeInForce { GTC, IOC, FOK, DAY, GTD };

struct Order {
  uint64_t id = 0;
  Side side = Side::Buy;
  Owner owner = Owner::External;
  int price = 0;
  int qty = 0;
  int remaining = 0;
  bool is_market = false;
  bool post_only = false;
  TimeInForce tif = TimeInForce::GTC;
  uint64_t expire_time = 0;
  bool priority_front = false;
  uint64_t time = 0;
  OrderStatus status = OrderStatus::Open;
};

struct Trade {
  uint64_t time = 0;
  int price = 0;
  int qty = 0;
  uint64_t maker_id = 0;
  uint64_t taker_id = 0;
  Side maker_side = Side::Buy;
  bool maker_is_mm = false;
  bool taker_is_mm = false;
};

inline std::string sideToString(Side side) {
  return side == Side::Buy ? "BUY" : "SELL";
}

inline std::string ownerToString(Owner owner) {
  return owner == Owner::MarketMaker ? "MM" : "EXT";
}

inline std::string tifToString(TimeInForce tif) {
  switch (tif) {
    case TimeInForce::IOC:
      return "IOC";
    case TimeInForce::FOK:
      return "FOK";
    case TimeInForce::DAY:
      return "DAY";
    case TimeInForce::GTD:
      return "GTD";
    default:
      return "GTC";
  }
}

}  // namespace lob
