#include <algorithm>
#include <cassert>
#include <memory>

#include "engine/matching_engine.h"

namespace {

lob::OrderPtr makeOrder(uint64_t id,
                        lob::Side side,
                        int price,
                        int qty,
                        bool is_market = false) {
  auto order = std::make_shared<lob::Order>();
  order->id = id;
  order->side = side;
  order->owner = lob::Owner::External;
  order->price = price;
  order->qty = qty;
  order->remaining = qty;
  order->is_market = is_market;
  order->time = id;
  return order;
}

}  // namespace

int main() {
  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid1 = makeOrder(1, lob::Side::Buy, 100, 5);
    auto bid2 = makeOrder(2, lob::Side::Buy, 100, 5);
    engine.process(bid1);
    engine.process(bid2);

    auto sell = makeOrder(3, lob::Side::Sell, 100, 6);
    auto trades = engine.process(sell);
    assert(trades.size() == 2);
    assert(trades[0].maker_id == 1);
    assert(trades[1].maker_id == 2);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid1 = makeOrder(1, lob::Side::Buy, 101, 5);
    auto bid2 = makeOrder(2, lob::Side::Buy, 100, 5);
    engine.process(bid1);
    engine.process(bid2);

    auto sell = makeOrder(3, lob::Side::Sell, 0, 8, true);
    auto trades = engine.process(sell);
    assert(trades.size() == 2);
    assert(trades[0].qty == 5);
    assert(trades[1].qty == 3);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid = makeOrder(1, lob::Side::Buy, 100, 5);
    engine.process(bid);

    auto sell_ioc = makeOrder(2, lob::Side::Sell, 100, 10);
    sell_ioc->tif = lob::TimeInForce::IOC;
    auto trades = engine.process(sell_ioc);
    assert(trades.size() == 1);
    assert(trades[0].qty == 5);
    assert(sell_ioc->status == lob::OrderStatus::Partial);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto sell_ioc = makeOrder(1, lob::Side::Sell, 100, 5);
    sell_ioc->tif = lob::TimeInForce::IOC;
    auto trades = engine.process(sell_ioc);
    assert(trades.empty());
    assert(sell_ioc->status == lob::OrderStatus::Canceled);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid = makeOrder(1, lob::Side::Buy, 100, 10);
    engine.process(bid);

    auto sell_fok = makeOrder(2, lob::Side::Sell, 100, 10);
    sell_fok->tif = lob::TimeInForce::FOK;
    auto trades = engine.process(sell_fok);
    assert(trades.size() == 1);
    assert(trades[0].qty == 10);
    assert(sell_fok->status == lob::OrderStatus::Filled);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid = makeOrder(1, lob::Side::Buy, 100, 5);
    engine.process(bid);

    auto sell_fok = makeOrder(2, lob::Side::Sell, 100, 10);
    sell_fok->tif = lob::TimeInForce::FOK;
    auto trades = engine.process(sell_fok);
    assert(trades.empty());
    assert(sell_fok->status == lob::OrderStatus::Canceled);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto ask = makeOrder(1, lob::Side::Sell, 100, 5);
    engine.process(ask);

    auto buy_post = makeOrder(2, lob::Side::Buy, 100, 5);
    buy_post->post_only = true;
    auto trades = engine.process(buy_post);
    assert(trades.empty());
    assert(buy_post->status == lob::OrderStatus::Canceled);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto ask1 = makeOrder(1, lob::Side::Sell, 100, 5);
    auto ask2 = makeOrder(2, lob::Side::Sell, 101, 5);
    engine.process(ask1);
    engine.process(ask2);

    auto buy = makeOrder(3, lob::Side::Buy, 0, 8, true);
    auto trades = engine.process(buy);
    assert(trades.size() == 2);
    assert(trades[0].qty == 5);
    assert(trades[1].qty == 3);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid1 = makeOrder(1, lob::Side::Buy, 100, 5);
    auto bid2 = makeOrder(2, lob::Side::Buy, 100, 5);
    engine.process(bid1);
    engine.process(bid2);

    book.cancelOrder(1);
    assert(book.bestBid().value_or(0) == 100);
    assert(book.queuePosition(2) == 0);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid = makeOrder(1, lob::Side::Buy, 100, 5);
    engine.process(bid);
    book.cancelOrder(1);
    assert(!book.bestBid().has_value());
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto ask = makeOrder(1, lob::Side::Sell, 101, 10);
    engine.process(ask);

    auto buy = makeOrder(2, lob::Side::Buy, 0, 3, true);
    auto trades = engine.process(buy);
    assert(trades.size() == 1);
    assert(trades[0].qty == 3);
    assert(book.queuePosition(1) == 0);

    bool cancelled = book.cancelOrder(1);
    assert(cancelled);
    assert(!book.bestAsk().has_value());
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto ask = makeOrder(1, lob::Side::Sell, 101, 5);
    engine.process(ask);

    auto buy = makeOrder(2, lob::Side::Buy, 0, 5, true);
    auto trades = engine.process(buy);
    assert(trades.size() == 1);
    assert(!book.bestAsk().has_value());
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    engine.process(makeOrder(1, lob::Side::Sell, 101, 3));
    engine.process(makeOrder(2, lob::Side::Sell, 102, 2));

    auto fok_too_big = makeOrder(3, lob::Side::Buy, 102, 6);
    fok_too_big->tif = lob::TimeInForce::FOK;
    auto trades = engine.process(fok_too_big);
    assert(trades.empty());
    assert(fok_too_big->status == lob::OrderStatus::Canceled);

    auto fok_ok = makeOrder(4, lob::Side::Buy, 102, 5);
    fok_ok->tif = lob::TimeInForce::FOK;
    trades = engine.process(fok_ok);
    assert(trades.size() == 2);
    assert(fok_ok->status == lob::OrderStatus::Filled);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto ask = makeOrder(1, lob::Side::Sell, 100, 5);
    engine.process(ask);

    auto buy = makeOrder(2, lob::Side::Buy, 101, 10);
    auto trades = engine.process(buy);
    assert(trades.size() == 1);
    assert(trades[0].qty == 5);
    assert(buy->remaining == 5);
    assert(buy->status == lob::OrderStatus::Partial);

    auto resting = book.find(2);
    assert(resting);
    assert(resting->remaining == 5);

    auto sell_mkt = makeOrder(3, lob::Side::Sell, 0, 5, true);
    auto trades2 = engine.process(sell_mkt);
    assert(trades2.size() == 1);
    assert(trades2[0].maker_id == 2);
    assert(trades2[0].qty == 5);
    assert(!book.bestBid().has_value());
  }

  {
    lob::OrderBook book;

    auto bid = makeOrder(1, lob::Side::Buy, 100, 3);
    auto ask = makeOrder(2, lob::Side::Sell, 100, 7);
    assert(book.addOrder(bid));
    assert(book.addOrder(ask));

    assert(book.depth(lob::Side::Buy, 1) == 3);
    assert(book.depth(lob::Side::Sell, 1) == 7);
    assert(book.availableLiquidity(lob::Side::Buy, 100) == 7);
    assert(book.availableLiquidity(lob::Side::Sell, 100) == 3);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto bid = makeOrder(1, lob::Side::Buy, 100, 10);
    engine.process(bid);
    auto sell_mkt = makeOrder(2, lob::Side::Sell, 0, 3, true);
    engine.process(sell_mkt);

    auto ids = book.externalOrderIds();
    auto it = std::find(ids.begin(), ids.end(), static_cast<uint64_t>(1));
    assert(it != ids.end());
  }

  {
    lob::OrderBook book;

    auto bid = makeOrder(1, lob::Side::Buy, 100, 5);
    assert(book.addOrder(bid));

    auto modified = makeOrder(1, lob::Side::Buy, 99, 8);
    assert(book.modifyOrder(modified));
    assert(book.bestBid().value_or(0) == 99);
    auto resting = book.find(1);
    assert(resting);
    assert(resting->price == 99);
    assert(resting->remaining == 8);
    assert(book.depth(lob::Side::Buy, 1) == 8);
  }

  {
    lob::OrderBook book;
    assert(book.addOrder(makeOrder(1, lob::Side::Buy, 100, 5)));
    assert(book.addOrder(makeOrder(2, lob::Side::Buy, 99, 4)));
    assert(book.addOrder(makeOrder(3, lob::Side::Sell, 101, 6)));

    std::vector<uint64_t> ids{1, 3, 777};
    std::vector<lob::OrderPtr> canceled;
    auto canceled_count = book.cancelOrders(ids, &canceled);
    assert(canceled_count == 2);
    assert(canceled.size() == 2);
    assert(canceled[0]->status == lob::OrderStatus::Canceled);
    assert(canceled[1]->status == lob::OrderStatus::Canceled);
    assert(book.find(1) == nullptr);
    assert(book.find(3) == nullptr);
    assert(book.bestBid().value_or(0) == 99);
    assert(!book.bestAsk().has_value());
  }

  {
    lob::OrderBook book;
    assert(book.addOrder(makeOrder(1, lob::Side::Buy, 101, 2)));
    assert(book.addOrder(makeOrder(2, lob::Side::Buy, 101, 3)));
    assert(book.addOrder(makeOrder(3, lob::Side::Buy, 100, 4)));
    assert(book.addOrder(makeOrder(4, lob::Side::Sell, 102, 5)));

    auto bids = book.levels(lob::Side::Buy, 2);
    assert(bids.size() == 2);
    assert(bids[0].price == 101);
    assert(bids[0].qty == 5);
    assert(bids[0].count == 2);
    assert(bids[1].price == 100);
    assert(bids[1].qty == 4);
    assert(bids[1].count == 1);

    auto asks = book.levels(lob::Side::Sell, 1);
    assert(asks.size() == 1);
    assert(asks[0].price == 102);
    assert(asks[0].qty == 5);
    assert(asks[0].count == 1);
  }

  {
    lob::OrderBook book;
    auto day = makeOrder(1, lob::Side::Buy, 100, 3);
    day->tif = lob::TimeInForce::DAY;
    day->expire_time = 10;
    assert(book.addOrder(day));
    auto gtd = makeOrder(2, lob::Side::Sell, 101, 4);
    gtd->tif = lob::TimeInForce::GTD;
    gtd->expire_time = 20;
    assert(book.addOrder(gtd));

    std::vector<lob::OrderPtr> expired;
    auto expired_count = book.expireOrders(15, &expired);
    assert(expired_count == 1);
    assert(expired.size() == 1);
    assert(expired[0]->id == 1);
    assert(expired[0]->status == lob::OrderStatus::Canceled);
    assert(book.find(1) == nullptr);
    assert(book.find(2) != nullptr);

    expired_count = book.expireOrders(25, &expired);
    assert(expired_count == 1);
    assert(expired.size() == 1);
    assert(expired[0]->id == 2);
    assert(book.find(2) == nullptr);
  }

  {
    lob::OrderBook book;
    lob::MatchingEngine engine(book);

    auto expired = makeOrder(1, lob::Side::Buy, 100, 2);
    expired->tif = lob::TimeInForce::GTD;
    expired->time = 50;
    expired->expire_time = 50;
    auto trades = engine.process(expired);
    assert(trades.empty());
    assert(expired->status == lob::OrderStatus::Canceled);
    assert(book.find(1) == nullptr);
  }

  return 0;
}
