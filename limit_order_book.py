import pandas as pd
import numpy as np
from collections import deque
from orderbook import OrderBook
import datetime as dt


class MarketSim(object):
    def __init__(self, order_arrival_lambda=2, initial_price=100., price_sigma=.2, quantity_mu=2, quantity_sigma=1,
                 periods=None, traders=50, tick_size=.0001, purge_after_vol=100,
                 open_time=dt.datetime(2017, 1, 1, 8, 0, 0), close_time=dt.datetime(2017, 1, 1, 15, 0, 0)):
        self.p0 = initial_price
        self.p = initial_price
        self.tick_size = tick_size
        self.purge_after_vol = purge_after_vol
        self.order_arrival_lambda = order_arrival_lambda
        self.price_sigma = price_sigma
        self.quantity_mu = quantity_mu
        self.quantity_sigma = quantity_sigma
        if periods:
            self.periods = periods
        else:
            self.periods = (close_time-open_time).total_seconds()
        if type(traders) == int:
            traders = range(traders)
        self.traders = traders

        self.orderbook = OrderBook()
        self.total_orders_submitted = 0
        self.sides = np.array(['ask', 'bid'])
        self.sign = np.array([1., -1.])
        self.time = 0
        self.orders_by_vol = deque(maxlen=None)

    def simulate_batch(self, size=10000, verbose=False):
        # number of orders that arrive
        inter_arrival_times = np.random.exponential(self.order_arrival_lambda, size=size)
        timestamps = inter_arrival_times.cumsum() + self.time
        self.process_orders(timestamps, verbose)
        self.purge_stale_orders()
        self.time = timestamps[-1]

    def simulate_periods(self, periods=None, verbose=False):
        if not periods:
            periods = self.periods
        while self.time < periods:
            self.simulate_batch(size=min(10000, periods/10), verbose=verbose)

    def process_orders(self, timestamps, verbose=False):
        # if new orders arrived--add them to the book
        n_orders = len(timestamps)
        # draw side (buy/sell) distribution is binomial, p=.5
        order_side = np.random.choice([0, 1], size=n_orders, replace=True)
        # randomly assign traders to each slot
        order_trader = np.random.choice(self.traders, size=n_orders, replace=True)
        order_price_shocks = np.random.lognormal(0, self.price_sigma, size=n_orders)
        order_quantity = np.ceil(np.random.lognormal(self.quantity_mu, self.quantity_sigma, size=n_orders))

        for i in xrange(n_orders):
            if timestamps[i] > self.periods:
                break
            side = self.sides[order_side][i]
            order_price = self.p * order_price_shocks[i] + self.sign[order_side][i] * self.tick_size * self.p
            quote = {
               "type": "limit",
               "side": side,
               "timestamp": timestamps[i],
               "trade_id": order_trader[i],
               "order_id": self.total_orders_submitted,
               "price": order_price,
               'quantity': int(order_quantity[i])
            }
            self.orderbook.process_order(quote, from_data=True, verbose=verbose)
            self.log_order_type(side)
            self.total_orders_submitted += 1
            self.sync_book_price()

    def purge_stale_orders(self):
        if len(self.orders_by_vol) > 0 & len(self.orderbook.tape) >= 100:
            current_vol = len(self.orderbook.tape)
            while current_vol-self.purge_after_vol >= self.orders_by_vol[0][0]:
                _, side, order_id = self.orders_by_vol.popleft()
                self.orderbook.cancel_order(side=side, order_id=order_id)
                if len(self.orders_by_vol) == 0:
                    break

    def sync_book_price(self):
        if len(self.orderbook.tape) > 0:
            self.p = float(self.orderbook.tape[-1]['price'])

    def plot_prices(self, time_scale=1.,plot_params={}):
        interval_trades = self.get_transactions(time_scale=time_scale)
        interval_trades.set_index('time')['price'].apply(lambda x: float(x)).plot(**plot_params)

    def get_tape_dataframe(self):
        return pd.DataFrame(list(self.orderbook.tape))

    def show_stats(self):
        n_trades = len(self.orderbook.tape)
        mean_quantity = self.get_tape_dataframe()['quantity'].apply(lambda x: float(x)).mean()
        print("trades per second: %3.3f" % (n_trades / self.time))
        print("quantity per 600: %3.3f" % (mean_quantity * 600))

    def log_order_type(self, side):
        order_id = self.total_orders_submitted
        current_vol = len(self.orderbook.tape)
        self.orders_by_vol.append((current_vol, str(side), int(order_id)))

    @staticmethod
    def get_side_id(side, party1, party2):
        if party1[1] == side:
            return party1[0]
        else:
            return party2[0]

    def get_transactions(self, time_scale=1):
        tape = self.get_tape_dataframe()
        tape['price'] = tape['price'].apply(float)
        tape['quantity'] = tape['quantity'].apply(int)
        tape['time'] = tape['timestamp'].apply(lambda x: np.floor(x/time_scale))
        tape['quantity'] = (tape.groupby('time')['quantity'].transform('sum'))
        agg = tape.groupby('time').last().reset_index()
        agg['ask_id'] = agg.apply(lambda row: MarketSim.get_side_id('ask', row['party1'], row['party2']), axis=1)
        agg['bid_id'] = agg.apply(lambda row: MarketSim.get_side_id('bid', row['party1'], row['party2']), axis=1)
        return agg[['time', 'ask_id', 'bid_id', 'quantity', 'price']]

    def export_transactions(self, filename):
        self.get_transactions().to_csv(filename, index=False)


if __name__=="__main__":
    # target volume is 600/23.3 per second
    # assume approximately 50% of the trades cross

    arrival_lambda = 1./((600/23.3)*(1./.50))

    params = {'order_arrival_lambda': arrival_lambda,
              'price_sigma': .005,
              'tick_size': .0025,
              'quantity_mu': 1,
              'quantity_sigma': 1.1
              }
    sim = MarketSim(**params)
    test = False

    if test:
        sim.simulate_batch(3e4, verbose=True)
    else:
        sim.simulate_periods(verbose=True)
        sim.export_transactions("~/Projects/MarketSim/day_sample.csv")






