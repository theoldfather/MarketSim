import pandas as pd
import numpy as np
from collections import deque
from orderbook import OrderBook


# limit order book

# simulator
# given


class Simulator(object):
    def __init__(self, order_arrival_lambda=2, initial_price=100., price_sigma=.2, quantity_sigma=1, periods=60*60*7,
                 traders=50, tick_size=.0001, draw_price_direction=True, purge_after_vol=100):
        self.p0 = initial_price
        self.p = initial_price
        self.tick_size = tick_size
        self.purge_after_vol = purge_after_vol
        self.order_arrival_lambda = order_arrival_lambda
        self.price_sigma = price_sigma
        self.quantity_sigma = quantity_sigma
        self.periods = periods
        self.draw_price_direction = draw_price_direction
        if type(traders) == int:
            traders = range(traders)
        self.traders = traders

        self.orderbook = OrderBook()
        self.total_orders_submitted = 0
        self.sides = np.array(['ask', 'bid'])
        self.sign = np.array([1., -1.])
        self.time = 0
        self.orders_by_vol = deque(maxlen=None)

    def simulate_batch(self, size=100):
        # number of orders that arrive
        inter_arrival_times = np.random.exponential(self.order_arrival_lambda, size=size)
        timestamps = inter_arrival_times.cumsum() + self.time
        self.process_orders(timestamps)
        self.purge_stale_orders()
        self.time = timestamps[-1]

    def process_orders(self, timestamps):
        # if new orders arrived--add them to the book
        n_orders = len(timestamps)
        # draw side (buy/sell) distribution is binomial, p=.5
        order_side = np.random.choice([0, 1], size=n_orders, replace=True)
        # randomly assign traders to each slot
        order_trader = np.random.choice(self.traders, size=n_orders, replace=True)
        price_direction = 1.
        if self.draw_price_direction:
            price_direction = np.random.choice([-1., 1.], size=n_orders, replace=True)
        order_price_shock = np.random.lognormal(0, self.price_sigma, size=n_orders)
        #order_price = self.p + order_price_shock*price_direction + self.sign[order_side] * self.tick_size * self.p
        order_price = self.p * order_price_shock + self.sign[order_side] * self.tick_size * self.p
        order_quantity = np.ceil(np.random.lognormal(1, self.quantity_sigma, size=n_orders))

        for i in xrange(n_orders):
            if timestamps[i] > self.periods:
                break
            side = self.sides[order_side][i]
            quote = {
               "type": "limit",
               "side": side,
               "timestamp": timestamps[i],
               "trade_id": order_trader[i],
               "order_id": self.total_orders_submitted,
               "price": order_price[i],
               'quantity': int(order_quantity[i])
            }
            self.orderbook.process_order(quote, from_data=True, verbose=False)
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

    def plot_prices(self):
        tape = pd.DataFrame(list(self.orderbook.tape))
        tape.set_index('timestamp')['price'].apply(lambda x: float(x)).plot()

    def get_tape_dataframe(self):
        return pd.DataFrame(list(self.orderbook.tape))

    def log_order_type(self, side):
        order_id = self.total_orders_submitted
        current_vol = len(self.orderbook.tape)
        self.orders_by_vol.append((current_vol, str(side), int(order_id)))


sim = Simulator(order_arrival_lambda=1./2, price_sigma=.001)
sim.simulate_batch(size=9000*50)
sim.plot_prices()
len(sim.orderbook.tape)

sim.orderbook.tape[-1]

tape = pd.DataFrame(list(sim.orderbook.tape))

tape['time'] = tape['timestamp'].apply(lambda x: np.floor(x))
