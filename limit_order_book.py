import heapq
import queue as Q
import numpy as np

pq = Q.PriorityQueue()

# limit order book

# simulator
# given


class Simulator(object):
    def __init__(self, order_arrival_lambda=2, initial_price=10, price_sigma=1, periods=60*60*7, traders=10 ):
        self.initial_price = initial_price
        self.order_arrival_lambda = order_arrival_lambda
        self.price_sigma = price_sigma
        self.period = periods
        if type(traders) == int:
            traders = range(traders)
        self.traders = traders

    def simulate_time_period(self):
        # number of orders that arrive
        n_orders = np.random.poisson(self.order_arrival_lambda, size=1)
        # randomly assign traders to each slot
        new_trades = np.random.choice(self.traders, size=n_orders, replace=True)

        if len(new_trades) > 0:
            # if new trades arrived--add them to the book
            # buy/sell distribution is binomial, p=.5
            order_type = np.random.binomial(1, p=.5, size=n_orders)
            
            print(new_trades)


sim = Simulator()
sim.simulate_time_period()