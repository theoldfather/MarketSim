#### Task

We adopt an agent-based model in which a fixed number of traders place orders which arrive with Exponential distributed inter-arrival times, implying a Poisson distributed number of orders for a fixed time period. Traders are sampled with replacement and assigned to each order. The quantity for each order is drawn from a lognormal distribution and with an equal probability of being a buy or a sell order. For a sell (buy) order the price is set a small fixed number above (below) the last transaction price plus a log-normally distributed, mean zero random variable. The (log-normally distributed, zero mean) random variable added to the order price is consistent with an equilibrium price function under the assumption of heterogeneous beliefs about the true price process.

As our simulation starts, orders begin to populate the limit order book. Each incoming order is compared to previously placed orders by price and time priority. If a match between a buy and sell order is made, a transaction takes place. If an order is only partially filled, we attempt to match the remaining quantity against other orders on the book, and if no match is found, the order remains in the limit order book. In order to avoid stale limit orders, each order expires and is withdrawn from the market after 100 subsequent transactions are executed.

We only considered the trades and build the network of trading counterparties. We record trading price, trading volume, time of transaction, buyer, seller.

For startin‎g price we used a reference price of 10 (I'll check).


#### XLS

We simulate orders until we attain 11,992,200 transactions, segment the data into 19,987 networks of 600 consecutive transactions, and compute simulated network and financial variables. The first spreadsheet contains the data (only a few rows – in total we have 19,987 rows). The second spreadsheet contains the explanation of each variable. In the paper we use only a few of them.
