import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import gym
import talib
from calculations import get_scalar

class StockStreet(gym.Env):
  metadata = {"render_modes": ["human"]}

  def __init__(self, stock_list, init_date="2009-12-31", end_date="2019-12-31", init_capital=1e6, buy_fee=1e-2,
               sell_fee=1e-2,
               max_stock_trade_qty=100, set_seed=72, log_transform=False, scale_factor=1e-3, reward_add_scale=1e-2,
               window_size=252 / 2,
               dyn_render=False,
               full_render=True, train_agent=False):
    '''
    StockStreet is a stock trading environment that allows a user or agent to trade multiple stocks (up to 2 stocks in current build)
    in a period of the user's choosing. Several capabilities have been implemented in this environment to allow for training using a reiforcement
    learning algorithm. This includes the ability to track the player's portfolio as well as the technical trading
    signals associated to selected stocks.

    Parameters:
    -------------------------------------
    stock_list - List: A list of the stocks to be tracked
    init_date - Str: The start date of trading
    end_date - Str: The end date of trading
    init_capital - float: The initial capital to be had in the portfolio
    buy_fee - float: A comission fee every time a buy order is executed
    sell_fee - float: A comission fee every time a sell order is executed
    max_stock_trade_qty - int: The maximum number of stocks that can be bought/sold in an order
    set_seed - int: set a seed for environment
    log_transform - boolean: If prices in observation space to be log transformed
    scale_factor - float: A factor applied to prices in the observation space to be useful for training (used if log transform False)
    window_size - int: A sliding window of historical prices to be shown as part of the observation of the environment
    dyn_render - boolean: If true, renders a plot of actions and market prices during the running of the environment. Shows only information within
                          the specifiied sliding window
    full_render - boolean: If true, allows the capaibility to create a full render after a specific time interval into the running of the
    environment
    train_agent - boolen: If true, environment adapts to a training agent
    '''

    self.stock_info = yf.download(stock_list, start=init_date, end=end_date,
                                  period="1d")  # create dataframe with stock info
    self.stock_names = stock_list  # create a list with stock names
    self.n_stocks = len(stock_list)  # number of stocks to be tracked
    self.dates = self.stock_info.reset_index().Date
    self.volatility = yf.download("^VIX", start=init_date, end=end_date, period="1d")  # Track volatility

    # Take only stock closing price
    if self.n_stocks > 1:
      self.stock_prices = self.stock_info.loc[:, "Close"]
    else:
      self.stock_prices = self.stock_info[["Close"]].rename(columns={"Close": stock_list[0]})

    self.init_portflio_val = init_capital  # initial captial
    self.buy_fee = buy_fee  # Commission on Buy
    self.sell_fee = sell_fee  # Commission on Sell
    self.max_stock = max_stock_trade_qty  # max number of tradabale stock in a period
    self.portfolio_shape = self.n_stocks + 1  # Portfolio size: N_stocks * (number of stocks owned) + cash on hand
    self.market_shape = self.n_stocks  # track each stock in the market
    self.signal_shape = 8 * self.n_stocks + 1  # 8 stock-related trading signals * n_stocks + 1 VIX signal
    self.seed = set_seed  # to ensure environment starts same on each episode
    self.max_days = self.stock_prices.shape[0]  # to track when the trading period is over
    self.log = log_transform  # to transform prices
    self.scale = scale_factor  # to scale prices
    self.reward_scaling = self.scale * reward_add_scale  # to scale rewards
    self.window = window_size  # size of the window to be retained from historical stock information
    self.dyn_render = dyn_render  # renders an environent during training at a fixed interval
    self.full_render = full_render  # renders a full picture of actions and the market
    self.dones = []  # to track wehre cash has run low when not training
    self.agent_train = train_agent  # boolean for environment to know if an agent is being trained

    # Observation space consists the current porfolio and market conditions. Specifically within the portfolio,
    # an observation store the current cash, number of stocks, and stock value in the porfolio.The market on the other hand
    # stores the prices of each each stock up to historically the sepcified window.
    # self.observation_space = gym.spaces.Dict({"Portfolio": gym.spaces.Box(-np.inf,np.inf,shape = (self.portfolio_shape,)),
    #                                           "Market": gym.spaces.Box(-np.inf,np.inf,shape = (self.market_shape,))
    #                                           })
    self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                            shape=(self.portfolio_shape + self.market_shape + self.signal_shape,))

    # Actions are made to be discrete. The player has the choice to pick any dicrete action between the maximum
    # minimum stock chosen in parameters. This discrete setting needs to be created for each stock that can be traded.
    self.action_space = gym.spaces.Box(-1, 1, seed=self.seed, shape=(self.n_stocks,),
                                       dtype=float)

  def _get_obs(self, market_prices):
    '''
    Creates an observation by either scaling or log transforming real-time pricing data.

    Parameters:
    ----------------------------
    market_prices - Array shape (window_size,num_stocks): Has the current historical market prices for each stock within a specified window

    Returns:
    -------------------------
    A dictionary with the current portfolio and market condions. Within the "Portfolio" key the user has access to an observation
    of their cash on hand, number of stocks, and stock value. In the "Market" key the user has access to transformed market prices
    '''
    # if self.log:
    #   return{"Portfolio": (np.log(self.cash),np.log(self.stock_total_cost),self.stocks),
    #          "Market": np.log(market_prices)
    #   }
    # else:
    #   return{"Portfolio": (self.cash * self.scale,self.stock_total_cost * self.scale,self.stocks),
    #          "Market": self.scale * market_prices
    #   }

    # signals are scaled by the signal scalars calculated at environment reset
    signals = np.hstack(self._get_signals()) / self.signal_scalars

    # cash and number of stocks scaled by the portfolio scalars calculated at environment reset
    if self.log:
      pass
    # return np.hstack((np.log(self.cash),np.log(self.stock_total_cost), self.stocks,np.log(market_prices)))
    else:
      return np.hstack((self.cash / self.portfolio_scalars[0], self.stocks / self.portfolio_scalars[1],
                        self.scale * market_prices[-1, :],
                        signals))

  def _get_info(self, market_prices, signals):

    '''
    Generates real time information about the environment. This is additional information that can be used in addition to the observation
    space.

    Parameters:
    ---------------------------
    market_prices - Array shape (window_size,num_stocks): Has the current historical market prices for each stock within a specified window

    Returns:
    ---------------------------
    A dictionary with the current day of training, cash on hand, numberof stocks, and real market prices of each stock (not transformed or scaled)
    '''
    return {"Time": self.day,
            "Cash": self.cash,
            "Stocks": self.stocks,
            "Last_prices": market_prices[-1, :],
            "VIX": self.volatility[self.day - 1: self.day].Close.values[0],
            "Portfolio_return": self._get_returns(),
            "RSI": signals[0],
            "MACD": signals[1],
            "MACD_signal": signals[2],
            "Price_momentum": signals[3],
            "WILL_%R": signals[4],
            "ATR": [5],
            "30_d_MA": signals[6],
            "OBV": signals[7],
            "ATR_VIX": signals[8]
            }

  def _get_returns(self):
    # get current price
    current_prices = self.stock_prices.iloc[self.day, :].values

    # get previous price
    previous_prices = self.stock_prices.iloc[self.day - 1, :].values

    # get price diff
    price_diff = (current_prices - previous_prices) / previous_prices

    # calculate current values of each stock
    current_values = self.stocks * current_prices

    # calculate weights
    weights = current_values / self.assets
    weight_cash = 1 - weights
    daily_cash_return = ((1 + 0.02) ** (1 / 365)) - 1
    # return the portfolio returns:
    return (price_diff * weights).sum() + weight_cash * daily_cash_return

  def _get_signals(self):

    '''
    Returns some technical trading signals that are commonly used in algorithmic trading to
    to make trade decisions

    Returns:
    -------------------------
    Returns several trading signals regarding each stock in the portfolio to as well as
    as well as a tracking of the volatility index (VIX) in the market
    '''
    #     scale = 1 if scale == False else self.scale

    rsi = []  # Relative Strength Index
    macd = []  # Moving Average Convergence/Divergence
    macd_signal = []  # Moving Average Convergence/Divergence Signal
    momentum = []  # Price Momentum - 2 month
    willr = []  # William %r
    atr = []  # Average True Range
    ma_30 = []  # 30 Day Moving Average
    atr_vix = []  # Average True Range VIX
    obv = []  # On-Balance Volume

    # Get stock info for relevant study window
    stock_info = self.stock_info[self.day - self.window: self.day]
    stock_info_close = stock_info.Close.values.reshape(len(stock_info), self.n_stocks)  # Day Close Price
    stock_info_high = stock_info.High.values.reshape(len(stock_info), self.n_stocks)  # Day High Price
    stock_info_low = stock_info.Low.values.reshape(len(stock_info), self.n_stocks)  # Day Low Price
    stock_info_vol = stock_info.Volume.values.reshape(len(stock_info), self.n_stocks)  # Traded Volume
    volatility_info = self.volatility[self.day - self.window: self.day]  # Get volatility info for study window

    # signal data for each stock
    for i in range(self.n_stocks):
      rsi += [talib.RSI(stock_info_close[:, i])[-1]]  # Get the last RSI for period
      macd += [talib.MACD(stock_info_close[:, i])[0][-1]]  # Get last MACD for period
      macd_signal += [talib.MACD(stock_info_close[:, i])[-1][-1]]  # Get last MACD signal for period
      momentum += [
        talib.MOM(stock_info_close[:, i], timeperiod=60)[-1]]  # Get last momentum calculated over a 60 day period
      willr += [
        talib.WILLR(stock_info_high[:, i], stock_info_low[:, i], stock_info_close[:, i])[-1]]  # Calculate last WILL %R
      atr += [
        talib.ATR(stock_info_high[:, i], stock_info_low[:, i], stock_info_close[:, i])[-1]]  # Get last ATR reading
      ma_30 += [talib.SMA(stock_info_close[:, i])[-1]]  # Get latest 30 day price moving average
      obv += [talib.OBV(stock_info_close[:, i], stock_info_vol[:, i].astype(float))[-1]]  # Get latest OBV signal

    # Get ATR of the volatility index
    atr_vix += [talib.ATR(volatility_info.High.values, volatility_info.Low.values, volatility_info.Close.values)[-1]]

    rsi = np.array([si / 100 for si in rsi])  # convet percent to decimal
    macd = np.array(macd)
    macd_signal = np.array(macd_signal)
    momentum = np.array(momentum)
    willr = np.array([r / 100 for r in willr])  # convet percent to decimal
    attr = np.array(atr)
    ma_30 = np.array(ma_30)
    obv = np.array(obv)
    atr_vix = np.array(atr_vix)

    # vix_day = volatility_info.Close.values[-1]
    # vix_30 = volatility_info.Close.values[-30]

    return rsi, macd, macd_signal, momentum, willr, atr, ma_30, obv, atr_vix

  def _window_render(self, market_prices):
    '''
    Renders the environment during operation within the specified initialized sliding window noting that:
       - The x-axis is a the number of days passed since starting the environment.
       - The primary y-axis in the order of stocks bought/or sold (actions)
       - The secondary y-axis is the makret price of the stock

    '''
    # Stores all historical actions taken into a numpy array
    hist_actions = np.array(self.actions) * self.max_stock
    # Creates as many subplot as stocks
    fig, ax = plt.subplots(1, self.n_stocks, figsize=(20, 5), sharex=True, sharey=True)

    if self.n_stocks == 1:
      ax = [ax]
    # create a secondary axis to track stock price
    ax2 = [axes.twinx() for axes in ax]

    # Add title
    fig.suptitle(f"Market Overview - {self.day} Days passed ")

    # Create sequence of time in specified window
    time = self.dates[self.day - self.window:self.day]

    # Create subplot of each tick
    for i in range(0, self.n_stocks):
      colors = ["red" if action < 0 else "green" for action in
                hist_actions[self.day - self.window:self.day, i]]  # Red for sell order and green for buy order
      ax[i].bar(time, hist_actions[self.day - self.window:self.day, i], color=colors)  # Barplot of orders
      ax[i].set_title(self.stock_names[i])  # add subtitles of stock names
      ax[i].set_xlabel("Date")  # Add x-label
      ax2[i].plot(time, market_prices[:, i], color="orange")  # lineplot of the market price within specifiedwinfow

    # Set Primary y-label
    ax[0].set_ylabel("Stocks Ordered")

    # Set Secondary y-label
    ax2[-1].set_ylabel("Stock Price (USD)")

    # show plot
    plt.show()

  def reset(self):
    '''
    Resets the evironment.

    Returns:
    --------------
    The observation and environment infor defined
    '''
    np.random.seed(self.seed)

    # reset cash in portfolio
    self.cash = self.init_portflio_val

    # reset the number of days passed since beginning of trading
    self.day = self.window

    # reset the value of stock in portfolio
    self.stock_total_cost = np.zeros(self.n_stocks)

    # Setting the previous market prices
    buy_prices = self.stock_prices.iloc[self.day - 1, :].values

    # as to the observations in similar scale all observations in the state shall be scaled
    # to the same scale as the scale sctock price. To do this we will get an idea of how
    # large/small the average stock price at the beginning of the trading is and scale all
    # observations by this value. This constant scalar value is only calculated at reset
    # as to not influence the changing output of each value part of the observation at each step
    self.scalar = self.scale * get_scalar(np.mean(buy_prices))

    # Find the scalars for each of the trading signals
    self.signal_scalars = get_scalar(np.hstack(self._get_signals())) / self.scalar

    # Randomize the amount of stock held in portfolio prior to start of trading
    buy_share = 0
    self.stocks = np.random.randint(0, self.max_stock, size=self.n_stocks).astype(float)

    # Random action continued until 10% of the initial portfolio has cash
    while buy_share < 0.1 * self.cash:
      # Randomize the amoutn of stock held in prtfolio prior to start of trading
      self.stocks += np.random.randint(0, self.max_stock, size=self.n_stocks).astype(float)
      buy_share = (self.stocks * buy_prices).sum()

    # Calculating the cost
    self.stock_total_cost += self.stocks * buy_prices
    order = self.stocks * buy_prices
    order = np.where(order < 0, (1 + self.sell_fee) * order, (1 + self.buy_fee) * order)
    self.cash -= order.sum()

    # Update total assets
    self.assets = self.cash + (self.stocks * buy_prices).sum()

    # Get market prices
    market_prices = self.stock_prices.iloc[self.day + 1 - self.window:self.day + 1, :].values

    # calculate scalars for the cash and # of stocks that form as the player's portfolio
    self.portfolio_scalars = [get_scalar(np.array(self.cash)) / self.scalar, get_scalar(self.stocks) / self.scalar]

    # list to store the returns
    self.returns = []

    # render envionment - If true, stores all actions
    if self.dyn_render + self.full_render:
      self.actions = [[0] * self.n_stocks] * (self.window)
      # self.actions += [list(self.stocks / self.max_stock)]

      # renders environment environment in set window
      if self.dyn_render:
        self._window_render(market_prices)

    return self._get_obs(market_prices)

  def step(self, actions):
    '''
    Stepping into the environment.

    Parameters:
    --------------------
    actions - Array: the order action specified by the player for each stock

    Returns:
    ---------------------
    The observations based on these actions, the reward, whether trading is done, environment info
    '''

    # start new day
    self.day += 1

    # If the environemnt is used in a non training setting, actions shall be
    # automatically overwritten to zero if the cash held is less than 5%
    if (self.cash <= 0.05 * self.init_portflio_val):
      actions = np.array([0 if a > 0 else a for a in actions])
      self.dones += [self.day]  # keeps track of the days of which the portfolio cash reached this level

    # Get current market prices
    market_prices = self.stock_prices.iloc[self.day + 1 - self.window: self.day + 1, :].values

    # Stores actions if rendering on
    if self.dyn_render + self.full_render:
      self.actions += [list(actions)]

      # displays sliding window render every 30 days
      if (self.dyn_render) & (self.day % 30 == 0):
        self._window_render(market_prices)

    # The prices at which the stocks are to be ordered at
    buy_prices = self.stock_prices.iloc[self.day - 1, :].values

    # Total assets = cash + num_stocks * price of day -1
    assets = self.cash + (self.stocks * buy_prices).sum()

    # Calculate reward based on order and previous asset value
    if self.log:
      reward = np.log(assets - self.assets)
    else:
      reward = (assets - self.assets) * self.reward_scaling

    if (self.cash <= 0.05 * self.init_portflio_val) * self.agent_train:
      reward = 0

    # Update total assets to new order after reward calculation
    self.assets = assets

    # Limiting actions according to stocks in hand
    actions = [max(-cur_val, action_val * self.max_stock) for cur_val, action_val in zip(self.stocks, actions)]

    # Accumulate cost of each stock
    self.stock_total_cost += actions * buy_prices

    # Place order
    order = actions * buy_prices

    # Account for comission fees
    order = np.where(order < 0, (1 + self.sell_fee) * order, (1 + self.buy_fee) * order)

    # Update cash based on order
    self.cash -= order.sum()

    # Update the amount of each stock
    self.stocks += actions

    # Environment ends if max number of days reached of cash 5% of initial amount during training of agent
    if self.agent_train:
      # done = ((self.day >= (self.max_days - 1)) + (self.cash <= 0.05 * self.init_portflio_val)) >= 1
      done = (self.day >= (self.max_days - 1))
    # Environment ends if max number of days reached of cash 5% when run under normal settings
    else:
      done = self.day >= (self.max_days - 1)

    return self._get_obs(market_prices), reward, done, self._get_info(market_prices, self._get_signals())

  def render(self, average):
    '''
    Allows a full rendering of the environment up to the current trading day.

    Parameters:
    --------------------
    average - int: The moving average period for orders and frequency of display on plot

    Output:
    --------------------
    Renders the environment for from the first day of trading to current day of trading:
       - The x-axis is a the number of days passed since starting the environment.
       - The primary y-axis in the order of stocks bought/or sold (actions)
       - The secondary y-axis is the makret price of the stock

    NOTE: FUNCTION IS STILL UNDER CONSTRUCTION AS THE MOVING AVERAGE FEATURE DOES NOT YET
    FUNCTION CORRECTLY
    '''

    # Setting up the x-axis
    time = np.arange(0, self.day)

    # storing historical actions as numpy array
    hist_actions = np.array(self.actions)

    # create subplots based on number of stocks tracked
    fig, ax = plt.subplots(1, self.n_stocks, figsize=(20, 5), sharex=True, sharey=True)

    # create secondary y-axis
    ax2 = [axes.twinx() for axes in ax]

    # create title
    fig.suptitle(f"Full Market Overview")

    # create subplot for each stock
    for i in range(0, self.n_stocks):
      stock_actions = hist_actions[:, i]
      j = self.window
      moving_averages = []

      # calculate moving average of orders placed
      while j < len(stock_actions) - average + 1:
        window = stock_actions[j: j + average]
        window_average = round(np.sum(window) / average, 0)
        moving_averages.append(window_average)
        j += 1

      # Stores moving average only every "average" period
      moving_average_time = np.arange(average, len(moving_averages), average)
      moving_averages = [avg for i, avg in enumerate(moving_averages) if i % average == 0]
      moving_averages = moving_averages[1:]

      # plot orders and market
      colors = ["red" if action < 0 else "green" for action in moving_averages]
      ax[i].bar(moving_average_time, moving_averages, color=colors)
      ax[i].set_title(self.stock_names[i])
      ax[i].set_xlabel("Days")
      ax2[i].plot(time, self.stock_prices.iloc[:self.day, i].values, color="orange")

    ax[0].set_ylim((-100, 100))
    ax[0].set_ylabel("Stocks Ordered")
    ax2[-1].set_ylabel("Stock Price (USD)")
    plt.show()