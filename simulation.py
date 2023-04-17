import numpy as np
from metrics import cal_sharpe_ratio
from env.version_0_build import StockStreet
from plot_utils import plot_performance


def trade(player, trained_model, stocks, start_cash, start_date, end_date, trade_fee, max_stock_order,
          scaling_factor, window, render, rolling_action=30, reward_scaling=1.0, plot=True):
    '''
    A Trading simulation of the environment over the selected period. Function allows the simulation of several player
    Functionalities and output the performances of the selected player. A long Only player takes a buying actions until
    their money drops below 5% and holds this position for the remaining of the period. A Clueless Player takes random actions
    throughout the simulation.

    Parameters:
    -------------------------
    player - str : Name of the player can be Clueless (random actions), Long Only(Buys Only and then Holds), PPO, A2C
    trained_model : If an agent , a model with a predict attribute. Else, None
    stocks - list: A list of the ticker names of stocks to trade
    start_cash - float: The starting portfolio cash
    start_date - str: The start date of trading
    end_date - str: The end date of trading
    trade fee - float: A comission fee every time a buy/sell order is executed
    max_stock_order - int: the maximum number of stock that can be purchased/sold per day
    scaling_factor - float: A factor applied to prices in the observation space to be useful for training
    window - int: A sliding window of historical prices to be shown as part of the observation of the environment
    render - boolean: If true, activates dynamic rendering
    rolling_action - int: The number of days to take the moving average of actions
    reward_scaling - float: The additional reward scaling
    plot - boolean: If true plots the performances of the player

    Returns:
    -----------------------
    rewards - list: A list of the player's rewards
    sharpe_rations - list: A list of the player's monthly sharpe ratios
    actions - list: A list of the player's actions
    env.actions - list: A list of the actions ajdusted by the environment of the player attempts to sell with no stock or buy with <= 5% of initial cash
    env.dates - series: A series of the dates of which stock information is available during the trading interval
    env.dones - list: A list of every day the player's cash went below the 5% of init cash limit
    '''

    assert player in ["Clueless", "Long", "PPO", "A2C", "Training"]

    env = StockStreet(stocks, init_date=start_date, end_date=end_date,
                      init_capital=start_cash, buy_fee=trade_fee, sell_fee=trade_fee,
                      max_stock_trade_qty=max_stock_order, set_seed=72, scale_factor=scaling_factor,
                      window_size=window, dyn_render=render, full_render=True, reward_add_scale=reward_scaling)
    rewards = []
    actions = []
    port_returns = []
    sharpe_ratios = []
    obs = env.reset()
    i = 0

    while True:

        if player == "Clueless":
            action = env.action_space.sample()

        elif player == "Long":
            action = np.array([1] * len(stocks))

        else:
            action, _states = trained_model.predict(obs)

        actions += [action]
        obs, reward, done, info = env.step(action)

        # if (player in ["A2C","PPO", "Training"]) * (len(env.dones) > 0):
        #     rewards += [0]
        #     port_returns = 0
        # else:
        rewards += [reward]
        port_returns += [info['Portfolio_return']]

        i += 1

        if i % 30 == 0:
            sharpe_ratios += [cal_sharpe_ratio(port_returns, ((1 + 0.02) ** (1 / 365)) - 1)]

            port_returns = []

        if done:
            break

    if plot:
        plot_performance(player, rewards, sharpe_ratios, actions, rolling_action,
                         scale=scaling_factor * reward_scaling, dates=env.dates, window=window)

    return rewards, sharpe_ratios, actions, env.actions, env.dates, env.dones