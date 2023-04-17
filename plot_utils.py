import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_performance(player, rewards, sharpe_ratios, actions, roll, scale, dates, window):
  '''
    Plots the performance of a player based on four different critera:

    1 - The cumulative Reward
    2 - The 30-day Moving Average Reward
    3 - The monthly Portfolio Sharpe Ratio
    4 - The moving average actions taken by the Player

    Parameters:
    --------------------------------
    player - str : The name of the Player
    rewards - list: A list of the players rewards
    sharpe_ratios - list: A list of the player's monthly sharpe ratios
    actions - list: A list of the player's actions
    roll - int: The number of days to take an action average
    dates - series: The series of the dates selected during the evenironment setting
    window - int: The size of the trading window
  '''


  plt.figure(figsize = (10,10))
  plt.plot(dates[window: window + len(rewards)], np.cumsum([rewards]) / scale)
  plt.title( player + " Cumulative Reward")
  plt.show()

  plt.figure(figsize = (10,10))
  plt.plot( dates[window: window + len(rewards)] ,np.array(rewards) / scale)
  plt.title(player + " Rewards")
  plt.show()

  plt.figure(figsize = (10,10))
  plt.plot(dates[window + 30 : len(rewards) + window : 30],
            sharpe_ratios)
  plt.title( player + " Sharpe Ratio")
  plt.show()

  plt.figure(figsize = (10,10))
  plt.plot(dates[window :  window + len(rewards)],
            pd.DataFrame(actions).rolling(roll).mean())
  plt.title(player + " Actions")
  plt.show()