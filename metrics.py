def cal_sharpe_ratio(portfolio_return, risk_free_rate):
    '''
        Function to calculate Sharpe ratio on a portfolio
    '''
    # Sharpe ratio = (Rp - Rf)/(std(Rp))
    return 0 if np.std(portfolio_return) == 0 else (np.mean(portfolio_return) - risk_free_rate) / np.std(portfolio_return)