import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
No_Simulations = 2000


def download_data():
    ticker = yf.Ticker("AUDUSD=X")
    data = ticker.history(start="2014-09-09",end="2024-09-09")['Close']  # considering previous 10 years of data
    stock_repo = pd.DataFrame(data)
    stock_repo["rets"] = np.log(stock_repo["Close"]/stock_repo["Close"].shift(1))
    stock_repo.dropna(inplace=True)
    print(stock_repo)
    return stock_repo

def stimulate_stock (S0,u,sigma):     # using the previous 10 days mean and std of log returns and current price to predict the next days price
    results = []                      # considering the average value of 2000 simulated prices for the next day
    for i in range(No_Simulations):
        price = [S0]
        S = price[-1]*np.exp((u-0.5*(sigma**2))+sigma*np.random.normal())  # (-1) ->index of last stock price
        price.append(S)
        results.append(price)
    res = pd.DataFrame(results)
    res = res.T
    res['mean'] = res.mean(axis=1)
    return res["mean"]

def backtest_strategy(stock_data):
    capital = 1000000  # Starting capital $1M
    position = 0
    daily_returns = []
    capital_over_time = [capital]
    position_changes = []
    wins = 0
    losses = 0
    for i in range(1, len(stock_data)):
        today_price = stock_data["Close"].iloc[i]
        tomorrow_gbm = stock_data["GBM"].iloc[i]

        # Determine new position
        if tomorrow_gbm > today_price:
            new_position = 1
        elif tomorrow_gbm < today_price:
            new_position = -1
        else:
            new_position = 0

        position_changes.append(abs(new_position - position))

        position = new_position

        today_return = position * stock_data["rets"].iloc[i]
        daily_returns.append(today_return)
        capital *= np.exp(today_return)
        capital_over_time.append(capital)

        if today_return > 0:
            wins += 1
        elif today_return < 0:
            losses += 1
    stock_data = stock_data.iloc[1:]

    stock_data.loc[:, "Strategy_Returns"] = daily_returns
    stock_data.loc[:, "Capital"] = capital_over_time[1:]
    total_return = capital / 1000000 - 1
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    max_drawdown = max(1 - stock_data["Capital"] / stock_data["Capital"].cummax())
    turnover = sum(position_changes) / len(stock_data)
    if losses > 0:
        win_loss_ratio = wins / losses
    else:
        win_loss_ratio = float('inf')  # If there are no losses, win/loss ratio is infinite

    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    print(f"Turnover: {turnover:.2f}")
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(stock_data["Capital"], label="Strategy Capital")
    plt.title("Capital Over Time")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    stock_data = download_data()
    window = 10
    stock_data["STD(s)"] = stock_data["rets"].rolling(window).std()
    stock_data["Mean(s)"] = stock_data["rets"].rolling(window).mean()
    stock_data = stock_data.dropna()
    x = np.array(stock_data["Close"])
    print(stock_data)
    y = []
    for i in range(len(x)):
        res = stimulate_stock(x[i], stock_data["Mean(s)"][i], stock_data["STD(s)"][i])
        y.append(res.iloc[-1])
    stock_data["GBM"] = np.array(y)
    print(stock_data)
    print(r2_score(stock_data["Close"][1:], stock_data["GBM"][0:len(stock_data["GBM"]) - 1]))
    stock_data.to_csv("GBM.csv")
    plt.plot(stock_data["Close"][1:],label="blue")
    plt.plot(stock_data["GBM"],label="orange")
    plt.title("GBM Exchange Rate vs Actual Value")
    plt.xlabel("Price in $")
    plt.show()
    backtest_strategy(stock_data)




