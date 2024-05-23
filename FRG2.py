#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta

data = pd.read_csv("Nat_gas.csv")
prices = data["Prices"].values
dates = data["Dates"].values
plt.plot(dates, prices)
plt.show()

start_date = date(2020, 10, 31)
end_date = date(2024, 9, 30)
months = []
year = start_date.year
month = start_date.month + 1
while True:
    current = date(year, month, 1) + timedelta(days=-1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    else:
        month = ((month + 1) % 12) or 12
        if month == 1:
            year += 1

days_from_start = [(day - start_date).days for day in months]
dfs = np.array(days_from_start)
slope, intercept, r, p, std_err = stats.linregress(dfs, prices)
plt.plot(dfs, prices)  # to show the upwards trend
plt.plot(dfs, dfs * slope + intercept)
plt.show()
sin_prices = prices - (dfs * slope + intercept)
sin_func = np.sin(2 * np.pi * dfs / (365))  # noticing a period rise and fall along with an upwards trend
cos_func = np.cos(2 * np.pi * dfs / (365))


def bilinear_regression(y, x1, x2):
    A = np.sum(y * x1) / np.sum(x1 ** 2)
    B = np.sum(y * x2) / np.sum(x2 ** 2)
    return (A, B)


A, B = bilinear_regression(sin_prices, sin_func, cos_func)  # P=Asin(x)+Bcos(x)
Amplitude = np.sqrt(A ** 2 + B ** 2)
phase_shift = np.arctan2(B, A)

time = np.arange(0, np.max(days_from_start) + 1)
smooth_estimate = Amplitude * np.sin(time * 2 * np.pi / 365 + phase_shift) + time * slope + intercept
final_price = np.zeros(np.max(days_from_start) + 1)
for t in time:
    if t in days_from_start:
        final_price[t] = prices[days_from_start.index(t)]
    else:
        final_price[t] = smooth_estimate[t]
plt.plot(days_from_start, prices, 'o')
plt.plot(time, final_price)
plt.plot(time, pd.DataFrame(smooth_estimate))
plt.show()

iw_rate = float(
    input("Enter the injection & withdrawal rate per million MMBtu :"))  # injection & withdrawal rate per unit volume
storage_cost = float(input("Enter total storage charge :"))  # storage cost
V = float(input("Enter the maximum volume in million MMBtu :"))  # maximum_volume
m = int(input("Enter No. of Injection Dates :"))
n = int(input("Enter No. of Withdrawal Dates :"))

prices_at_injec_dates = []
prices_at_withd_dates = []

for j in range(m):  # Noting down the prices of Natural Gas at injection dates
    inj = input("Enter a date (MM DD YY format): ")
    inj = datetime.datetime.strptime(inj, "%m/%d/%y").date()
    d1 = (inj - start_date).days
    if d1 in time:
        prices_at_injec_dates.append(final_price[d1])
    else:
        prices_at_injec_dates.append(Amplitude * np.sin(d1 * 2 * np.pi / 365 + phase_shift) + d1 * slope + intercept)

for k in range(n):  # Noting down the prices of Natural Gas at withdrawal dates
    wth = input("Enter a date (MM DD YY format): ")
    wth = datetime.datetime.strptime(wth, "%m/%d/%y").date()
    d2 = (wth - start_date).days
    if d2 in time:
        prices_at_withd_dates.append(final_price[d2])
    else:
        prices_at_withd_dates.append(Amplitude * np.sin(d2 * 2 * np.pi / 365 + phase_shift) + d2 * slope + intercept)

print(prices_at_injec_dates)
print(prices_at_withd_dates)

# assuming that there are m input dates and n withdrawal dates and the prices on these dates are a1,a2,a3...am prices
# withdrawal prices b1,b2,b3....bn  prices volumes injected on the repective dates X1,X2..Xm
# assume that withrawal volume on these dates are Y1,Y2,Y3.....Yn
# as we know that the price of the contract is the profit = selling amount -buy amount -storage cost-2*V*iwr
# P=(b1Y1+b2Y2+b3Y3+...bnYn)-(a1X1+a2X2+a3X3+...anXn) - storage_cost - 2*Maximum_volume*iwr
# max(P)=max((b1Y1+b2Y2+b3Y3+...bnYn))-min((a1X1+a2X2+a3X3+...anXn))-storage_cost - 2*iw_rate*Maximum_volume

from scipy.optimize import minimize


def objective_x(X):
    a = prices_at_injec_dates
    return sum(a[i] * X[i] for i in range(len(X)))


def constraint_eq_x(X):
    return (sum(X) - V)


# making an initial guess
X0 = [1] * m
# defining the bounds
bounds_x = [(0, None)] * m
# defining the constraints
constraint_x = {'type': 'eq', 'fun': constraint_eq_x}
# finally minimizing the objective function under the given constraint
result_x = minimize(objective_x, X0, bounds=bounds_x, constraints=constraint_x).fun


# similarly we can evaulate the maximum of b1Y1+b2Y2+b3Y3+...bnYn

def objective_y(Y):
    b = np.array(prices_at_withd_dates) * (-1)
    return sum(b[i] * Y[i] for i in range(n))


def constraint_eq_y(Y):
    return (sum(Y) - V)


# making an initial guess
Y0 = [1] * n
# defining the bounds
bounds_y = [(0, None)] * n
# defining the constraints
constraint_y = {'type': 'eq', 'fun': constraint_eq_y}
# finally minimizing the objective function under the given constraint
result_y = minimize(objective_y, Y0, bounds=bounds_y, constraints=constraint_y).fun * (-1)

price_of_contract = (result_y - result_x) * (10 ** 6) - storage_cost - (2 * V * iw_rate)

print(price_of_contract)


# In[ ]:




