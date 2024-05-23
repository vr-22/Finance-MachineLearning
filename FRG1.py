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

inp_date = input("Enter a date (MM DD YY format): ")
inp_date = datetime.datetime.strptime(inp_date, "%m/%d/%y").date()
d = (inp_date - start_date).days
if d in time:
    print(final_price[t])
else:
    print(Amplitude * np.sin(d * 2 * np.pi / 365 + phase_shift) + d * slope + intercept)


# In[ ]:




