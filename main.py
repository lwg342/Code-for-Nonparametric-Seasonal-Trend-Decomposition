# -> Created on 04 October 2020
# -> Author: Weiguang Liu
# %% Libraries
from importlib import reload as rl
rl(NPSTL)
from NPSTL import NPSTL_Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
''' Simulation'''
Y = 5*np.sin(np.arange(0, 2392)/200) + \
    np.random.default_rng().standard_normal(2392)
# %%
"""============="""
"""The Empirical Part on COVID data"""
"""============="""
# %% Import data
Data = pd.read_csv("/Users/lwg342/Downloads/data_2020-Oct-07_UK.csv")
Y = np.array(np.flip(Data.iloc[:, 4]))
plt.figure()
plt.scatter(range(len(Y)), Y)
# %%
'''Use Local Linear Method'''
'''--------------------------------'''
M = NPSTL_Model(Y[50:-2], 7)
M.fit()
M.plot(save = True, title = 'local-linear-Covid')
# %%
'''Using BSpline Method'''
'''--------------------------------'''
M = NPSTL_Model(Y[50:-2], 7) # Construct a model with time series Y, number of seasons = J
R = M.fit(method='BSpline') # Fit the model with B-Spline Method
M.plot(save = True, title= 'BSPline-Covid')

# %%
'''
Application on CO_2 data
'''
Data = pd.read_csv("co2_data_cleaned.csv")
Y = np.array(Data['-999.99'])
Y = Y[Y != -999.99]
# %%
'''Using BSpline Method'''
'''--------------------------------'''
M = NPSTL_Model(
    Y[:-2], 7)  # Construct a model with time series Y, number of seasons = J
R = M.fit(method='BSpline')  # Fit the model with B-Spline Method
M.plot(save=True, title='BSPline Covid')

# %%
'''Application on Scots Death Number'''
'''--------------------------------'''
Data = pd.read_excel("scotsdeaths.xlsx")
Data = Data.drop([0,1,2] + list(range(56,64)))
Data = Data.drop(Data.columns[[0,-1]], axis=1)
Y = (np.array(Data)).flatten()
# %%
plt.plot(Y)