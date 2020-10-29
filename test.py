# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as ST
from scipy import linalg as LA
# %%
# # %% The local linear regression
# We take Gaussian Kernel
# Bandwidth is choosen as 1/T^0.2
# It can be used for multidimensional case. The plot is different


def loc_poly(Y, X):
    N = X.shape[0]
    k = np.linalg.matrix_rank(X)
    m = np.empty(N)
    i = 0
    h = np.std(X)/(N**(1/5))
    # grid = np.arange(start=X.min(), stop=X.max(), step=np.ptp(X)/200)
    for x in X:
        Xx = X - (np.ones(N))*x
        Xx1 = sm.tools.tools.add_constant(Xx, has_constant='add')
        Wx = np.diag(ST.norm.pdf(Xx/h))
        Sx = ((Xx1.T)@Wx@Xx1 + 1e-90*np.eye(k))
        m[i] = ((LA.inv(Sx)) @ (Xx1.T) @ Wx @ Y)[0]
        i = i + 1
    # plt.figure()
    # plt.scatter(X, Y)
    # plt.plot(grid, m, color= 'red')
    # plt.scatter(X, m, color='red')
    return m
# %%


def my_seasonal_series(df, n_seasons):
    """
    Input: a dataframe and number of seasons
    Output: a dataframe of time series for each seasons
    """
    date_list = df.index
    T = len(date_list)
    df_seasonal = [df.loc[date_list[range(i, T, n_seasons)]]
                   for i in range(n_seasons)]
    return df_seasonal


# %%
COVID = pd.read_csv("data_2020-Oct-07_UK.csv")
COVID = pd.pivot_table(COVID, values='newCasesBySpecimenDate', index='date')
COVID.index = pd.to_datetime(COVID.index, format='%Y-%m-%d')
# %%
plt.figure()
plt.plot(COVID)
# %%
T = len(COVID.index)
COVID['normalized_date'] = np.array(range(T))/T
COVID_seasons = my_seasonal_series(COVID, n_seasons=7)

# %%
m = pd.DataFrame([loc_poly(np.array(COVID_seasons[i].iloc[:, 0]), np.array(
    COVID_seasons[i].iloc[:, 1])) for i in range(7)]).transpose()
trend = m.sum(axis=1)
seasonal = m - trend
seasonal = [seasonal.iloc[i,j] for j in range(7) for i in range(38)]
# plt.plot(trend)
for j in range(7):
    plt.plot(seasonal.iloc[:, j])
# plt.plot(seasonal.sum(axis=1))
# %%
fig, ax= plt.subplots(2, 2)
ax[0][0].plot(COVID.iloc[:, 0])
ax[0][1].plot(trend)
ax[1][0].plot(seasonal)
ax[1][1].plot(COVID.iloc[:,0] - trend - seasonal)
# %%
