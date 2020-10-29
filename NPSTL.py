# %%
# -> Created on 26 October 2020
# -> Author: Weiguang Liu
# %%
import sys
sys.path.append('/Users/lwg342/Documents/GitHub/Python-Functions/')
sys.path.append('C:/Users/wl342/Documents/Github/Python-Functions/')
from WLRegression import BSpline as BSpline
from WLRegression import OLS, matlocl
from scipy import stats as ST
from scipy import linalg as LA
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
'''The Class of Nonparametric STL Model'''

class NPSTL_Model():
    """
    This is the class of models for Nonparametric STL projects
    """

    def __init__(self, Y, NUM_SEASON):
        self.Y = Y
        self.X = np.linspace(0, 1, len(self.Y))
        self.J = NUM_SEASON

    def fit(self, method='Local Linear'):
        from WLRegression import matlocl
        v = list(range(self.J))
        tt = np.arange(len(self.Y)) % self.J
        n = len(self.Y)
        dd = np.array([[(i == j)*1.0 for j in v] for i in tt])
        self.dd = dd
        X = self.X
        y = self.Y
        if method == 'Local Linear':

            # yy = np.diag(self.Y)@dd
            # xx = np.diag(x)@dd
            trend = np.zeros([n, self.J])
            for j in range(self.J):
                yj = np.array([y[i] for i in range(j, n, self.J)])
                xj = np.array([X[i] for i in range(j, n, self.J)])
                w = matlocl(xj, X)
                trend[:, j] = w@yj

            mtrend = trend.sum(axis=1) / self.J
            # m11 = 1
            # m12 = self.mtrend.sum()
            # m21 = m12
            # m22 = (self.mtrend**2).sum()
            # m = np.array([[m11, m12], [m21, m22]])
            alpha = []
            for j in range(self.J):
                # w1 = trend.sum()
                # w2 = (trend[:, j]*self.mtrend).sum()
                # w = np.array([w1, w2])
                yj = np.array([y[i] for i in range(j, n, self.J)])
                mtrendj = np.array([mtrend[i]
                                    for i in range(j, len(mtrend), self.J)])
                alpha = alpha + [OLS(mtrendj, yj).beta_hat()]
                # self.alpha = (alpha - alpha.mean(axis=0))[0]
                # self.beta = (alpha - alpha.mean(axis=0))[1]
            # print(w1)
            alpha = np.array(alpha)
            mtrend_dd = np.diag(mtrend)@dd
            seasonal = dd@alpha[:, 0] + mtrend_dd@alpha[:, 1]

            self.alpha = alpha
            self.mtrend = mtrend
            self.seasonal = seasonal - mtrend
            self.residual = self.Y - self.mtrend - self.seasonal
            
        if method == 'BSpline':
            from scipy.optimize import least_squares
            dim_basis = BSpline(self.X, self.Y, 3).dim_basis
            '''
            Notice the -2 in the dimension of the parameters, it comes from the two restriction we put on alpha, beta
            sum alpha = 0, sum beta = 1
            '''
            print('dim_basis ', dim_basis)
            print('J ', self.J)
            print(len(np.zeros(2*self.J + dim_basis - 2)))
            result = least_squares(self.bspline_resid, x0 = np.zeros(2*self.J + dim_basis - 2), verbose = 1)
            self.mtrend, self.seasonal, self.residual = self.bspline_resid(result.x, option = 'full')
            return result


    def report(self):
        """
        Report the results
        """
        print('The alpha, beta coeffs are \n', self.alpha)

    def plot(self, option='plot components', save = False, title = 'fig'):
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(self.X, self.Y, label='Data Time Series')
        ax.plot(self.X, self.mtrend, label='Trend')
        ax.plot(self.X, self.seasonal, label='Seasonal Component')
        ax.legend(loc='upper left', frameon=False)
        ax.set_title('Decomposition of the Time Series')
        if save: 
            fig1.savefig(title + '.eps')
        if option == 'plot components':
            fig2, ax = plt.subplots(2, 2) 
            ax[0][0].plot(self.Y, label='Data Time Series')
            ax[0][0].legend(loc='upper left', frameon=False)
            ax[0][1].plot(self.mtrend, label='Mean Trend')
            ax[0][1].legend(loc='upper left', frameon=False)
            ax[1][0].plot(self.seasonal, label='Seasonal Component')
            ax[1][0].legend(loc='upper left', frameon=False)
            ax[1][1].plot(self.residual, label='Residual')
            ax[1][1].legend(loc='upper left', frameon=False)
            fig2.savefig(title + '-Components.eps')
        plt.show()

    def bspline_resid(self, params, option = 'resid'):
        # -> 2020-10-28T155727
        """
        Find the residual for given parameter value alpha, beta, c
        """
        Y = self.Y
        J = self.J
        X = self.X
        v = list(range(J))
        tt = np.arange(len(Y)) % J
        dd = np.array([[(i == j)*1.0 for j in v] for i in tt])
        B = BSpline(X, Y, 3).basis_function_evaluate()
        # c = np.ones(B.shape[1])
        # beta= np.ones(J)
        # alpha = np.zeros(J)
        '''This is important, params contains J-1 alphas, J-1 betas and the rest are c'''
        alpha = np.append(params[0:J-1], (0-params[0:J-1].sum()))
        beta = np.append(params[J-1: 2*J-2], J - params[J-1: 2*J-2].sum())
        c = params[2*J-2:]
        mtrend = B@c
        f = np.diag(B@c)@dd
        seasonal = dd @ alpha + f @ beta - mtrend
        epsilon = Y - dd @ alpha - f @ beta
        # loss = np.inner(epsilon, epsilon)
        if option == 'resid':
            return epsilon
        elif option == 'full':
            return mtrend, seasonal, epsilon

# %%
