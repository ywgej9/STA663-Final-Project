# -*- coding: utf-8 -*-
"""
Created on Wed May  1 00:20:55 2019

@author: yc
"""
import numpy as np
import pandas as pd
import scipy.stats
import scipy.linalg
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the diabetes dataset
d = datasets.load_diabetes()
data = pd.DataFrame(data= np.c_[d['data']],
                     columns= d['feature_names'])
data = data.iloc[:, 0:4]
data.iloc[0:3,:]

# Load the diabetes dataset
d = datasets.load_diabetes()
X = d['data'][:,0:4]
y = d['target']
n, p = X.shape

# Compute the OLS variance
reg = LinearRegression().fit(X, y)
y_hat = reg.predict(X)
sig2 = sum((y - y_hat)**2)/(n - p)
sig = np.sqrt(sig2)
sig2 = sum((y-y_hat)**2)/(n-p)
theta_p = reg.coef_
sig_p = np.eye(p)*500

# Test for HMC
d = X.shape[1]
theta_0 = reg.coef_.copy()
eps = 0.01
niter = 10000
M = np.eye(d)
m = 3

Theta_hmc, R, a = hmc(X, y, theta_0, eps, niter, m, M, U, grad_U, sig, theta_p, sig_p)

#%timeit hmc(X, y, theta_0, eps, niter, m, M, U, grad_U, sig, theta_p, sig_p)
#%timeit hmc_numba(X, y, theta_0, eps, niter, m, M, U, grad_U, sig, theta_p, sig_p)

# Test for SG-HMC
eps = 1.5
niter = 10000
M = np.eye(d)
m = 7
perc = 0.5
B_hat = 0 # should be a matrix here
C = np.eye(d)*0.01

Theta_sghmc, R= sghmc(X, y, theta_0, eps, niter, m, M, U, grad_U, sig, theta_p, sig_p, perc, C, B_hat)

sig_p_hmc = scipy.linalg.inv(sig_p)
cov_pos_hmc = scipy.linalg.inv(sig_p_hmc+X.T@X/sig2)
mu_pos_hmc = cov_pos_hmc @ (sig_p_hmc@theta_0+X.T @ y/sig2)
beta_pos_hmc = np.random.multivariate_normal(mu_pos_hmc, cov_pos_hmc, 50000)
sns.distplot(beta_pos_hmc[:,1], label="Analytic")
sns.distplot(Theta_hmc[1, :], label="HMC")
sns.distplot(Theta_sghmc[1, :], label="SGHMC")
plt.legend(labels=['Analytic', 'HMC', 'SGHMC'])
plt.savefig("real_plot")
pass

