# -*- coding: utf-8 -*-
"""
Created on Wed May  1 00:13:25 2019

@author: yc
"""

import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns

# data simulation
mu_1 = 0
sigma_1 = 1
mu_2 = 5
sigma_2 = 3
mu_3 = 7
sigma_3 = 3
n = 100
x_1 = np.random.normal(mu_1, sigma_1, n)
x_2 = np.random.normal(mu_2, sigma_2, n)
x_3 = np.random.normal(mu_3, sigma_3, n)
noise = np.random.normal(0, 5, n)
b1 = 5
b2 = 3
b3 = -2

y_sim = b1*x_1 + b2*x_2 + b3*x_3+noise
X_sim = np.array([x_1, x_2, x_3]).T

d = X_sim.shape[1]
theta_0_sim = np.array([3.0, 2.0, -1.0])
eps = 0.09
niter = 10000
M = np.eye(d)
m = 3
p_sim = 3
sig_sim = 5
sig_sim2 = sig_sim**2
sig_p_sim = np.diag([10.0, 10.0, 10.0])
theta_p_sim = np.array([0, 0, 0])
Theta_sim, R_sim, a_sim = hmc(X_sim, y_sim, theta_0_sim, eps, niter, m, M, U, grad_U, sig_sim, theta_p_sim, sig_p_sim)

## Analytic solution
sig_p_sim_inv = scipy.linalg.inv(sig_p_sim)
cov_sim_pos = scipy.linalg.inv(sig_p_sim_inv+X_sim.T@X_sim/sig_sim2)
mu_sim_pos = cov_sim_pos@(sig_p_sim_inv@theta_0_sim+X_sim.T@y_sim/sig_sim2)

d = X_sim.shape[1]
theta_0_sim = np.array([3.0, 2.0, -1.0])
eps = 0.09
niter = 10000
M = np.eye(d)
m = 3
p_sim = 3
sig_sim = 5
sig_sim2 = sig_sim**2
sig_p_sim = np.diag([10.0, 10.0, 10.0])
theta_p_sim = np.array([0, 0, 0])
perc = 0.3
B_hat = 0 # should be a matrix here
C = np.eye(d)
Theta_sghmc,_ = sghmc(X_sim, y_sim, theta_0_sim, eps, niter, m, M, U, grad_U, sig_sim, theta_p_sim, sig_p_sim, perc, C, B_hat)

beta_sim_pos = np.random.multivariate_normal(mu_sim_pos, cov_sim_pos, 50000)
sns.distplot(beta_sim_pos[:,1], label="Analytic")
sns.distplot(Theta_sim[1, :], label="HMC")
sns.distplot(Theta_sghmc[1, :], label="SGHMC")
plt.legend(labels=['Analytic', 'HMC', 'SGHMC'])
plt.savefig("simulated_dist_plot")
pass