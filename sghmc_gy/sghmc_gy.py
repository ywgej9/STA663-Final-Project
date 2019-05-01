import numpy as np
import scipy.stats
import scipy.linalg


# define functions
def U(X, y, theta, theta_p, sig, sig_p):
    """Compute likelihood function U with vector"""
    return - sum(np.log(scipy.stats.norm.pdf(y, X @ theta, sig))
                 -np.log(scipy.stats.multivariate_normal.pdf(theta, mean = theta_p, cov = sig_p)))

def grad_U(X, y, theta, theta_p, sig, sig_p_inv):
    return -(y - X @ theta).T @ np.eye(X.shape[0]) / (sig**2) @ X + ((theta-theta_p) @ sig_p_inv).flatten()

def H(X, y, theta, sig, r ,U, M_inv, theta_p, sig_p_inv):
    '''Hamiltonian function of total energy'''
    return U(X, y, theta, theta_p, sig, sig_p_inv) + 0.5 * r.T @ M_inv @ r

def hmc(X, y, theta_0, eps, niter, m, M, U, grad_U, sig, theta_p, sig_p):
    """Hamiltonian Monte Carlo"""
    d = theta_0.shape[0]
    Theta = np.empty((d, niter))
    R = np.empty((d, niter))
    M_inv = scipy.linalg.inv(M)
    sig_p_inv = scipy.linalg.inv(sig_p)
    theta = theta_0.copy()
    a = 0
    for i in range(niter):
        r = np.random.multivariate_normal(np.zeros(d), M, 1).flatten()
        theta_old = theta.copy()
        r_old = r.copy()          
        #discretize Hamiltonian dynamics
        r -= eps / 2 * grad_U(X, y, theta, theta_p, sig, sig_p_inv)
        for j in range(m):
            theta += eps * M_inv @ r
            r -= eps * grad_U(X, y, theta, theta_p, sig, sig_p_inv)
        r -= eps / 2 * grad_U(X, y, theta, theta_p, sig, sig_p_inv)
        #M-H correction
        u = np.random.uniform(0, 1, 1)
        rho = np.exp(H(X, y, theta, sig, r ,U, M_inv, theta_p, sig_p_inv) -
                     H(X, y, theta_old, sig, r_old ,U, M_inv, theta_p, sig_p_inv))
        if u < rho:
            Theta[:,i] = theta
            R[:,i] = r
            a += 1
        else:
            Theta[:,i] = theta_old
            R[:,i] = r_old
        theta = Theta[:,i]        
    return Theta, R, a

def sghmc(X, y, theta_0, eps, niter, m, M, U, grad_U, sig, theta_p, sig_p, perc, C, B_hat):
    """Stochastic Gradient Hamiltonian Monte Carlo"""
    # theta_0: the starting point
    # eps: step length in solving the dynamic system
    # niter: the number sampling
    # m: the maximum number of steps to take for solving the Hamiltonian
    # M: the mass matrix in the Hamiltonian
    # perc: the percentage of minbatch size over total size
    d = theta_0.shape[0]
    batch_size = int(np.floor(X.shape[0] * perc))
    Theta = np.empty((d, niter))
    R = np.empty((d, niter))
    M_inv_eps = scipy.linalg.inv(M) * eps
    sig_p_inv = scipy.linalg.inv(sig_p)
    theta = theta_0.copy()
    M0 = C @ M_inv_eps
    V = 2*eps*(C - B_hat)
    for i in range(niter):
        r = np.random.multivariate_normal(np.zeros(d), M, 1).flatten()      
        for j in range(m):
            theta += M_inv_eps @ r.flatten()           
            idx = np.random.randint(batch_size, size = (batch_size,))
            X0 = X[idx, ] #sample X
            y0 = y[idx]
            r = r - eps * grad_U(X0, y0, theta, theta_p, sig, sig_p_inv) - M0 @ r + np.random.multivariate_normal(np.zeros(d), V, 1).flatten()        
        Theta[:,i] = theta
        R[:,i] = r
        
    return Theta, R