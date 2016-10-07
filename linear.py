# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 00:14:40 2016

@author: ryanyu

Define two funcs, each solving Linear Regression with L2 Regularization (Ridge Regression):
lin_bfgs: use BFGS method by providing gradients vector.
lin_newtcg: use Newton-Conjugate-Gradient method by providing Hessian matrix.
"""

import numpy as np
from scipy.optimize import minimize
from __future__ import division


### use BFGS method by providing gradient
def lin_bfgs(X, y, a=0, rand_state=0):
    """    
    Parameters:
    X: numpy array including predictor features. May need pre-processing.
    y: 1D numpy array including target value.
    a: a non-negative real number, regularization coefficient.
       when it's 0, there's no regularization.
    rand_state: an integer indicating the seed to generate the initial vector. 
    
    Return: 
    Scipy OptimizeResult. Its attributes include x, jac, etc. Check
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    for more attributes.
    """
    
    def lin_cost(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1]
        
        Return:
        A real number, the cost of linear regression with L2 regularization.
        """
        return np.sum((X.dot(theta)-y)**2.0)/(2.0*X.shape[0]) + np.sum(theta[1:]**2)*a/(2.0*X.shape[0])
    
    def lin_cost_der(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A 1D numpy array with same length of theta, gradient of theta in the 
        linear cost function lin_cost.
        """
        grad = np.zeros_like(theta)
        grad = X.T.dot(X.dot(theta)-y)/X.shape[0]
        grad[1:] += theta[1:]*a/X.shape[0]
        return grad
        
    np.random.seed(rand_state)
    return minimize(lin_cost, x0=np.random.randn(X.shape[1]), method='BFGS', 
            jac=lin_cost_der, options={'disp': True})

#reg = lin_bfgs(X_train, y_train, 0.1)


### use Newton-Conjugate-Gradient method by providing Hessian matrix
def lin_newtcg(X, y, a=0, rand_state=0):
    """    
    Parameters:
    X: numpy array including predictor features. May need pre-processing.
    y: 1D numpy array including target value.
    a: a non-negative real number, regularization coefficient.
       when it's 0, there's no regularization.
    rand_state: an integer indicating the seed to generate the initial vector. 
    
    Return: 
    Scipy OptimizeResult. Its attributes include x, jac, etc. Check
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    for more attributes.
    """
    
    def lin_cost(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1]
        
        Return:
        A real number, the cost of linear regression with L2 regularization.
        """
        return np.sum((X.dot(theta)-y)**2.0)/(2.0*X.shape[0]) + np.sum(theta[1:]**2)*a/(2.0*X.shape[0])
    
    def lin_cost_der(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A 1D numpy array with same length of theta, gradient of theta in the 
        linear cost function lin_cost.
        """
        grad = np.zeros_like(theta)
        grad = X.T.dot(X.dot(theta)-y)/X.shape[0]
        grad[1:] += theta[1:]*a/X.shape[0]
        return grad
        
    def lin_cost_hess(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A numpy array, Hess matrix.
        """
        H = X.T.dot(X)/X.shape[0]
        np.fill_diagonal(H, H.diagonal() + a/X.shape[0])
        return H
    
    np.random.seed(rand_state)
    return minimize(lin_cost, x0=np.random.randn(X.shape[1]), method='Newton-CG', 
            jac=lin_cost_der, hess=lin_cost_hess, options={'disp': True})

# reg = lin_newtcg(X_train, y_train, 0.1)
