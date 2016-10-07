# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 00:14:40 2016

@author: ryanyu

Define two funcs, each solving Logistic Regression with L2 Regularization:
logit_bfgs: use BFGS method by providing gradients vector.
logit_newtcg: use Newton-Conjugate-Gradient method by providing Hessian matrix.
"""

import numpy as np
from scipy.optimize import minimize
from __future__ import division


def sigmoid(x):
    """
    sigmoid function g(x)=1/(1+exp(-x))
    """
    return 1/(1+np.exp(-1*x))
    
def sigmoid_der(x):
    """
    gradient of sigmoid function g'(x)=exp(-x)/(1+exp(-x))**2
    """
    return np.exp(-1*x)/(1+np.exp(-1*x))**2
    
### use BFGS method by providing gradient 
def logit_bfgs(X, y, a=0.0, rand_state=0):
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
        
    def logit_cost(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1]
        
        Return:
        A real number, the cost of logistic regression with L2 regularization.
        """
        return -1*np.sum(y*np.log(sigmoid(X.dot(theta)))+(1-y)*np.log(1-sigmoid(X.dot(theta))))/X.shape[0]+np.sum(theta[1:]**2)*a/(2.0*X.shape[0])
    
    def logit_cost_der(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A 1D numpy array with same length of theta, gradient of theta in the 
        logistic cost function logit_cost.
        """
        grad=np.zeros_like(theta)
        grad=X.T.dot(sigmoid(X.dot(theta))-y)/X.shape[0]
        grad[1:] += theta[1:]*a/X.shape[0]
        return grad
        
    np.random.seed(rand_state)
    return minimize(logit_cost, x0=np.random.randn(X.shape[1]), method='BFGS',
            jac=logit_cost_der, options={'disp': True})

# clf = logit_bfgs(X_train, y_train)


### use Newton-Conjugate-Gradient method by providing Hessian matrix
def logit_newtcg(X, y, a=0.0, rand_state=0):
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
    
    def logit_cost(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1]
        
        Return:
        A real number, the cost of logistic regression with L2 regularization.
        """
        return -1*np.sum(y*np.log(sigmoid(X.dot(theta)))+(1-y)*np.log(1-sigmoid(X.dot(theta))))/X.shape[0]+np.sum(theta[1:]**2)*a/(2.0*X.shape[0])
    
    def logit_cost_der(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A 1D numpy array with same length of theta, gradient of theta in the 
        logistic cost function logit_cost.
        """
        grad=np.zeros_like(theta)
        grad=X.T.dot(sigmoid(X.dot(theta))-y)/X.shape[0]
        grad[1:] += theta[1:]*a/X.shape[0]
        return grad
    
    def logit_cost_hess(theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A numpy array, Hess matrix.
        """        
        H=X.T.dot(X*sigmoid_der(X.dot(theta))[:, np.newaxis])/X.shape[0]
        np.fill_diagonal(H, H.diagonal() + a/X.shape[0])
        return H
        
    np.random.seed(rand_state)
    return minimize(logit_cost, x0=np.random.randn(X.shape[1]), method='Newton-CG',
            jac=logit_cost_der, hess=logit_cost_hess, options={'disp': True})

# clf = logit_newtcg(X_train, y_train, 0.5)
