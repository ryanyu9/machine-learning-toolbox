# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:55:43 2017

@author: ryanyu

Define a class, used to solve Linear Regression with L2 Regularization (Ridge Regression).
It uses BFGS method by providing gradients vector.
An example followed.
"""

import numpy as np
from scipy.optimize import minimize
from __future__ import division

class Linear_grad(object):
    """
    Create a class, which is used to solve ridge regression problems.
    """
    
    def __init__(self, X, y, alpha):
        """    
        Parameters:
        X: numpy array including predictor features. May need pre-processing.
        y: 1D numpy array including target value.
        alpha: a non-negative real number, regularization coefficient.
               when it's 0, there's no regularization.        
        """
        
        self.X = X
        self.y = y
        self.alpha = alpha
    
    def cost(self, theta):
        """        
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1]
        
        Return:
        A real number, the cost of linear regression with L2 regularization.
        """
        
        return np.sum((self.X.dot(theta) - self.y)**2.0) / (2.0*self.X.shape[0]) + \
                        np.sum(theta[1:]**2)*self.alpha / (2.0*self.X.shape[0])

    def grad(self, theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A 1D numpy array with same length of theta, gradient of theta in the 
        cost function.
        """
        
        grad = np.zeros_like(theta)
        grad = self.X.T.dot(self.X.dot(theta) - self.y) / self.X.shape[0]
        grad[1:] += theta[1:]*self.alpha / self.X.shape[0]
        return grad

    def fit(self, method='BFGS', rand_state=0):
        """
        This method is used to fit the model.
        
        Parameters:
        method: fitting method. The default is BFGS.
        rand_state: an integer indicating the seed to generate the initial vector. 
        """
        
        np.random.seed(rand_state)
        res = minimize(self.cost, x0=np.random.randn(self.X.shape[1]), 
                        method=method, jac=self.grad, options={'disp': True})
        self.theta = res.x

    def predict(self, newX):
        """        
        Parameters:
        newX: new data, same format as X.
        
        Return:
        Predicted values.
        """

        return newX.dot(self.theta)

# an example
rg = Linear_grad(X_train, y_train, 0.1) # create an object by feeding traing data
rg.fit() # fit the model
y = rg.predict(X_train) # make predictions