# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 23:11:09 2017

@author: ryanyu

Define a class, used to solve Logistic Regression with L2 Regularization.
It uses Newton-Conjugate-Gradient method by providing Hessian matrix.
An example followed.
"""

import numpy as np
from scipy.optimize import minimize
from __future__ import division

def sigmoid(x):
    """
    sigmoid function g(x)=1/(1+exp(-x))
    """
    
    return 1 / (1 + np.exp(-1*x))
    
def sigmoid_der(x):
    """
    gradient of sigmoid function g'(x)=exp(-x)/(1+exp(-x))**2
    """
    
    return np.exp(-1*x) / (1 + np.exp(-1*x))**2

class Logit_hess(object):
    """
    Create a class, which is used to solve Logistic Regression with L2 Regularization problems.
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
        A real number, the cost of logistic regression with L2 regularization.
        """
        
        return -1*np.sum(self.y*np.log(sigmoid(self.X.dot(theta))) + \
            (1 - self.y)*np.log(1 - sigmoid(self.X.dot(theta)))) / self.X.shape[0] + \
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
        grad = self.X.T.dot(sigmoid(self.X.dot(theta)) - self.y) / self.X.shape[0]
        grad[1:] += theta[1:]*self.alpha / self.X.shape[0]
        return grad

    def hess(self, theta):
        """
        Parameter:
        theta: a 1D numpy array with length equal to X.shape[1].
        
        Return:
        A numpy array, Hess matrix.
        """
        
        H = self.X.T.dot(self.X*sigmoid_der(self.X.dot(theta))[:, np.newaxis]) / self.X.shape[0]
        np.fill_diagonal(H, H.diagonal() + self.alpha / self.X.shape[0])
        return H
        
    def fit(self, method='Newton-CG', rand_state=0):
        """
        This method is used to fit the model.
        
        Parameters:
        method: fitting method. The default is Newton-CG.
        rand_state: an integer indicating the seed to generate the initial vector. 
        """
        
        np.random.seed(rand_state)
        res = minimize(self.cost, x0=np.random.randn(self.X.shape[1]), 
                        method=method, jac=self.grad, hess=self.hess, 
                        options={'disp': True})
        self.theta = res.x

    def predict(self, newX):
        """        
        Parameters:
        newX: new data, same format as X.
        
        Return:
        Predicted probabilities.
        """

        return sigmoid(newX.dot(self.theta))

# an example
lg = Logit_hess(X_train, y_train, 0.1) # create an object by feeding traing data
lg.fit() # fit the model
y = lg.predict(X_train) # make predictions: probabilities of the positive