# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 00:03:02 2016

@author: ryanyu
"""
# Note: so far this only works on category features.

from __future__ import division
import numpy as np
import pandas as pd


def entropy(x):
    """
    The entropy measures the expected uncertainty in random variable x.
    When x is discrete r.v., H(x) = - sum(p*logp)
    
    Input:
    x: a pandas series or numpy array.
    
    Return:
    entropy, a float.
    """
    if isinstance(x, np.ndarray):
        y = pd.Series(x)
    pmf = x.value_counts()/x.size
    return -1*((pmf*np.log2(pmf)).sum())


def cond_entropy(y, x):
    """
    This func calculates conditional entropy H(y|x), which measures how much 
    uncertainty remains about the random variable y when we know the value of x.
    
    Inputs:
    y must be a 1D numpy array or Series; 
    x could be 1D (Series), or multi-D (list of Series or DataFrame).
    
    Return:
    conditional entropy, a float.
    """
    # check y
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    # check x
    if isinstance(x, list):
        lst = x
    elif isinstance(x, pd.DataFrame):
        lst = [x[c] for c in x]
    else:
        lst = [x]
    lst_y = list(lst)
    lst_y.append(y)
    
    freq = y.groupby(lst_y).size()
    freq.name = 'freq'
    # get joint probability P(X=x,Y=y)
    Pxy = freq/len(y)
    freq_x = y.groupby(lst).size()
    freq_x.name = 'freq_x'
    # get conditional probability estimate P(Y=y|X=x)
    merged = pd.merge(freq.reset_index(), freq_x.reset_index())
    Py_x = merged['freq']/merged['freq_x']
    Py_x.index = freq.index
    P = Pxy * np.log2(Py_x)
    return max(0.0, -1 * P.sum())


def mutual_info(y, x):
    """
    This func calculates mutual information: I(y;x), which is the uncertainty 
    shared by x and y.
    
    Inputs:
    y must be a 1D numpy array or Series; 
    x could be 1D (Series) or multi-D (list of Series or DataFrame).
    When x is multi-D, it calculates I(y;x1, x2,..., xn)
    
    Return:
    mutual information, a float.
    
    Note: When x is 1D, mutual_info(y,x) == mutual_info(x,y) because 
    I(y;x)=H(y)-H(y|x)=H(x)-H(x|y)
    """
    print entropy(y)
    print cond_entropy(y, x)
    return entropy(y) - cond_entropy(y, x)