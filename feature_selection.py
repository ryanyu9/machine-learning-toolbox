# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 00:10:31 2016

@author: ryanyu
"""

from __future__ import division
import numpy as np
import pandas as pd


def feature_selection_jmi(y, df, k=None):
    """
    This func selects the first k features from df to minimize conditional entropy(CE) of y.
    If CE of y based on less than k features reaches to 0 or stop decreasing, the selection
    process is early terminated.
    
    Inputs:
    y: usually the target of prediction.
    df: the dataframe including a pool of features to be selected.
    k: number of features to be selected. The default value is no. of all features in df.
    
    Return:
    A panda series including conditional entropy with feature name as index.
    
    The motivation is originally based on maximization of joint mutual information (JMI), 
    this implementation calculates conditional entropy instead.
    Conditional entropy = H(y) - I(y;X1, X2, ... Xk). While
    H(y) is fixed, smaller conditional entropy corresponds to 
    larger JMI. One advantage of calculating conditional entropy is when it's 0, we know
    this is the best we can get.
    
    # Note: so far this only works on category features.
    """
    if k is None:
        k = df.shape[1] # consider all features
    print 'Selecting features...'
    lst = [] # a list including features selected
    lst_he = [] # a list including conditional entropy, cumulatively
    while len(lst) < k:
        newdf = df.drop(lst, axis=1)
        newdf_HE = newdf.apply(lambda f: cond_entropy(y, df[lst].join(f)))
        min_HE = np.min(newdf_HE)
        print (np.argmin(newdf_HE), min_HE)
        if len(lst) == 0 or (len(lst) > 0 and min_HE < lst_he[-1]):
            lst.append(np.argmin(newdf_HE))
            lst_he.append(min_HE)
        elif len(lst) > 0 and min_HE == lst_he[-1]:
            print ("Conditional entropy cannot be decreased")
            break
        if min_HE == 0:
            print ("Conditional entropy is 0 now")
            break
    return pd.Series(lst_he, index=lst)