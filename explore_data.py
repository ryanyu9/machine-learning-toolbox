# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:58:28 2016

@author: ryanyu
"""

from __future__ import division
import numpy as np
import pandas as pd


def explore_data(df):
    # Purpose: get general information of all features in a dataframe
    # df is a DataFrame
    # return a DataFrame object with dimension of ((no. of features), 3)
    # the index is feature, and the columns are feature type, no. of unique values
    # and no. of missing values
    return pd.DataFrame({'type':df.dtypes,
                         'uni_val_len':df.apply(lambda x: x.unique().size),
                         'nan_count':df.apply(lambda x: x.isnull().sum())},
                        columns=['type', 'uni_val_len', 'nan_count'])

