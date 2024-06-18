

import pandas as pd
import numpy as np
from itertools import chain
import torch
  # Added missing import statement

def normalization(a):
    a = (a) #added this to convert everything to torch
    return (a - a.min(dim = 0).values)/(a.max(dim = 0).values- a.min(dim = 0).values), a.max(dim = 0).values, a.min(dim = 0).values

def normalization_with_inputs(a, amax, amin):
    return (a - amin)/(amax - amin)

def reverse_normalization(a,amax,amin):
    a = (a) #added this to convert everything to torch

    return a*(amax-amin) + amin

def data_processing(df, name):
    temp = df[name]
    temp = torch.tensor(temp)
    temp = torch.reshape(temp, (temp.shape[0],1))
    return temp

def flat(list_2D):
    flatten_list = list(chain.from_iterable(list_2D))
    flatten_list = torch.tensor(flatten_list)
    return flatten_list
