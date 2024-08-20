

import pandas as pd
import numpy as np
from itertools import chain
import torch
  # Added missing import statement

def normalization(a):
    return (a - a.min())/(a.max()- a.min()), a.max(), a.min()

def normalization_with_inputs(a, amax, amin):
    return (a - amin)/(amax - amin)

def reverse_normalization(a,amax,amin):
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


def reshape_vae(x):
    if len(x.shape) == 5:
        x = torch.reshape(x, (x.shape[0]* x.shape[1],  1, 50, 100))
    elif len(x.shape) == 6:
        x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2], 1, 50, 100))
    elif len(x.shape) == 3:
        x = torch.reshape(x, (x.shape[0]*x.shape[1], -1))
    else:
        print("Invalid shape:", x.shape)
        return None
    return x

def unshape_vae(x, n_configs, n_time, lat):
    if lat:
        x = torch.reshape(x, (n_configs, n_time, -1))
    else:
        x = torch.reshape(x, (n_configs, n_time, 50,100))
    return x