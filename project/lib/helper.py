

import pandas as pd
import numpy as np
from itertools import chain
import torch
from pytorch_lightning.callbacks import Callback
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

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def s_mape(A, F):

    A = np.array(A)
    F = np.array(F)
    
    # Avoid division by zero
    denominator = np.abs(A) + np.abs(F)
    denominator[denominator == 0] = 1e-10  
    return 100*np.mean((2 * np.abs(F - A) / denominator))

class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))

