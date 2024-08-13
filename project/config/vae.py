import torch
import numpy as np
from lib.data_loading import *

class VAE_config:

    '''
        Configuration for the VAE model
    '''
    part_class = 1
    #Training settings
    batch_size = 4000
    num_epochs = 800
    lr = 0.001
    KLD_weight =1.0
    L_sup_weight = 0.01

    latent_dim = 25
    base = 4
    
    #Data
    data_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_lin_80_20.npy'), dtype=torch.float32)
    data_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_lin_80_20.npy'), dtype=torch.float32)

    train_params = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt")

    test_params = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_list.txt")

    model_name = 'model_vae_lin.pth'