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
    num_epochs = 500
    lr = 0.001
    KLD_weight =1.0
    
    latent_dim = 25
    base = 4
    n_steps= 500
    #Data
    data_train_avg = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_lin_80_20_avg.npy'), dtype=torch.float32)
    data_test_avg = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_lin_80_20_avg.npy'), dtype=torch.float32)

    data_train_std = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_lin_80_20_std.npy'), dtype=torch.float32)
    data_test_std = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_lin_80_20_std.npy'), dtype=torch.float32)
    
    model_name = 'model_vae_lin.pth'