import torch
import numpy as np
from lib.data_loading import *


class Comb_config:

    
    #VAE parameters
    latent_dim = 25
    base = 4
    part_class = 1
    #LNN parameters
    units = 60
    backbone_layers = 3
    backbone_units = 20
    backbone_dropout = 0.1
    #Training parameters
    KLD_weight =0.001
    L_sup_weight = 0.01
    lr = 0.001
    n_epochs = 50
    batch_size = 10
    
    pretrained = False
    VAE_model_pretrained = "models/model_vae_lin.pth"

    data_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_lin_80_20.npy'), dtype=torch.float32)
    data_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_lin_80_20.npy'), dtype=torch.float32)

    train_params = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt")

    test_params = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_list.txt")

    model_name = "model_comb_lin.pth"