import torch
import numpy as np  
from .vae import VAE_config
from lib.data_loading import get_config
class Tft_config():

    data_train_avg = torch.tensor(np.load("../../data_kmc/2d_sets/train_set_lin_80_20_avg.npy"), dtype=torch.float32)
    data_test_avg = torch.tensor(np.load("../../data_kmc/2d_sets/test_set_lin_80_20_avg.npy"), dtype=torch.float32)
    data_train_std = torch.tensor(np.load("../../data_kmc/2d_sets/train_set_lin_80_20_std.npy"), dtype=torch.float32)
    data_test_std = torch.tensor(np.load("../../data_kmc/2d_sets/test_set_lin_80_20_std.npy"), dtype=torch.float32)
    train_params_avg = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_avg_list.txt")
    test_params_avg = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_avg_list.txt")
    train_params_std = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_std_list.txt")
    test_params_std = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_std_list.txt")
    n_steps = 500
    inference_steps = 995
    n_samples_inference = 100
    model_name = "tft"
    
    #VAE parameters
    VAE_path = "models/model_vae_lin.pth"
    latent_dim = VAE_config.latent_dim
    base = VAE_config.base
    part_class = VAE_config.part_class

    
    seq_length = 5
    pred_length = 62
    # default quantiles for QuantileRegression
    quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    lr = 0.0008
    lstm_layers = 2
    hidden = 96
    num_attention_heads = 1
    dropout = 0.02
    batch_size = 500
    n_epochs = 50
    train_size = 74 #train validation split
