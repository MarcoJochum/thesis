import torch
import numpy as np  
from .vae import VAE_config
class Tft_config():

    data_train = torch.tensor(np.load("../../data_kmc/2d_sets/train_set_lin_80_20.npy"), dtype=torch.float32)
    data_test = torch.tensor(np.load("../../data_kmc/2d_sets/test_set_lin_80_20.npy"), dtype=torch.float32)
    train_params = VAE_config.train_params
    test_params = VAE_config.test_params
    n_steps = 300
    inference_steps = 500
    n_samples_inference = 100
    model_name = "tft"
    
    #VAE parameters
    VAE_path = "models/model_vae_lin.pth"
    latent_dim = VAE_config.latent_dim
    base = VAE_config.base
    part_class = VAE_config.part_class

    
    seq_length = 5
    pred_length = 45
    # default quantiles for QuantileRegression
    quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    
    lstm_layers = 1
    hidden = 64
    num_attention_heads = 4
    dropout = 0.1
    batch_size = 500
    n_epochs = 50
