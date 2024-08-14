import numpy as np
import torch
from .vae import VAE_config
class Seq_config:


    batch_size = 20
    lr = 0.005
    n_epochs = 100


    latent_dim = VAE_config.latent_dim
    base = VAE_config.base
    part_class = VAE_config.part_class

    hidden = 50
    num_layers = 1
    seq_length = 10 
    pred_length = 30

    data_train = torch.tensor(np.load("../../data_kmc/2d_sets/train_set_lin_80_20.npy"), dtype=torch.float32)
    data_test = torch.tensor(np.load("../../data_kmc/2d_sets/test_set_lin_80_20.npy"), dtype=torch.float32)
    train_params = VAE_config.train_params
    test_params = VAE_config.test_params
    n_steps = 200
    model_name = "model_seq2seq.pth"