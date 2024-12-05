from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
#test = DataLoading('../../data_kmc/2d/', 49, 1000, 50, 0, 100, "local_density_li+.txt")
from config.vae import VAE_config


x_test = VAE_config.data_test
x_train = VAE_config.data_train 
x_train_mean = torch.mean(x_train)
x_test = x_test/x_train_mean

latent_dim = VAE_config.latent_dim
base = VAE_config.base
part_class = VAE_config.part_class  
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim)
vae = (torch.load('models/model_vae_lin_lat_10.pth', map_location=torch.device('cpu')))
vae.eval()
#data = torch.tensor(np.load('../../data_kmc/2d_sets/2d_red_5.npy'), dtype=torch.float32)
#print("Mean of data", data.mean(), "Std of data", data.std())
with open("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
x_test = (x_test[8:16])
names = names[8:16]##these two have to match for correct labeling in plot

x_test_std = torch.std(x_test, dim=3).squeeze()
x_avg = torch.mean(x_test, dim=3).squeeze()

