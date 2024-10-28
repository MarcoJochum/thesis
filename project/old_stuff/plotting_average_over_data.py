from NNs.autoencoder import *
from NNs.ae_layers import * 
from lib.train import *
from NNs.RNN import *
from lib.data import *
from lib.data_loading import *
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from config.vae import VAE_config
data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"
import random
random.seed(42)
n_prod = 50
n_x = 50
n_y = 1
n_z= 100
part_class = VAE_config.part_class
latent_dim = VAE_config.latent_dim  
base = VAE_config.base

## Creating datasets for the LSTM model


data = np.load("../../data_kmc/2d_sets/train_set_lin_80_20.npy",)

data = np.reshape(data, (data.shape[0],data.shape[1],50,100))
 
data = data[:,:500]
data = np.reshape(data, (data.shape[0]*data.shape[1],1,50,100))

encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)

vae = VAE(encoder, decoder, latent_dim=latent_dim)
data_mean_decoded = vae(torch.tensor(data, dtype=torch.float32))[0].detach().numpy().squeeze()
data_mean_decoded = np.reshape(data_mean_decoded, (78,500,50,100))
print("data_mean_decoded", data_mean_decoded.shape) 
data_mean_decoded = np.mean(data_mean_decoded, axis=2)
data_mean_decoded = np.mean(data_mean_decoded, axis=0)
print("data_mean_decoded", data_mean_decoded.shape) 
print(data.shape)


z = np.linspace(0,100,100)
time = np.linspace(1e-07,1e-04,1000)
for i in range(0, 100,10):
            
        
            
            plt.plot(np.linspace(0,100,100), data_mean_decoded[i])
            #plt.labels.append([f"t = {time[i]}"])
            plt.ylim(0, 5)
            
            #axs[0].legend(labels)
plt.savefig("average_allconfigs.png")