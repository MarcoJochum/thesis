from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
import torch
import matplotlib.pyplot as plt 
from config.vae import VAE_config
from lib.helper import *

from config.seq import Seq_config

num_configs = 8

seq_len = Seq_config.seq_length
latent_dim = Seq_config.latent_dim
base = Seq_config.base
part_class = Seq_config.part_class


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim)
vae.load_state_dict(torch.load("models/model_vae_lin.pth", map_location=torch.device("cpu")))
vae.eval()


seq2seq = torch.load("model_seq2seq.pth", map_location=torch.device("cpu"))


x_train = Seq_config.data_train
x_test = Seq_config.data_test
x_train = x_train/torch.mean(x_train)
x_test = x_test/torch.mean(x_train)
y_train = x_train
test_params = Seq_config.test_params    
train_params = Seq_config.train_params##epsr, cbulk, v_bias

##Rescale parameters with the mean
params_scale = torch.mean(train_params, dim=0)
train_params = train_params/params_scale
train_params_non_stack = train_params

test_params = test_params/params_scale
test_params_non_stack = test_params

n_configs = x_train.shape[0]
n_configs_test = x_test.shape[0]    
n_time = x_train.shape[1]
x_train = reshape_vae(x_train)
x_test = reshape_vae(x_test)
with torch.no_grad():
    x_train = vae(x_train)[2]
    x_test = vae(x_test)[2]

x_train= unshape_vae(x_train, n_configs, n_time, lat=True)
print(x_train.shape)


t = 150
##seq_len  
test_trj = y_train[:num_configs]
test_trj_lat = x_train[:num_configs]
with open("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names[:num_configs]  

y_pred_lat = torch.zeros((num_configs,t+seq_len+1,latent_dim))
y_pred_lat[:,:seq_len] = test_trj_lat[:,:seq_len]

for i in range(seq_len,seq_len+t):
    with torch.no_grad():
        y_pred = seq2seq.evaluate(y_pred_lat[:,i-seq_len:i])
        y_pred_lat[:,i] = y_pred.squeeze()  
    


y_pred = reshape_vae(y_pred_lat)
y_pred = vae.decoder(y_pred)

y_pred = unshape_vae(y_pred, num_configs, t+seq_len+1, False)
time = np.logspace(-8, -4, 1000, 'o')

test_trj = torch.mean(test_trj.squeeze(), dim=2)
y_pred = torch.mean(y_pred, dim=2).squeeze()


labels = [] 
with torch.no_grad():
      
    fig,axs = plt.subplots(num_configs,2, figsize=(6,20))
    for j in range(num_configs):

        for i in range(10, 50,1):
            
           
            axs[j,0].plot(np.linspace(0,100,100), y_pred[j, i].numpy())
            labels.append([f"t = {time[i]}"])
            axs[j,0].set_ylim(0, 5)
            axs[j,0].set_title("Prediction"+ names[j])
            #axs[0].legend(labels)

            axs[j,1].plot(np.linspace(0,100,100), test_trj[j,i].numpy())
            #labels.append([f"t = {time[i-1]}"])
            axs[j,1].set_ylim(0, 5)
            axs[j,1].set_title("Ground truth"+ names[j])
            #axs[1].legend(labels)


 
plt.savefig("seq2seq"+str(num_configs)+".png")
print("Number of parameters in lstm:",sum(p.numel() for p in seq2seq.parameters() if p.requires_grad))