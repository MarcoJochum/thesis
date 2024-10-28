import torch
from config.vae import VAE_config
import numpy as np
import matplotlib.pyplot as plt

with open("../../data_kmc/2d_sets/train_set_lin_80_20_std_list.txt", "r") as f:
    names= f.readlines()


names = [line.strip() for line in names]
vae = VAE_config
train_std = VAE_config.data_train_std
train_avg = VAE_config.data_train_avg
test_std = VAE_config.data_test_std
test_avg = VAE_config.data_test_avg

print("number of nans", torch.sum(torch.isnan(train_std)))
print("number of nans", torch.sum(torch.isnan(train_avg)))

train_std = torch.mean(train_std.squeeze(), dim=2)
train_avg = torch.mean(train_avg.squeeze(), dim=2)
test_std = torch.mean(test_std, dim=2)
test_avg = torch.mean(test_avg, dim=2)
num_plots = 8   
fig,axs = plt.subplots(num_plots,2, figsize=(6,20))
z_width = np.linspace(0,100,100)
t_space = 5
for j in range(0, num_plots):
          
    	for i in range(0, VAE_config.n_steps,t_space):
    
            
            axs[j,0].plot(z_width, train_std[j,i, :])
            #axs[j,0].set_ylim(0, 5.0) 
            axs[j,0].set_yscale("log")
            axs[j,0].set_title("std over time" +str(names[j]))
            
            axs[j,1].plot(z_width, train_avg[j,i, :])
            #axs[j,1].set_ylim(0, 5.0)
            axs[j,1].set_title("avg over time" +str(names[j]))

plt.tight_layout()
plt.savefig("std_investigation_test"+str(1)+".png")