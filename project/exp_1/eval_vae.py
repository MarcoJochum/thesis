from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
#test = DataLoading('../../data_kmc/2d/', 49, 1000, 50, 0, 100, "local_density_li+.txt")

x_test = torch.tensor(np.load('../../data_kmc/2d/677_3e+18_2.0/avg_trj.npy'), dtype=torch.float64)
latent_dim = 40
base = 32
part_class = 1  
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
cae = VAE(encoder, decoder, latent_dim=latent_dim)
cae.double()    
cae.load_state_dict(torch.load('model_vae_kmc_red_data_e1000.pth'))
cae.eval()
data = torch.tensor(np.load('../../data_kmc/2d_sets/2d_red_5.npy'), dtype=torch.float32)
print("Mean of data", data.mean(), "Std of data", data.std())

x_test = x_test.permute(0,2,1,3)
x_avg = torch.mean(x_test, dim=2)


time = np.logspace(-8, -4, 1000, 'o')
labels = [] 

t_end = 500
t_space = 100
num_plots=2
fig,axs = plt.subplots(1,num_plots, figsize=(15,5))
x_pred,_,_,_ = cae(x_test)
x_pred = torch.mean(x_pred, dim=2)
for i in range(1, t_end,t_space):
    
    axs[0].plot(np.linspace(0,100,100), x_avg[i, 0,:])
    labels.append([f"t = {time[i-1]}"])
    #axs[0].set_ylim(0, 5) 
    axs[0].set_title("True")

    axs[1].plot(np.linspace(0,100,100), x_pred[i, 0,:].detach().numpy())
    labels.append([f"t = {time[i-1]}"])
    #axs[1].set_ylim(0, 5)
    axs[1].set_title("Predicted")
plt.legend(labels)
plt.tight_layout()
plt.savefig("vae_compare.png")

#plt.savefig("vae_labels.png")



#print("before mean",x_pred[50,0, 5,:])



    
plt.figure()

plt.plot(np.linspace(0,100,100), x_avg[700, 0,:])
plt.plot(np.linspace(0,100,100), x_pred[700, 0,:].detach().numpy())
plt.legend(["True", "Predicted"])
plt.savefig("vae_compare_1plot.png")
print(sum(p.numel() for p in cae.parameters() if p.requires_grad))