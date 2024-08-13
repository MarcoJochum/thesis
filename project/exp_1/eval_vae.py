from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
#test = DataLoading('../../data_kmc/2d/', 49, 1000, 50, 0, 100, "local_density_li+.txt")
from config.vae import VAE_config
n = 5

x_test = VAE_config.data_test
x_train = VAE_config.data_train 
#x_train = torch.reshape(x_train,(x_train.shape[0]*x_train.shape[1],1,50,100))
train_params = VAE_config.train_params
test_params = VAE_config.test_params
params_scale = torch.mean(train_params, dim=0)
test_params = test_params/params_scale
train_params = torch.repeat_interleave(train_params[n].unsqueeze(0), 1000, dim=0)

x_train_mean = torch.mean(x_train)

x_test = x_test/x_train_mean

latent_dim = VAE_config.latent_dim
base = VAE_config.base
part_class = VAE_config.part_class  
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim)
vae.load_state_dict(torch.load('models/model_vae_lin.pth', map_location=torch.device('cpu')))
vae.eval()
#data = torch.tensor(np.load('../../data_kmc/2d_sets/2d_red_5.npy'), dtype=torch.float32)
#print("Mean of data", data.mean(), "Std of data", data.std())
with open("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
x_test = (x_test[8:16])
names = names[8:16]##these two have to match for correct labeling in plot
print("train params shape:", train_params.shape)
x_test_std = torch.std(x_test, dim=3).squeeze()
x_avg = torch.mean(x_test, dim=3).squeeze()

print("x_avg std shape:", x_test_std.shape)

time = np.logspace(-8, -4, 1000, 'o')
labels = [] 

t_end = 50
t_space = 1
num_plots=8
fig,axs = plt.subplots(1,num_plots, figsize=(15,5))
x_test = torch.reshape(x_test, (x_test.shape[0]*x_test.shape[1],1,50,100))
with torch.no_grad():
    x_pred,_,_,_ = vae(x_test)
x_pred = torch.reshape(x_pred,(num_plots, 1000, 50, 100))

x_pred_std = torch.std(x_pred, dim=2)
x_pred = torch.mean(x_pred, dim=2)

fig,axs = plt.subplots(num_plots,2, figsize=(6,20))
z_width = np.linspace(0,100,100)
for j in range(0, num_plots):
          
    	for i in range(0, t_end,t_space):
    
            axs[j,1].plot(z_width, x_pred[j,i,:].numpy())
            axs[j, 1].fill_between(z_width, x_pred[j,i,:].numpy() - x_pred_std[j,i,:].numpy(), x_pred[j,i,:].numpy() + x_pred_std[j,i,:].numpy(), alpha=0.3)
            #labels.append([f"t = {time[i-1]}"])
            axs[j,1].set_ylim(0, 3.0)
            axs[j,1].set_title("Predicted"+ names[j])
            axs[j,1].legend(labels)

            axs[j,0].plot(z_width, x_avg[j,i, :])
            axs[j,0].fill_between(z_width, x_avg[j,i, :] - x_test_std[j,i,:], x_avg[j,i, :] + x_test_std[j,i,:], alpha=0.3)
            #labels.append([f"t = {time[i-1]}"])
            axs[j,0].set_ylim(0, 3.0) 
            axs[j,0].set_title("True" + names[j])
            axs[j,0].legend(labels)

plt.tight_layout()
plt.savefig("std_test"+str(n)+".png")

#plt.savefig("vae_labels.png")



#print("before mean",x_pred[50,0, 5,:])



    
plt.figure()

# plt.plot(np.linspace(0,100,100), x_avg[50, 0,:])
# plt.plot(np.linspace(0,100,100), x_pred[50, 0,:].detach().numpy())
# plt.legend(["True", "Predicted"])
# plt.savefig("vae_params.png")
print(sum(p.numel() for p in vae.parameters() if p.requires_grad))