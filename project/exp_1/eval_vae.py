from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
#test = DataLoading('../../data_kmc/2d/', 49, 1000, 50, 0, 100, "local_density_li+.txt")

n = 10

x_test =  torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32) #  ('../../data_kmc/2d/677_3e+18_2.0/avg_trj.npy'), dtype=torch.float64)
x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
#x_train = torch.reshape(x_train,(x_train.shape[0]*x_train.shape[1],1,50,100))
train_params = get_config("../../data_kmc/2d_sets/train_set_80_20_list.txt")
params = get_config("../../data_kmc/2d_sets/test_set_80_20_list.txt")
params_scale = torch.mean(train_params, dim=0)
params = params/params_scale
params = torch.repeat_interleave(train_params[n].unsqueeze(0), 1000, dim=0)

x_train_mean = torch.mean(x_train)

x_test = x_test/x_train_mean

latent_dim = 50
base = 8
part_class = 1  
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim, mode='train')
vae.load_state_dict(torch.load('model_vae_500_act.pth', map_location=torch.device('cpu')))
vae.eval()
#data = torch.tensor(np.load('../../data_kmc/2d_sets/2d_red_5.npy'), dtype=torch.float32)
#print("Mean of data", data.mean(), "Std of data", data.std())

x_test = (x_train[n])#.permute(0,2,1,3)
print("x_test shape:", x_test.shape)
print("params shape:", params.shape)
x_avg = torch.mean(x_test, dim=2)


time = np.logspace(-8, -4, 1000, 'o')
labels = [] 

t_end = 1000
t_space = 10
num_plots=2
fig,axs = plt.subplots(1,num_plots, figsize=(15,5))
x_pred,_,_,_ = vae(x_test, params)
print("x_pred shape:", x_pred.shape)
x_pred = torch.mean(x_pred, dim=2)


for i in range(500, t_end,t_space):
    
    axs[0].plot(np.linspace(0,100,100), x_avg[i, 0,:])
    #labels.append([f"t = {time[i-1]}"])
    axs[0].set_ylim(0, 3.0) 
    axs[0].set_title("True")
    axs[0].legend(labels)

    axs[1].plot(np.linspace(0,100,100), x_pred[i, 0,:].detach().numpy())
    #labels.append([f"t = {time[i-1]}"])
    axs[1].set_ylim(0, 3.0)
    axs[1].set_title("Predicted")
    axs[1].legend(labels)
#plt.legend(labels)
plt.tight_layout()
plt.savefig("vae_500"+str(n)+".png")

#plt.savefig("vae_labels.png")



#print("before mean",x_pred[50,0, 5,:])



    
plt.figure()

plt.plot(np.linspace(0,100,100), x_avg[50, 0,:])
plt.plot(np.linspace(0,100,100), x_pred[50, 0,:].detach().numpy())
plt.legend(["True", "Predicted"])
plt.savefig("vae_params.png")
print(sum(p.numel() for p in vae.parameters() if p.requires_grad))