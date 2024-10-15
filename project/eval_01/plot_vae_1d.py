from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
#test = DataLoading('../../data_kmc/2d/', 49, 1000, 50, 0, 100, "local_density_li+.txt")
from config.vae import VAE_config
import argparse
from config.tft import Tft_config
from lib.helper import *    
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.serif'] = ['Palatino']

mpl.rcParams["axes.prop_cycle"] =mpl.cycler('color', ['black', '#003DFD', '#b512b8', '#11a9ba', '#0d780f', '#f77f07', '#ba0f0f', '#0b0bf7', '#f70bf7', '#0bf7f7', '#0bf70b', '#f70b0b', '#f7f7f7', '#7f7f7f', '#0b0b0b'])
#plt.style.use('thesis_style.mplstyle')
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()

vae= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))

if args.data_type == 'avg':
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg

elif args.data_type == 'std':
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std

data_test = data_test/ torch.mean(data_train)
data_train = data_train/ torch.mean(data_train)

data_train_decoded, data_test_decoded, data_train_encoded, data_test_encoded = get_encoded_decoded(vae, data_train, data_test)

data_test_std = torch.std(data_test, dim=3).squeeze()
data_test_decoded_std = torch.std(data_test_decoded, dim=2).squeeze()
data_test_decoded = torch.mean(data_test_decoded, dim=2).squeeze()
data_test_mean = torch.mean(data_test, dim=3).squeeze()

time = np.linspace(1e-7, 1e-4, 1000)
labels = [] 

t_end = 300
t_space = 50
num_plots=6

fig,axs = plt.subplots(2,3, figsize=(10,8))
z_width = np.linspace(0,100,100)
k=0
column_labels = ["a", "b", "c"]

config_labels = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_"+args.data_type+"_list.txt")
configs = [ 14, 9, 17]
for j in configs:
          
    for i in [10, 20, 50 , 100,  200,  300]:
    
            

            axs[0,k].plot(z_width, data_test_mean[j,i, :],label=f"t = {time[i]:.2e}")
            axs[0,k].fill_between(z_width, data_test_mean[j,i, :] - data_test_std[j,i,:], data_test_mean[j,i, :] + data_test_std[j,i,:], alpha=0.3)
            

            axs[0,k].set_ylim(0, 5.0) 
            axs[0,k].set_title("Ground truth")
            
            axs[1,k].plot(z_width, data_test_decoded[j,i,:].numpy(), label=f"t = {time[i]:.2e}")
            axs[1,k].fill_between(z_width, data_test_decoded[j,i,:].numpy() - data_test_decoded_std[j,i,:].numpy(),
                                    data_test_decoded[j,i,:].numpy() + data_test_decoded_std[j,i,:].numpy(), alpha=0.3)
            
            axs[1,k].set_ylim(0, 5.0)
            axs[1,k].set_title("Predicted")

            if k == 0:  
                
                axs[0,k].legend()
                axs[1,k].legend()
                axs[0,k].set_ylabel("Relative Concentration")
                axs[1,k].set_ylabel("Relative Concentration")
                axs[1,k].set_xlabel("Position z [nm]")
                axs[0,k].set_xlabel("Position z [nm]")
    title = "{})  $\Lambda$=[{:.0f},{:.1e},{:.2}] ".format(column_labels[k],float(config_labels[j, 0]), float(config_labels[j, 1]), float(config_labels[j, 2]))

    fig.text(0.5, -0.3, title, ha='center', fontsize=12, transform=axs[1, k].transAxes)
    k+=1
for ax in axs.flat:
    ax.grid(True, which='both')
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.subplots_adjust(wspace=0.2, hspace=0.5, bottom=0.2)

plt.savefig("fig_report/1d_vae/vae_1d_"+ str(configs[0])+str(configs[1])+str(configs[2])+".pdf", format="pdf")





    
plt.figure()

# plt.plot(np.linspace(0,100,100), x_avg[50, 0,:])
# plt.plot(np.linspace(0,100,100), x_pred[50, 0,:].detach().numpy())
# plt.legend(["True", "Predicted"])
# plt.savefig("vae_params.png")
print(sum(p.numel() for p in vae.parameters() if p.requires_grad))