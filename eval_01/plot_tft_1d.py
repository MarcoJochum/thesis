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
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg

elif args.data_type == 'std':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std
data_test = data_test/ torch.mean(data_train) ## This has to preced the next line because the mean of the training data is used to normalize the test data
data_train = data_train/ torch.mean(data_train)
n_configs = y_pred.shape[0]

data_test_std = torch.std(data_test, dim=3).squeeze()

data_test_mean = torch.mean(data_test, dim=3).squeeze()
y_pred_test_mean = torch.mean(y_pred_test, dim=2).squeeze()
time = np.linspace(1e-7, 1e-4, 1000)
labels = [] 

t_end = 300
t_space = 50
num_plots=6

fig,axs = plt.subplots(2,3, figsize=(10,8))
z_width = np.linspace(0,100,100)
k=0
column_labels = ["a", "b", "c"]
config_list = [ 4, 9, 17]
config_labels = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_"+args.data_type+"_list.txt")
for j in config_list:
          
    for i in [10, 20, 50 , 100,  200,  300]:
    
            

            axs[0,k].plot(z_width, data_test_mean[j,i+5],label=f"t = {time[i+5]:.2e}s")
            
            

            axs[0,k].set_ylim(0, 20) 
            axs[0,k].set_title("Ground truth")
            
            axs[1,k].plot(z_width, y_pred_test_mean[j,i].numpy(), label=f"t = {time[i+5]:.2e}s")
           
            
            axs[1,k].set_ylim(0, 20)
            axs[1,k].set_title("Predicted")

            if k == 0:  
                
                axs[0,k].legend()
                axs[1,k].legend()
                axs[0,k].set_ylabel("Relative Concentration")
                axs[1,k].set_ylabel("Relative Concentration")
                axs[1,k].set_xlabel("Position z [nm]")
                axs[0,k].set_xlabel("Position z [nm]")
    title = "{})  $\Lambda^{}$=[{:.0f}, {:.1e} $\\text{{cm}}^{{-3}}$, {:.2} V] ".format(column_labels[k],column_labels[k],float(config_labels[j, 0]), float(config_labels[j, 1]), float(config_labels[j, 2]))

    fig.text(0.5, -0.3, title, ha='center', fontsize=12, transform=axs[1, k].transAxes)
    k+=1
for ax in axs.flat:
    ax.grid(True, which='both')
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.subplots_adjust(wspace=0.2, hspace=0.5, bottom=0.2)
plt.savefig("fig_report/tft_1d_"+args.data_type+"_conf_"+str(config_list[0])+"_"+str(config_list[1])+"_"+str(config_list[2])+"_presentation.pdf", 
            format="pdf", bbox_inches='tight')

print(sum(p.numel() for p in vae.parameters() if p.requires_grad))


