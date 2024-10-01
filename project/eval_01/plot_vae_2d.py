import torch
import numpy as np
import matplotlib.pyplot as plt
from config.tft import Tft_config
from lib.helper import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()

VAE= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))
y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/tft_model_lb_5_2/y_pred_train.npy"), dtype=torch.float32)
y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/tft_model_lb_5_2/y_pred_test.npy"), dtype=torch.float32) 
y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/tft_model_lb_5_2/y_pred_test_std.npy"), dtype= torch.float32)

if args.data_type == 'avg':
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg

elif args.data_type == 'std':
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std

train_params = Tft_config.train_params##
n_configs = y_pred.shape[0]
test_params = Tft_config.test_params##
train_trj = data_train[:n_configs]
test_trj = data_test[:n_configs]

with torch.no_grad():
    data_train_vae = reshape_vae(data_train)
    data_test_vae = reshape_vae(data_test)
    data_train_vae = VAE(data_train_vae)[2]
    data_test_vae = VAE(data_test_vae)[2]
    data_train_decoded = VAE.decoder(data_train_vae)
    data_test_decoded = VAE.decoder(data_test_vae)
    data_train_decoded = unshape_vae(data_train_decoded, data_train.shape[0], data_train.shape[1], False)
    data_test_decoded = unshape_vae(data_test_decoded, data_test.shape[0], data_test.shape[1], False)

with open("../../data_kmc/2d_sets/test_set_lin_80_20_"+args.data_type+"_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names[:n_configs]  

##2D plots
z_width = np.linspace(0, 100, 100)
x_width = np.linspace(0, 50, 50)
fig, axs = plt.subplots(3, 2, figsize=(12, 14))
time = np.linspace(1e-7, 1e-4, 1000)

k = 0
fig.suptitle(f"Prediction {names[9]}", fontsize=16)
for t in [5,50,499]:
    # Plot y_pred_test
    im1 = axs[k,0].imshow(data_test_decoded[9,t].squeeze().numpy(), aspect='auto', 
                          extent=[z_width.min(), z_width.max(), x_width.min(), x_width.max()], 
                          vmin=0.4,vmax=2.5,origin='lower')
    axs[k,0].set_title(f"Prediction", fontsize=12)
    axs[k,0].set_xlabel("Position z [nm]")
    axs[k,0].set_ylabel("Position x [nm]")
    
    # if k == 2:
    #     cbar = fig.colorbar(im1, fraction=0.08, pad=2)
    #     cbar.set_label("Relative concentration")


    im2=axs[k,1].imshow(data_test[9,t].squeeze().numpy(), aspect='auto', 
                        extent=[z_width.min(), z_width.max(), x_width.min(), x_width.max()], 
                        vmin=0.4, vmax=2.5,origin='lower')
    axs[k,1].set_title(f"Ground Truth ", fontsize=12)
    axs[k,1].set_xlabel("Position z [nm]")
    axs[k,1].set_ylabel("Position x [nm]")
    # if k == 2:
    #     cbar = fig.colorbar(im2,ax=axs[:,1], fraction=0.08, pad=0.04)
    #     cbar.set_label("Relative concentration")

    k+=1
# Plot test_trj
cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Relative concentration")


#plt.tight_layout(rect=[0, 0, 1.2, 0.96]) 
plt.subplots_adjust(wspace=0.2, hspace=0.4)
fig.text(0.5, 0.34, "c) Time step 499", ha='center', fontsize=12)
fig.text(0.5, 0.63, "b) Time step 50", ha='center', fontsize=12)
fig.text(0.5, 0.92, "a) Time step 10", ha='center', fontsize=12)
plt.savefig("fig_report/test/vae_2d_test.png", format="png",dpi=700)
plt.close(fig)


#



