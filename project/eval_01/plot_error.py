
import torch
import numpy as np
import matplotlib.pyplot as plt
from config.tft import Tft_config
from lib.helper import *
import argparse
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.serif'] = ['Palatino']


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()



if args.data_type == 'avg':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg
    test_params = Tft_config.test_params_avg#
    train_params = Tft_config.train_params_avg
elif args.data_type == 'std':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    data_train = Tft_config.data_train_avg
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std
    test_params = Tft_config.test_params_std#
    train_params = Tft_config.train_params_std
else:
    print("Data type not recognized")
    exit()



data_test = data_test/ torch.mean(data_train) ## This has to preced the next line because the mean of the training data is used to normalize the test data
data_train = data_train/ torch.mean(data_train)
##
n_configs = y_pred.shape[0]
#
train_trj = data_train
test_trj = data_test.squeeze()
z_width = np.linspace(0, 100, 100)
steps = np.linspace(5, 1000, 995)


error_t = np.abs(test_trj[:,5:].detach().numpy() - y_pred_test[:,:995].detach().numpy())#/y_pred_test.detach().numpy()


error_t_mean = np.mean(error_t, axis=0).squeeze()
error_t_mean_x_mean = np.mean(error_t_mean, axis=1).squeeze()
error_t_std = np.std(error_t, axis=0).squeeze() 

time = np.linspace(1e-7, 1e-4, 1000, 'o')

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5])
ax1 = fig.add_subplot(gs[0, 0])
for i in range(0, 994, 100):
    #time_label = f"t = {time[i]:.2e}"

    ax1.plot(z_width, error_t_mean_x_mean[i], label=f"t = {time[i+5]:.2e}s")
    ax1.legend(fontsize=15) 

ax1.set_xlabel("Position z [nm]", fontsize=15)
ax1.set_ylabel("MAE" , fontsize=15)
#ax1.set_title("Error in prediction")
ax1.grid()

### Second plot error over time 
ax2 = fig.add_subplot(gs[0, 1])
error_t_mean_c_z = np.mean(error_t_mean_x_mean, axis=1)

ax2.plot(steps, error_t_mean_c_z)
ax2.axvline(x=steps[62], color='red', linestyle='--', linewidth=2, label=r'$1 \times \tau$')
ax2.axvline(x=steps[2*62], color='red', linestyle='-.', linewidth=2, label=r'$2 \times \tau$')
ax2.axvline(x=steps[3*62], color='red', linestyle='dotted', linewidth=2, label=r'$3 \times \tau$')
ax2.axvline(x=steps[500], color='#f77f07', linestyle='--', linewidth=2, label='End of training')

#ax2.axvline(x=steps[15*62], color='blue', linestyle='--', linewidth=1, label='$15 \\times L_{{out}}$')
ax2.set_xlabel("# of time steps", fontsize=15)
ax2.legend(fontsize=15)
#ax2.yscale('log')
ax2.set_ylabel("MAE")
ax2.grid()

#ax2.set_title("Mean absolute prediction error over time")

ax3 = fig.add_subplot(gs[1, :])
data_test_mean = torch.mean(data_test.squeeze(), dim=2).squeeze()
y_pred_test_mean = torch.mean(y_pred_test.squeeze(), dim=2).squeeze()

k=0 
for i in [10, 20, 50 , 100,  200,  300]:
    #time_label = f"t = {time[i]:.2e}"

    ax3.plot(z_width, data_test_mean[9,i], color="r",label=f"Ground truth")
    ax3.plot(z_width, y_pred_test_mean[9,i].detach().numpy(),color="b", label=f"Prediction")
    
    if k == 0:  
        ax3.legend(fontsize=15) 
        
    k+=1
ax3.set_xlabel("Position z [nm]", fontsize=15)
ax3.set_ylabel("Relative Concentration", fontsize=15)
ax3.grid()

fig.subplots_adjust(wspace=0.2, hspace=0.4)
ax3.set_title( "c)", y=-0.4, fontsize=16)
ax2.set_title( "b)", y=-0.2, fontsize=16)
ax1.set_title( "a)", y=-0.2, fontsize=16)
# fig.text(0.75, 0.4, "b)", ha='center', fontsize=12)
# fig.text(0.5, 0.02, "c)", ha='center', fontsize=12)
fig.savefig("fig_report/error/error_t_"+args.data_type+".pdf", format="pdf")