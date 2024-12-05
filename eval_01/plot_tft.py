import torch
import numpy as np
import matplotlib.pyplot as plt
from config.tft import Tft_config
from lib.helper import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()





if args.data_type == 'avg':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)

    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg
    train_params = Tft_config.train_params_avg
    test_params = Tft_config.test_params_avg
elif args.data_type == 'std':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_300_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_300_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_300_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_no_sv_300_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    train_params = Tft_config.train_params_std
    test_params = Tft_config.test_params_std

    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std
else:
    print("Data type not recognized")
    exit()
data_test = data_test/ torch.mean(data_train) ## This has to preced the next line because the mean of the training data is used to normalize the test data
data_train = data_train/ torch.mean(data_train)

n_configs = y_pred.shape[0]
train_trj = data_train
test_trj = data_test


with open("../../data_kmc/2d_sets/test_set_lin_80_20_"+ args.data_type + "_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names  

##2D plots
z_width = np.linspace(0, 100, 100)
x_width = np.linspace(0, 50, 50)
for j in range(0):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot y_pred_test
    im1 = axs[0].imshow(y_pred_test[9,j].squeeze().numpy(), aspect='auto', extent=[z_width.min(), z_width.max(), x_width.min(), x_width.max()], vmax=2.5,origin='lower')
    axs[0].set_title(f"Prediction {names[9]}", fontsize=12)
    axs[0].set_xlabel("Position z [nm]")
    axs[0].set_ylabel("Position x [nm]")
    fig.colorbar(im1, ax=axs[0], label="Relative concentration")

    # Plot test_trj
    im2 = axs[1].imshow(test_trj[9,5+j].squeeze().numpy(), aspect='auto', extent=[z_width.min(), z_width.max(), x_width.min(), x_width.max()],vmax = 2.5,origin='lower')
    axs[1].set_title(f"Ground truth {names[9]}", fontsize=12)
    axs[1].set_xlabel("Position z [nm]")
    axs[1].set_ylabel("Position x [nm]")
    fig.colorbar(im1, ax=axs[1], label="Relative concentration")

    plt.tight_layout()
    plt.savefig(f"fig_report/animation/test"+str(j)+".png", format="png")
    plt.close(fig)


##1D plots
## average across x dimension

y_pred = torch.mean(y_pred, 2)
y_pred_std = torch.mean(y_pred_std, 2)
y_pred_test = torch.mean(y_pred_test, 2)
y_pred_test_std = torch.mean(y_pred_test_std, 2)
train_trj = torch.mean(train_trj.squeeze(), dim=2)
test_trj = torch.mean(test_trj.squeeze(), dim=2)

##plot

labels = [] 
time = np.linspace(1e-7, 1e-4, 1000, 'o')   
z_width = np.linspace(0,100,100)
with torch.no_grad():
      
    fig,axs = plt.subplots(2,3, figsize=(10,5))
    k = 0
    for j in [ 4, 9, 17]:#[14,15,9]:
        
        for i in [10, 20, 50 , 100,  200,  300]:#[ 100, 300,400, 500,600, 800, 990]:
            
            time_label = f"t = {time[i]:.2e}"
            axs[0,k].plot(z_width, y_pred_test[j, i].numpy(), label = time_label)
            #axs[0,k].fill_between(z_width, y_pred_test[j,i,:].numpy() - y_pred_test_std[j,i,:].numpy(), y_pred_test[j,i,:].numpy() + y_pred_test_std[j,i,:].numpy(), alpha=0.3)
            axs[0,k].set_ylim(0, 5)
            #axs[0, k].set_yscale('log') 
            axs[0,k].set_xlabel("Position z [nm]")
            axs[0,k].set_ylabel("Relative concentration")
            axs[0,k].set_title("Prediction ")#+ names[j], fontsize=10)
            if k== 0:
                axs[0,k].legend()

            axs[1,k].plot(z_width, test_trj[j,5+i].numpy(), label = time_label)
            axs[1,k].set_ylim(0, 5)
            #axs[1, k].set_yscale('log') 
            axs[1, k].set_xlabel("Position z [nm]")
            axs[1, k].set_ylabel("Relative concentration")
            axs[1,k].set_title("Ground truth ")#+ names[j], fontsize=10)
            if k == 0:    
                axs[1,k].legend()
        
        k+=1


fig.suptitle("Prediction on test set with 5 steps given as input.\n Prediction horizon 500 steps.", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.savefig("logscale.png", format="png")
plt.close(fig)  
mape_list = []
smape_list = []
for i in range(Tft_config.train_size, train_trj.shape[0]):
    mape = mean_absolute_percentage_error(train_trj[i,5:505].detach().numpy(), y_pred[i,:500].detach().numpy())
    mape_list.append(mape)
    smape = s_mape(train_trj[i,5:505].detach().numpy(), y_pred[i,:500].detach().numpy())
    smape_list.append(smape)
mape_list = np.array(mape_list)
smape_list = np.array(smape_list)
print(mape_list)
mape_list_test = []
smape_list_test = []
for i in range(0, test_trj.shape[0]):
    mape = mean_absolute_percentage_error(train_trj[i,5:505].detach().numpy(), y_pred[i,:500].detach().numpy())
    mape_list_test.append(mape)
    smape = s_mape(train_trj[i,5:505].detach().numpy(), y_pred[i,:500].detach().numpy())
    smape_list_test.append(smape)

mape_list_test = np.array(mape_list_test)
smape_list_test = np.array(smape_list_test)
print("Mean of all MAPE on val set:", np.sum(mape_list)/(train_trj.shape[0]-Tft_config.train_size))
print("Standard deviation of MAPE on val set:", np.std(mape_list))
print("Mean of all SMAPE on val set:", np.sum(smape_list)/(train_trj.shape[0]-Tft_config.train_size))
print("Standard deviation of SMAPE on val set:", np.std(smape_list))
print("mean squared prediction error", torch.mean((train_trj[:,5:505] - y_pred[:, :500])**2))
print("######################################\n")

print("Mean of all MAPE on test set:", np.sum(mape_list_test)/(test_trj.shape[0]))
print("Standard deviation of MAPE on test set:", np.std(mape_list_test))
print("Mean of all SMAPE on test set:", np.sum(smape_list_test)/(test_trj.shape[0]))
print("Standard deviation of SMAPE on test set:", np.std(smape_list_test))
print("mean squared prediction error_test", torch.mean((test_trj[:,5:505] - y_pred_test[:, :500])**2))
mean_error = mean_absolute_percentage_error(train_trj[-1,5:505].detach().numpy(), y_pred[-1,:500].detach().numpy())


### Error plot depending on time step

error_t = np.abs(test_trj[:,5:].detach().numpy() - y_pred_test[:,:995].detach().numpy())#/y_pred_test.detach().numpy()
print("minimum prediction",)
error_t_mean = np.mean(error_t, axis=0).squeeze()

error_t_std = np.std(error_t, axis=0).squeeze() 

time = np.linspace(1e-7, 1e-4, 1000, 'o')

fig,ax = plt.subplots(
    figsize=(10,6)
)
#breakpoint()
for i in range(0, 995, 100):
    #time_label = f"t = {time[i]:.2e}"

    ax.plot(z_width, error_t_mean[i], label=f"t = {time[i]:.2e}")
    ax.legend() 

plt.xlabel("Position z [nm]")
plt.ylabel("Error")
plt.title("Error in prediction")
plt.savefig("fig_report/error_z.png", format="png")
plt.close(fig)
error_t_mean_c_z = np.mean(error_t_mean, axis=1)
steps = np.linspace(5, 1000, 995)
plt.plot(steps, error_t_mean_c_z, label="Mean error")
plt.axvline(x=steps[62], color='red', linestyle='--', linewidth=2, label='Vertical Line at tstep 62')
plt.axvline(x=steps[2*62], color='red', linestyle='--', linewidth=2, label='Vertical Line at x=250')
plt.axvline(x=steps[8*62], color='red', linestyle='--', linewidth=2, label='Vertical Line at x=250')
#plt.axvline(x=steps[300], color='blue', linestyle='--', linewidth=2, label='Vertical Line at x=250')
#plt.axvline(x=steps[940], color='blue', linestyle='--', linewidth=2, label='Vertical Line at x=250')
plt.xlabel("# of time steps")
#plt.yscale('log')
plt.ylabel("Error")

plt.title("Mean error across x,z and configurations in prediction")
plt.savefig("fig_report/error_Test_t.png", format="png")