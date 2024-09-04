import torch
import numpy as np
import matplotlib.pyplot as plt
from config.tft import Tft_config
from lib.helper import mean_absolute_percentage_error
y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/tft_model_lb_5_2/y_pred_train.npy"), dtype=torch.float32)
y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/tft_model_lb_5_2/y_pred_test.npy"), dtype=torch.float32) 
y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/tft_model_lb_5_2/y_pred_test_std.npy"), dtype= torch.float32)
data_train = Tft_config.data_train
train_params = Tft_config.train_params##
n_configs = y_pred.shape[0]
data_test = Tft_config.data_test
test_params = Tft_config.test_params##
train_trj = data_train[:n_configs]
test_trj = data_test[:n_configs]


with open("../../data_kmc/2d_sets/test_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names[:n_configs]  

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
    plt.savefig(f"fig_report/animation/surface_plot_t_"+str(j)+".png", format="png")
    plt.close(fig)


##1D plots
## average across x dimension

y_pred = torch.mean(y_pred, 2)
y_pred_test = torch.mean(y_pred_test, 2)
y_pred_test_std = torch.mean(y_pred_test_std, 2)
train_trj = torch.mean(train_trj.squeeze(), dim=2)
test_trj = torch.mean(test_trj.squeeze(), dim=2)

##plot

labels = [] 
time = np.linspace(1e-7, 1e-4, 1000, 'o')   
z_width = np.linspace(0,100,100)
with torch.no_grad():
      
    fig,axs = plt.subplots(2,3, figsize=(10,6))
    k = 0
    for j in [14,15,9]:
        
        for i in [10, 15, 20, 50, 100, 300, 499]:
            
            time_label = f"t = {time[i]:.2e}"
            axs[0,k].plot(z_width, y_pred_test[j, i].numpy(), label = time_label)
            #axs[0,k].fill_between(z_width, y_pred_test[j,i,:].numpy() - y_pred_test_std[j,i,:].numpy(), y_pred_test[j,i,:].numpy() + y_pred_test_std[j,i,:].numpy(), alpha=0.3)
            axs[0,k].set_ylim(0, 5)
            axs[0, k].set_yscale('log') 
            axs[0,k].set_xlabel("Position z [nm]")
            axs[0,k].set_ylabel("Relative concentration")
            axs[0,k].set_title("Prediction "+ names[j], fontsize=10)
            if k== 0:
                axs[0,k].legend()

            axs[1,k].plot(z_width, test_trj[j,5+i].numpy(), label = time_label)
            axs[1,k].set_ylim(0, 5)
            axs[1, k].set_yscale('log') 
            axs[1, k].set_xlabel("Position z [nm]")
            axs[1, k].set_ylabel("Relative concentration")
            axs[1,k].set_title("Ground truth "+ names[j], fontsize=10)
            if k == 0:    
                axs[1,k].legend()
        
        k+=1


fig.suptitle("Prediction on test set with 5 steps given as input.\n Prediction horizon 500 steps.", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.savefig("logscale.png", format="png")
mape_list = []
for i in range(0, n_configs):
    mape = mean_absolute_percentage_error(test_trj[i,5:505].detach().numpy(), y_pred_test[i].detach().numpy())
    mape_list.append(mape)
mape_list = np.array(mape_list)


mape_list[13] = 0

print("Mean of all Mean absolute percentage errors on test set:", np.sum(mape_list)/(n_configs-1))
print("Standard deviation of mean absolute percentage error on test set:", np.std(mape_list))


mean_error = mean_absolute_percentage_error(train_trj[-1,5:505].detach().numpy(), y_pred[-1].detach().numpy())



