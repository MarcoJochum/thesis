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
fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    # Plot y_pred_test
im1 = axs.imshow(y_pred_test[9,50].squeeze().numpy(), aspect='auto', extent=[z_width.min(), z_width.max(), x_width.min(), x_width.max()], vmax=2.5,origin='lower')
axs.axis('off')

# Plot test_trj


plt.tight_layout()
plt.savefig("fig_report/perspective_test.png", format="png", bbox_inches='tight', pad_inches=0,dpi=700)
plt.close(fig)


#



