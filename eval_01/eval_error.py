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
config_list = [ 4, 9, 17]
mape_list = []
smape_list = []
mse_list = []   

for i in config_list:

    mape = mean_absolute_percentage_error(data_test[i,5:505].squeeze().numpy(), y_pred_test[i,:500].squeeze().numpy())
    smape = s_mape(data_test[i,5:505].squeeze().numpy(), y_pred_test[i,:500].squeeze().numpy())
    mse = torch.mean((data_test[i,5:505].squeeze() - y_pred_test[i,:500].squeeze())**2).item()
    mape_list.append(mape)
    smape_list.append(smape)
    mse_list.append(mse)

print("MAPE: ", mape_list)
print("S_MAPE: ", smape_list)
print("MSE: ", mse_list)
