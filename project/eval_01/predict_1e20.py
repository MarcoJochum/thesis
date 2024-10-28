import numpy as np
import pandas as pd
import matplotlib.animation as animation
from darts.explainability import TFTExplainer
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from config.tft import Tft_config
from NNs.autoencoder import *
from lib.helper import *
import warnings
import argparse
import time
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.serif'] = ['Palatino']

mpl.rcParams["axes.prop_cycle"] =mpl.cycler('color', ['black', '#003DFD', '#b512b8', '#11a9ba', '#0d780f', '#f77f07', '#ba0f0f', '#0b0bf7', '#f70bf7', '#0bf7f7', '#0bf70b', '#f70b0b', '#f7f7f7', '#7f7f7f', '#0b0b0b'])

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))

tft_model = TFTModel.load("models/scaled_sv/tft_study_5_opt_qr_"+ args.data_type+"_scaled_sv_500_steps.pt", map_location="cpu")##
tft_model.to_cpu()

epsr= "1400"
c_bulk = "1e+20"
v_bias = "2.0"
trj = torch.tensor(np.load("../../data_kmc/2d_high_c/"+epsr+"_"+c_bulk+"_"+v_bias+"/"+ args.data_type+"_trj.npy"), dtype=torch.float32)
trj = torch.reshape(trj,(1000,1,50,100))
trj.unsqueeze(0)
with torch.no_grad():
   #trj_vae = reshape_vae(trj)##
   #breakpoint()
   trj_vae = VAE(trj)[2]##
   
   #data_train_vae = unshape_vae(trj_vae, trj.shape[0], trj.shape[1], True).squeeze()##


stat_cov = pd.DataFrame(data ={"epsr": [float(1400)], "conc": [float(np.log10(1e20))], "V_bias": [float(1.0)]})

trj_series = TimeSeries.from_values(trj_vae[:5].numpy(), static_covariates=stat_cov)#all training data convertd to time series type

pred_trj = tft_model.predict(series= trj_series, n=500, num_samples= 100)
y_pred_lat = torch.mean(torch.tensor(pred_trj.all_values()), 2)
y_pred_lat_std = torch.std(torch.tensor(pred_trj.all_values()),2)

if args.data_type == 'avg':
    np.save("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_avg/y_pred_"+epsr+"_"+c_bulk+"_"+v_bias+".npy", y_pred_lat.numpy())
else:
    np.save("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_"+epsr+"_"+c_bulk+"_"+v_bias+".npy",y_pred_lat.numpy()) 

#y_pred_lat  = reshape_vae(y_pred_lat)
with torch.no_grad():
    y_pred = VAE.decoder(y_pred_lat)
    y_pred_std = VAE.decoder(y_pred_lat_std)
    
    trj_decoded = VAE.decoder(trj_vae)  

#y_pred = unshape_vae(y_pred, 1, Tft_config.pred_length, False)
y_pred = y_pred.squeeze()
trj_decoded = trj_decoded.squeeze()
y_pred_std = y_pred_std.squeeze()
y_pred = torch.mean(y_pred, 1)
trj_decoded = torch.mean(trj_decoded, 1)
y_pred_std = torch.mean(y_pred_std, 1)

trj = torch.mean(trj, 2).squeeze()
time = np.linspace(1e-7,1e-4, 1000)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))  
z_width = np.linspace(0,100,100)
# breakpoint()
for i in [ 10, 50, 100, 300]:

    axs[1].plot(z_width, y_pred[i].numpy(), label = f"t = {time[i]:.2e}")
    axs[1].set_xlabel("z-width [nm]") 
    axs[1].set_ylabel("Relative concentration")
    axs[1].set_ylim(0, 10)
    axs[1].set_title("Prediction ")
    axs[1].legend()

    axs[0].plot(z_width, trj[i], label = f"t = {time[i]:.2e}")
    axs[0].set_xlabel("z-width [nm]")
    axs[0].set_ylim(0, 10)
    axs[0].set_ylabel("Relative concentration", fontsize=14)
    axs[0].set_title("Ground truth")
    axs[0].legend()   
for ax in axs:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
title = "Parameters: {:.0f}, {:.1e} $\\text{{cm}}^{{-3}}$, {:.2} V".format(float(epsr), float(c_bulk), float(v_bias))
fig.suptitle(title, fontsize=16)
smape = s_mape(y_pred, trj[500])
mape = mean_absolute_percentage_error(y_pred, trj[:500])
mse = torch.mean((y_pred- trj[:500])**2)
print("MSE:", mse)
print("SMAPE:", smape)
print("MAPE:", mape)

plt.savefig("fig_report/high_c_bulk.pdf", format="pdf", bbox_inches='tight')
