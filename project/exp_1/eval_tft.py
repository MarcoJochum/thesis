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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.disable(logging.CRITICAL)

latent_dim = Tft_config.latent_dim

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


VAE = torch.load("models/model_vae_lin.pth", map_location=torch.device("cpu"))
tft_model = TFTModel.load("models/tft_model_lb_no_stat_cov.pt", map_location="cpu")##
tft_model.to_cpu()

data_train = Tft_config.data_train
train_params = Tft_config.train_params##

data_test = Tft_config.data_test
test_params = Tft_config.test_params##
n_configs = data_test.shape[0]   

with torch.no_grad():
   data_train_vae = reshape_vae(data_train)##
   data_train_vae = VAE(data_train_vae)[2]##
   data_test_vae = reshape_vae(data_test)##
   data_test_vae = VAE(data_test_vae)[2]##
   data_train_vae = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)##
   data_test_vae = unshape_vae(data_test_vae, data_test.shape[0], data_test.shape[1], True)##


train_series_list = []    
for i in range(data_train.shape[0]):
    #add scaling in this loop
    #train scaler 
    #stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [float(train_params[i,1])], "V_bias": [float(train_params[i,2])]})##
    series = TimeSeries.from_values(data_train_vae[i,:5].numpy())#, static_covariates=stat_cov)##
    
    train_series_list.append(series)##

test_series_list = []
for i in range(data_test.shape[0]):
    #add scaling in this loop
    #train scaler 
    #stat_cov = pd.DataFrame(data ={"epsr": [float(test_params[i,0])], "conc": [float(test_params[i,1])], "V_bias": [float(test_params[i,2])]})##
    series = TimeSeries.from_values(data_test_vae[i,:5].numpy())#, static_covariates=stat_cov)##
    
    test_series_list.append(series)##




y_pred_lat = torch.zeros((n_configs,Tft_config.inference_steps,latent_dim))##
y_pred_test_lat = torch.zeros((n_configs,Tft_config.inference_steps,latent_dim))##
train_trj = data_train[:n_configs]
test_trj = data_test[:n_configs]

for j in range(0,n_configs):
    pred = tft_model.predict(series=train_series_list[j], n = Tft_config.inference_steps, num_samples=Tft_config.n_samples_inference)##
    

    y_pred_lat[j] = torch.mean(torch.tensor(pred.all_values()), 2)##

    pred_test= tft_model.predict(series=test_series_list[j], n = Tft_config.inference_steps, num_samples=Tft_config.n_samples_inference)##
    
    y_pred_test_lat[j] = torch.mean(torch.tensor(pred_test.all_values()), 2)##

    


##decode with vae
y_pred_lat = reshape_vae(y_pred_lat)

print("MAPE on train set:", mape)
y_pred_test_lat = reshape_vae(y_pred_test_lat)

y_pred = VAE.decoder(y_pred_lat)
y_pred_test = VAE.decoder(y_pred_test_lat)

y_pred = unshape_vae(y_pred, n_configs, Tft_config.inference_steps, False)
y_pred_test = unshape_vae(y_pred_test, n_configs, Tft_config.inference_steps, False)

## average across x dimension

y_pred = torch.mean(y_pred, 2)
y_pred_test = torch.mean(y_pred_test, 2)

with open("../../data_kmc/2d_sets/test_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names[:n_configs]  

train_trj = torch.mean(train_trj.squeeze(), dim=2)
test_trj = torch.mean(test_trj.squeeze(), dim=2)

##plot

labels = [] 
time = np.logspace(-7, -4, 1000, 'o')   
with torch.no_grad():
      
    fig,axs = plt.subplots(n_configs,2, figsize=(6,24))
    for j in range(5, n_configs):

        for i in range(0, 200,1):
            
           
            axs[j,0].plot(np.linspace(0,100,100), y_pred_test[j, i].numpy())
            labels.append([f"t = {time[i]}"])
            axs[j,0].set_ylim(0, 5)
            axs[j,0].set_title("Prediction "+ names[j])
            #axs[0].legend(labels)

            axs[j,1].plot(np.linspace(0,100,100), test_trj[j,5+i].numpy())
            #labels.append([f"t = {time[i-1]}"])
            axs[j,1].set_ylim(0, 5)
            axs[j,1].set_title("Ground truth "+ names[j])
            #axs[1].legend(labels)


fig.suptitle("Prediction on test set. With 5 steps given as input.\n Prediction horizon 200 steps.\n No static covariates", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.savefig("tft_pred_lb_5_no_stat_cov.png")
mape_list = []
for i in range(0, n_configs):
    mape = mean_absolute_percentage_error(test_trj[i,5:505].detach().numpy(), y_pred_test[i].detach().numpy())
    mape_list.append(mape)
mape_list = np.array(mape_list)
print("Mean absolute percentage error on test set:", (mape_list))
print("Mean of all Mean absolute percentage errors on test set:", np.mean(mape_list))
print("Standard deviation of mean absolute percentage error on test set:", np.std(mape_list))


mean_error = mean_absolute_percentage_error(train_trj[-1,5:505].detach().numpy(), y_pred[-1].detach().numpy())



# explainer = TFTExplainer(tft_model,background_series=test_series_list)
# explanation = explainer.explain(test_series_list[5])
# explainer.plot_variable_selection(explanation)
# plt.savefig("tft_expl_2.png")