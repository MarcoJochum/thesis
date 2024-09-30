
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
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


latent_dim = 10

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


VAE= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))

tft_model = TFTModel.load("models/study_5/tft_study_5_opt_qr_"+ args.data_type+".pt", map_location="cpu")##

tft_model.to_cpu()
#best_model = tft_model.load_from_checkpoint(model_name="lat_10_opt", best=True)
data_train = Tft_config.data_train
data_train = data_train/ torch.mean(data_train)
train_params = Tft_config.train_params##
data_test = Tft_config.data_test
data_test = data_test/ torch.mean(data_train)
test_params = Tft_config.test_params##
n_configs = data_test.shape[0]   

with torch.no_grad():
   data_train_vae = reshape_vae(data_train)##
   data_train_vae = VAE(data_train_vae)[2]##
   data_test_vae = reshape_vae(data_test)##
   data_test_vae = VAE(data_test_vae)[2]##
   data_train_vae = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)##
   data_test_vae = unshape_vae(data_test_vae, data_test.shape[0], data_test.shape[1], True)##

model_input = data_test_vae[:, :5]



train_series_list = []    
for i in range(data_train.shape[0]):
   
    stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [float(train_params[i,1])], "V_bias": [float(train_params[i,2])]})##
    series = TimeSeries.from_values(data_train_vae[i,:5].numpy(), static_covariates=stat_cov)##
    
    train_series_list.append(series)##

test_series_list = []
for i in range(data_test.shape[0]):
    
    stat_cov = pd.DataFrame(data ={"epsr": [float(test_params[i,0])], "conc": [float(test_params[i,1])], "V_bias": [float(test_params[i,2])]})##
    series = TimeSeries.from_values(model_input[i].numpy(), static_covariates=stat_cov)##
    
    test_series_list.append(series)##
y_pred_lat = torch.zeros((data_train.shape[0],Tft_config.inference_steps,latent_dim))##
y_pred_lat_std = torch.zeros((data_train.shape[0],Tft_config.inference_steps,latent_dim))##
y_pred_test_lat = torch.zeros((data_test.shape[0],Tft_config.inference_steps,latent_dim))###
y_pred_test_std = torch.zeros((data_test.shape[0],Tft_config.inference_steps,latent_dim))###

pred_test_series_list = []
pred_train_series_list = []


for j in range(0,data_train.shape[0]):
    pred_train = tft_model.predict(series=train_series_list[j], n = Tft_config.inference_steps, num_samples=Tft_config.n_samples_inference)##
    
    pred_train_series_list.append(pred_train)
    y_pred_lat[j] = torch.mean(torch.tensor(pred_train.all_values()), 2)##
    y_pred_lat_std[j] = torch.std(torch.tensor(pred_train.all_values()), 2)##
    if j < data_test.shape[0]:
        pred_test= tft_model.predict(series=test_series_list[j], n = Tft_config.inference_steps, num_samples=Tft_config.n_samples_inference)##

        pred_test_series_list.append(pred_test)
        y_pred_test_lat[j] = torch.mean(torch.tensor(pred_test.all_values()), 2)##
        y_pred_test_std[j] = torch.std(torch.tensor(pred_test.all_values()), 2)##

explainer = TFTExplainer(tft_model,background_series=pred_test_series_list)
explanation = explainer.explain()
explainer.plot_variable_selection(explanation)
stat_cov = explanation.get_static_covariates_importance()
encoder_imp = explanation.get_encoder_importance()
print("encoder_imp", encoder_imp[0].shape)
#encoder_mean = np.mean(encoder_imp[:].iloc[0].values)/20
fig,ax = plt.subplots()
ax.bar(encoder_imp[1].columns, encoder_imp[1].iloc[0].values)
plt.xticks(rotation=45)  
plt.tight_layout()
plt.savefig("encoder_importance.png")
print("stat_cov", stat_cov)

first_row = stat_cov[0].iloc[0]
print("first row", first_row)
fig, ax = plt.subplots()
ax.bar(stat_cov[0].columns, stat_cov[0].iloc[0].values)

ax.set_ylabel('Importance [%]')
ax.set_title('Static Covariates Importance')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.savefig("stat_cov_importance.png")
plt.savefig("fig_report/stat_cov_importance_native.png", format="png")


explainer.plot_attention(explanation, plot_type="time")
plt.savefig("attention.png", format="png")
explainer.plot_attention(explanation, plot_type="heatmap")
plt.savefig("attention_heatmap.png", format="png")