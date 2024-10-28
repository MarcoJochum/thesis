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




if args.data_type == 'avg':
    tft_model = TFTModel.load("models/no_sv/study_5_no_sv_500_steps_"+ args.data_type +".pt", map_location="cpu")##
    tft_model.to_cpu()
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg
    train_params = Tft_config.train_params_avg##
    test_params = Tft_config.test_params_avg##
elif args.data_type == 'std':
    tft_model = TFTModel.load("models/no_sv/study_5_no_sv_300_steps_"+ args.data_type +".pt", map_location="cpu")##
    tft_model.to_cpu()
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std
    train_params = Tft_config.train_params_std##
    test_params = Tft_config.test_params_std##
else:
    print("Data type not recognized")
    exit()

print("Reminder: Change scaling of c_bulk depending on model used!!!")

data_test = data_test/ torch.mean(data_train) ## This has to preced the next line because the mean of the training data is used to normalize the test data
data_train = data_train/ torch.mean(data_train)


##Number of configurations to perform inference on
n_configs = data_test.shape[0]   ## all configs

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
   
    stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [float(np.log10(train_params[i,1]))], "V_bias": [float(train_params[i,2])]})##
    series = TimeSeries.from_values(data_train_vae[i,:5].numpy())#, static_covariates=stat_cov)##
    
    train_series_list.append(series)##

test_series_list = []
for i in range(data_test.shape[0]):
    
    stat_cov = pd.DataFrame(data ={"epsr": [float(test_params[i,0])], "conc": [float(np.log10(test_params[i,1]))], "V_bias": [float(test_params[i,2])]})##
    series = TimeSeries.from_values(model_input[i].numpy())#, static_covariates=stat_cov)##
    
    test_series_list.append(series)##




y_pred_lat = torch.zeros((data_train.shape[0],Tft_config.inference_steps,latent_dim))##
y_pred_lat_std = torch.zeros((data_train.shape[0],Tft_config.inference_steps,latent_dim))##
y_pred_test_lat = torch.zeros((data_test.shape[0],Tft_config.inference_steps,latent_dim))###
y_pred_test_std = torch.zeros((data_test.shape[0],Tft_config.inference_steps,latent_dim))###

pred_test_series_list = []
pred_train_series_list = []
## Measure time needed for inference
start_time = time.time()
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

##Measure time needed for inference
end_time = time.time()


##Reshape for VAE
y_pred_lat = reshape_vae(y_pred_lat)
y_pred_lat_std = reshape_vae(y_pred_lat_std)
y_pred_test_lat = reshape_vae(y_pred_test_lat)
y_pred_test_std = reshape_vae(y_pred_test_std)
##Decode with VAE
y_pred = VAE.decoder(y_pred_lat)
y_pred_std = VAE.decoder(y_pred_lat_std)
y_pred_test = VAE.decoder(y_pred_test_lat)
y_pred_test_std = VAE.decoder(y_pred_test_std)
##Reshape to original shape
y_pred = unshape_vae(y_pred, data_train.shape[0], Tft_config.inference_steps, False)
y_pred_std = unshape_vae(y_pred_std, data_train.shape[0], Tft_config.inference_steps, False)
y_pred_test = unshape_vae(y_pred_test, data_test.shape[0], Tft_config.inference_steps, False)
y_pred_test_std = unshape_vae(y_pred_test_std, data_test.shape[0], Tft_config.inference_steps, False)

##save predictions
np.save("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_test.npy", y_pred_test.detach().numpy())
np.save("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_test_std.npy", y_pred_test_std.detach().numpy())
np.save("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_train.npy", y_pred.detach().numpy())
np.save("../../data_kmc/2d_results/lin_time/study_5_no_sv_500_steps_"+ args.data_type+ "/y_pred_train_std.npy", y_pred_std.detach().numpy())
print("Inference time:", end_time-start_time)
