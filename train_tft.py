import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.likelihood_models import QuantileRegression
from config.tft import Tft_config
from NNs.autoencoder import *
from lib.helper import *
import warnings
import time
import argparse
import pytorch_lightning as pl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


n_steps = 300
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()


##Load the VAE model
VAE= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))

## Switch between mean (avg) and standar deviation (std) field data
if args.data_type == 'avg':
    data_train = Tft_config.data_train_avg
    train_params = Tft_config.train_params_avg

elif args.data_type == 'std':
    data_train = Tft_config.data_train_std
    train_params = Tft_config.train_params_std
data_train = data_train/ torch.mean(data_train)
data_train = data_train[:,:n_steps]



with torch.no_grad():
   data_train_vae = reshape_vae(data_train)
   data_train_vae = VAE(data_train_vae)[2]
   data_train = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)



series_list = []    
for i in range(data_train.shape[0]):
    ## Create static covariates for each time series
    stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [np.log10(float(train_params[i,1]))], "V_bias": [float(train_params[i,2])]})
    ## Create time series object witht the kMC data and the corresponding static covariates
    series = TimeSeries.from_values(data_train[i].numpy(), static_covariates=stat_cov)#all training data convertd to time series type
    ## Create a list (sequence) of time series objects which can be fed into Darts models
    series_list.append(series)

loss_logger = LossLogger()

train_series = series_list[:Tft_config.train_size]
val_series = series_list[Tft_config.train_size:]


my_model = TFTModel(
    input_chunk_length=Tft_config.seq_length,
    output_chunk_length=Tft_config.pred_length,
    hidden_size=Tft_config.hidden,
    lstm_layers=Tft_config.lstm_layers,
    num_attention_heads=Tft_config.num_attention_heads,
    dropout=Tft_config.dropout,
    batch_size=Tft_config.batch_size,
    n_epochs=Tft_config.n_epochs,
    add_relative_index=True, #this needs to be true becaue i do not have covariant data
    add_encoders=None,
    optimizer_kwargs={"lr": Tft_config.lr},
    model_name="lat_10_opt",
    likelihood= QuantileRegression(
        quantiles=Tft_config.quantiles
    ),  # QuantileRegression is set per default
    
    #loss_fn=torch.nn.MSELoss(),
    random_state=42,
    force_reset=True,
    pl_trainer_kwargs={
       "accelerator": "gpu",
       "devices": [0],
       "callbacks": [loss_logger],
       
    },
    save_checkpoints=True, 
)
start_time = time.time()
my_model.fit(series=train_series, val_series=val_series, verbose=True)


end_time = time.time()
print("Training time:", end_time-start_time)
print("loss_logger.train_loss", loss_logger.train_loss)
print("loss_logger.val_loss", loss_logger.val_loss)
train_loss = np.array(loss_logger.train_loss)
val_loss = np.array(loss_logger.val_loss)
np.save("models/no_sv/study_5_opt_qr_train_loss_"+ args.data_type +"no_sv_300_steps.npy", train_loss)
np.save("models/no_sv/study_5_opt_qr_val_loss_"+ args.data_type +"no_sv_300_steps.npy", val_loss)

best_model = my_model.load_from_checkpoint(model_name="lat_10_opt", best=True)
best_model.save("models/no_sv/tft_study_5_opt_qr_"+ args.data_type +"_no_sv_300_steps.pt")
