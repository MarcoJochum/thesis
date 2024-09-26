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
import pytorch_lightning as pl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")



config_opt_tft_study_4 = Tft_config()




VAE= torch.load("models/model_vae_lin_lat_10.pth", map_location=torch.device("cpu"))

data_train = Tft_config.data_train
data_train = data_train/ torch.mean(data_train)
data_train = data_train[:,:Tft_config.n_steps]
train_params = Tft_config.train_params



with torch.no_grad():
   data_train_vae = reshape_vae(data_train)
   data_train_vae = VAE(data_train_vae)[2]
   data_train = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)



series_list = []    
for i in range(data_train.shape[0]):
    stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [float(train_params[i,1])], "V_bias": [float(train_params[i,2])]})
    series = TimeSeries.from_values(data_train[i].numpy(), static_covariates=stat_cov)#all training data convertd to time series type
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
)
start_time = time.time()
my_model.fit(series=train_series, val_series=val_series, verbose=True)


end_time = time.time()
print("Training time:", end_time-start_time)
print("loss_logger.train_loss", loss_logger.train_loss)
print("loss_logger.val_loss", loss_logger.val_loss)
train_loss = np.array(loss_logger.train_loss)
val_loss = np.array(loss_logger.val_loss)
np.save("models/study_4/study_4_opt_qr_train_loss.npy", train_loss)
np.save("models/study_4study_4_opt_qr_val_loss.npy", val_loss)
my_model.save("models/study_4/tft_study_4_opt_qr.pt")
