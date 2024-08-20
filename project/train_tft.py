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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
import logging
from pytorch_lightning.utilities.model_summary import ModelSummary
logging.disable(logging.CRITICAL)




VAE= torch.load("models/model_vae_lin.pth", map_location=torch.device("cpu"))

data_train = Tft_config.data_train
data_train = data_train[:,:Tft_config.n_steps]
train_params = Tft_config.train_params



with torch.no_grad():
   data_train_vae = reshape_vae(data_train)
   data_train_vae = VAE(data_train_vae)[2]
   data_train = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)



series_list = []    
for i in range(data_train.shape[0]):
    stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [float(train_params[i,1])], "V_bias": [float(train_params[i,2])]})
    series = TimeSeries.from_values(data_train[i].numpy())#, static_covariates=stat_cov)#all training data convertd to time series type
    series_list.append(series)
print("series list 0 ", series_list[0])

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
    likelihood=QuantileRegression(
        quantiles=Tft_config.quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
    force_reset=True,
    pl_trainer_kwargs={
       "accelerator": "gpu",
       "devices": [0],   
    },
)

my_model.fit(series_list, verbose=True)
my_model.save("models/tft_model_lb_no_stat_cov.pt")
