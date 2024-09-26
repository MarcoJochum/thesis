import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape, smape
from darts.utils.likelihood_models import QuantileRegression
from config.tft import Tft_config
from NNs.autoencoder import *
from lib.helper import *
import warnings
import sys
import optuna
import pickle

from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
import logging

def metric_comp(preds, val_series, pred_length, vae):
    val_pred_tens= torch.tensor([preds[i].all_values() for i in range(len(preds))])
    val_pred_tens = torch.mean(val_pred_tens, -1)
    val_pred_tens = reshape_vae(val_pred_tens)
    with torch.no_grad():
        val_pred_tens = vae.decoder(val_pred_tens)
    val_pred_tens = unshape_vae(val_pred_tens, len(val_series), pred_length, False)
    mape_list = []
    for i in range(len(val_series)):
        mape = s_mape(data_train[Tft_config.train_size+i,Tft_config.seq_length:pred_length].detach().numpy(),
         val_pred_tens[i])
        mape_list.append(mape)
    mape_list = np.array(mape_list)
    mape_mean = np.mean(mape_list)
    return mape_mean 

##Workaround because optuna has problem with lightning and pytorch_lightning callbacks
class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




vae= torch.load("models/model_vae_lin_lat_10.pth", map_location=torch.device("cpu"))

data_train = Tft_config.data_train
##Rescaling the data, setting mean=1
data_train = data_train/ torch.mean(data_train)
data_train = data_train[:,:Tft_config.n_steps]
train_params = Tft_config.train_params



with torch.no_grad():
   data_train_vae = reshape_vae(data_train)
   data_train_vae = vae(data_train_vae)[2] ## Encoding the data into the latent space
   data_train_encoded = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)



series_list = []    
for i in range(data_train_encoded.shape[0]):
    stat_cov = pd.DataFrame(data ={"epsr": [float(train_params[i,0])], "conc": [float(train_params[i,1])], "V_bias": [float(train_params[i,2])]})
    series = TimeSeries.from_values(data_train_encoded[i].numpy(), static_covariates=stat_cov)#all training data convertd to time series type
    series_list.append(series)




train_series = series_list[:Tft_config.train_size]
val_series = series_list[Tft_config.train_size:]
def objective(trial):
    

    #in_len = trial.suggest_int("in_len", 3, 15)
    out_len = trial.suggest_int("out_len", 30, 80)
    hidden = trial.suggest_int("hidden", 40, 100)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    num_attention_heads = trial.suggest_int("num_attention_heads", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    
    callbacks = [OptunaPruning(pruner, early_stopper)]

    my_model = TFTModel(
    input_chunk_length=Tft_config.seq_length,
    output_chunk_length=out_len,
    hidden_size=hidden,
    lstm_layers=lstm_layers,
    num_attention_heads=num_attention_heads,
    dropout=dropout,
    batch_size=Tft_config.batch_size,
    n_epochs=Tft_config.n_epochs,
    add_relative_index=True, #this needs to be true becaue i do not have covariant data
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=Tft_config.quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
    optimizer_kwargs={"lr": lr},
    force_reset=True,
    save_checkpoints=True,
    model_name="tft_model",
    pl_trainer_kwargs={
       "accelerator": "gpu",
       "devices": [0],  
       "callbacks":callbacks,
    },)

    my_model.fit(series=train_series, val_series=val_series, verbose=True)
    my_model.load_from_checkpoint("tft_model")
    preds = []
    val_series_eval = []   
    ##Just use the initial part of the validation series
    for i in range(len(val_series)):
        dummy = val_series[i][:Tft_config.seq_length]    
        val_series_eval.append(dummy)
    
    pred_length = Tft_config.n_steps - Tft_config.seq_length
    preds = my_model.predict(series=val_series_eval, n=pred_length, num_samples=Tft_config.n_samples_inference)
    #mape_mean = metric_comp(preds, val_series, pred_length, vae)
    smapes = smape(val_series, preds, n_jobs=-1, verbose=False)
    smape_val = np.mean(smapes)
    
    return smape_val if smape_val != np.inf else float("inf")



def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "optuna_studies/tft_study_4/tft_study_4"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)


study = optuna.create_study(study_name=study_name, storage=storage_name,direction="minimize")

study.optimize(objective, n_trials=50, callbacks=[print_callback])

# We could also have used a command as follows to limit the number of trials instead:
# study.optimize(objective, n_trials=100, callbacks=[print_callback])

# Finally, print the best value and best hyperparameters:
print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")



# Save the sampler with pickle to be loaded later.
with open("optuna_studies/tft_study_2/sampler.pkl", "wb") as fout:
    pickle.dump(study.sampler, fout)