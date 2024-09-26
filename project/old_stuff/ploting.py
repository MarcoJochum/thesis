
from NNs.autoencoder import *
from NNs.ae_layers import * 
from lib.train import *
from NNs.RNN import *
from lib.data import *
from lib.data_loading import *
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from config.vae import VAE_config
from config.tft import Tft_config
import torch
data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"
import random
from config.seq import Seq_config
batch_vec = np.linspace(1, 100, 1)
import pickle
import matplotlib.pyplot as plt
import numpy as np


val_loss = np.load("models/study_4/study_4_opt_qr_val_loss.npy")
train_loss = np.load("models/study_4/study_4_opt_qr_train_loss.npy")

plt.plot(val_loss[1:], label="Validation loss")
plt.plot(train_loss, label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("tft_loss.png")

# config_study_4_opt = Tft_config()
# with open("test_config.pkl", "wb") as f:
#     pickle.dump(config_study_4_opt, f)

# with open("test_config.pkl", "rb") as f:
#     config = pickle.load(f)

