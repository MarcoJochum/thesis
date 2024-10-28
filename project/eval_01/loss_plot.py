
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

batch_vec = np.linspace(1, 100, 1)
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Use LaTeX for text rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
train_loss_avg = np.load("models/study_5/study_5_opt_qr_train_loss_avg.npy")
val_loss_avg = np.load("models/study_5/study_5_opt_qr_val_loss_avg.npy")
val_loss_std = np.load("models/study_5/study_5_opt_qr_val_loss_std.npy")
train_loss_std = np.load("models/study_5/study_5_opt_qr_train_loss_std.npy")

plt.plot(val_loss_avg[1:], label="Validation loss for average", linestyle='-', color='blue')
plt.plot(train_loss_avg, label="Training loss for average", linestyle='--', color='blue')
plt.plot(val_loss_std[1:], label="Validation loss for std", linestyle='-', color='red')
plt.plot(train_loss_std, label="Training loss for std", linestyle='--', color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(-0.2, 6)
plt.xlim(-2, 50)
plt.legend()
plt.grid(True, "both")
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig("fig_report/tft_loss.png")

# config_study_4_opt = Tft_config()
# with open("test_config.pkl", "wb") as f:
#     pickle.dump(config_study_4_opt, f)

# with open("test_config.pkl", "rb") as f:
#     config = pickle.load(f)

