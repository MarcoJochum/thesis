from NNs.autoencoder import *
from NNs.ae_layers import * 
from .train import *
from NNs.RNN import *
from .data import *
from .data_loading import *
import submitit

## Execute from the project root

## Data Loading
executor = submitit.AutoExecutor(folder='logs/submitit_logs')
executor.update_parameters(timeout_min=10, slurm_partition="microcloud", mem_gb=16)


data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"

n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000

train_data = DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z, suffix, run_type="2d_lin_time")


jobs = []
config_folder = train_data.kmc_data
with executor.batch():
    for config_folders in os.listdir(config_folder):
        job = executor.submit(train_data.std_trj, config_folders)
        jobs.append(job)
#Combining data sets
#files = train_data.make_folder_list("2d_lin_time_file_list.txt")

#train_data.make_data_set("2d_red_5.npy", files[:-5])
