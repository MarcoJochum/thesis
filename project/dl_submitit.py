from NNs.autoencoder import *
from NNs.ae_layers import * 
from lib.train import *
from NNs.RNN import *
from lib.data import *
from lib.data_loading import *
import submitit
## Data Loading
executor = submitit.AutoExecutor(folder='logs/submitit_logs')
executor.update_parameters(timeout_min=10, slurm_partition="carlos")


data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"

n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000

train_data = DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z, suffix, run_type="2d/epsr_100")


jobs = []
config_folder = train_data.kmc_data
with executor.batch():
    for config_folder in os.listdir(config_folder):
        job = executor.submit(train_data.avg_trj, config_folder)
        jobs.append(job)
#Combining data sets
files = train_data.make_folder_list("file_list.txt")

#train_data.make_data_set("2d_red_5.npy", files[:-5])
