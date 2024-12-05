import torch
from data_loading import *
import random

##Script to make train/ test split from avg_trj files


#fix seed
random.seed(42)

data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"

n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000

data = DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z, suffix, run_type="2d_lin_time")
files = data.make_folder_list("2d_lin_time_file_list.txt")

#randomly shuffle contents of files
random.shuffle(files)


#split files into training and test sets with 80 /  20 ratio

train_files = files[:int(len(files)*0.8)]
test_files = files[int(len(files)*0.8):]

# make data sets

train_set = data.make_data_set("train_set_lin_80_20", train_files)
test_set = data.make_data_set("test_set_lin_80_20", test_files)

