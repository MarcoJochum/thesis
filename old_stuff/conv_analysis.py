import matplotlib.pyplot as plt
import numpy as np
#from data_loading import load_data
import torch
import pandas as pd 

file_path = "data_kmc/2d/80_1e+19_2.0/prod_1/0.2V/average_density_li+.txt"

data = pd.read_csv(file_path, sep="\s+", header=None)

    # Convert all data to numeric, replacing non-numeric values with NaN
    #Replacing initial 3 NAN values with Nan
data = data.apply(pd.to_numeric, errors='coerce')
    

data = torch.tensor(data.values)

print(data.shape)
time = data[1:,0]
labels = []

# for i in range(1, 50,5):
#     plt.plot(data[0, 1:], data[i, 1:]/1e19)
#     labels.append([f"t = {time[i-1]}"])
#     plt.yrange = (80, 100)
# plt.legend(labels)

spaceing = np.logspace(-8, -4, 1000, 'o')
density = np.zeros(1000)
plt.scatter(spaceing, density)
plt.savefig("density.png")