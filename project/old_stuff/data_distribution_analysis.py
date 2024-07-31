from NNs.autoencoder import *
from NNs.ae_layers import * 
from lib.train import *
from NNs.RNN import *
from lib.data import *
from lib.data_loading import *
from torch.utils.data import TensorDataset, random_split
data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"
import random
random.seed(42)
n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000
latent_dim = 40
part_class = 1  
base = 32
batch_size = 2000
## Creating datasets for the LSTM model
data = DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z, suffix)

###THis could be done in the data class with new function to split data
file_list = data.make_folder_list()
random.shuffle(file_list)
split = int(0.8*len(file_list))
train_list = file_list[:split]
test_list = file_list[split:]
print("test list:", len(test_list))
print("Train list:", len(train_list))
train_data = data.make_data_set(file_name="train_set_80_20",list_configs=train_list)
test_data = data.make_data_set(file_name="test_set_80_20", list_configs=test_list)


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
model = VAE(encoder, decoder, latent_dim=latent_dim)
model.load_state_dict(torch.load('models/model_vae_kmc_red_data_e1000.pth', map_location=torch.device('cpu')))

Encodings = []
for config in train_data:

    encoding = model.encoder(config)
    Encodings.append(encoding)
Encodings =torch.stack(Encodings)
print("Encodings shape:", Encodings.shape)

### Data distribution analysis
x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
x_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)


x_train = (np.reshape(x_train,(x_train.shape[0]*x_train.shape[1],1,50,100)))
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)

x_test = (np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],1,50,100)))
x_test = torch.tensor(x_test, device=device)
test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=True)
x_train_flattened = x_train.flatten()

# Create the histogram
plt.hist(x_train_flattened, bins=100, alpha=0.75, log=True)

plt.title('Histogram of x_train values')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.savefig("data_distr.png")
##Normalize input data
# Count the number of zeros
num_zeros = np.count_nonzero(x_train == 0)
print("Number of zeros in x_train", num_zeros)
print("Befor normalizatio: Xtrain mean:", x_train.mean(), "Xtrain std:", x_train.std())
print("xtrain max, xtrain min", x_train.max(), x_train.min())
x_train, x_train_max, x_train_min = normalization(x_train)


print("Xtrain mean:", x_train.mean(), "Xtrain std:", x_train.std())
print("xtrain max, xtrain min", x_train.max(), x_train.min())   
