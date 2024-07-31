import torch.nn as nn
import numpy as np
import torch
import torch.utils
from NNs.autoencoder import *
from lib.train import *
from lib.helper import *
import random
from itertools import product
from lib.data_loading import *
from torch.utils.tensorboard import SummaryWriter
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##Hyperparameters:
part_class = 1
batch_size=4000
latent_dim = 10
base = 8
num_epochs = 1000

parameters = dict(  
    latent_dim = [  50],
    base = [8],
    lr = [0.001]
    
)
##Data
data_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
data_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)


time = torch.logspace(-8, -4, 1000)
##combine timesteps and config dim for training of the ae model
data = torch.reshape(data_train,(data_train.shape[0]*data_train.shape[1],1,50,100))
##Normalize data
data = data/torch.mean(data)

train_params = get_config("../../data_kmc/2d_sets/train_set_80_20_list.txt").to(device)##epsr, cbulk, v_bias
##Rescale parameters with the mean
params_scale = torch.mean(train_params, dim=0)
train_params = train_params/params_scale


train_params = torch.repeat_interleave(train_params, 1000, dim=0)
time_train = torch.repeat_interleave(time, 51, dim=0) #repeat for each config --> 51 in training
data  =torch.utils.data.TensorDataset(data, train_params, time_train)

data_train, data_val = torch.utils.data.random_split((data), [40000, 11000])##Corresponds almost to 80/20 split and 
#ensures that the configs are not split within

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=10200, shuffle=True)



time_test = torch.repeat_interleave(time, 13, dim=0) #repeat for each config --> 13 in training
test_params = get_config("../../data_kmc/2d_sets/test_set_80_20_list.txt")
test_params = torch.repeat_interleave(test_params, 1000, dim=0)
data_test = (np.reshape(data_test,(data_test.shape[0]*data_test.shape[1],1,50,100)))
data_test  =torch.utils.data.TensorDataset(data_test, test_params, time_test)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)


##Create model


criterion = nn.MSELoss(reduction='sum')


## hyper parameter loop

for run_id, (latent_dim, base, lr) in enumerate(product(*parameters.values())):
    encoder = pro_encoder2d(part_class,base, latent_dim)
    decoder = pro_decoder2d(part_class,base, latent_dim)
    model = VAE(encoder, decoder, latent_dim=latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    comment = f' latent_dim={latent_dim} base={base} lr={lr}'
    tb = SummaryWriter(comment=comment) # tensorboard writer
    model.train()
    train_loss, train_KLD, best_model = train_vae_tb(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs, tb=tb)
    model.cpu() # move the model to the cpu
    torch.save(best_model, 'model_vae_p.pth')
    model.eval()
    model.load_state_dict(torch.load('model_vae_p.pth', map_location=device))
    test_loss = 0
    for x_test, params, time in test_loader:
            # Get the inputs; data is a list of [inputs, labels]
            #data = data.unsqueeze(1)
            #x_test = data.to(x_test)
            decoded, _,_,_ = model(x_test)
            test_loss += criterion(decoded, x_test)#+ (torch.sum(y_train!=y_pred))
    
    test_loss = test_loss/len(test_loader.dataset)
    print("test loss", test_loss.item())
    #tb.add_hparams(
    #{"latent_dim": latent_dim, "base": base, "lr": lr},{"Train loss": train_loss, "Train KLD": train_KLD,  "Test loss": test_loss.item()}
    #)
tb.close()



