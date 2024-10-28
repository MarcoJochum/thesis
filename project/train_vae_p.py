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
from config.vae import VAE_config
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
part_class = VAE_config.part_class
batch_size= VAE_config.batch_size

num_epochs = VAE_config.num_epochs

parameters = dict(  
    latent_dim = [VAE_config.latent_dim],
    base = [VAE_config.base],
    lr = [VAE_config.lr]
    
)
model_name = VAE_config.model_name
##Data
data_train = VAE_config.data_train
data_test = VAE_config.data_test

mean = torch.mean(data_train)
##Scale data by mean
data_train = data_train/mean
data_test = data_test/mean
#Use subset of data for training
n_steps = 500
data_train = data_train[:,:n_steps]
data_test = data_test[:,:n_steps]

time = torch.linspace(1e-07,1e-04,1000)
time = time[:n_steps]
##combine timesteps and config dim for training of the ae model
data = torch.reshape(data_train,(data_train.shape[0]*data_train.shape[1],1,50,100))



train_params = VAE_config.train_params
test_params = VAE_config.test_params
##Rescale parameters with the mean
params_scale = torch.mean(train_params, dim=0)
train_params = train_params/params_scale
train_params = torch.repeat_interleave(train_params, n_steps, dim=0)
time_train = torch.repeat_interleave(time, data_train.shape[0], dim=0) #repeat for each config --> 51 in training

data  =torch.utils.data.TensorDataset(data, train_params, time_train)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])##Corresponds almost to 80/20 split and 
#ensures that the configs are not split within

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=val_size, shuffle=True)

time_test = torch.repeat_interleave(time, data_test.shape[0], dim=0) #repeat for each config --> 13 in training

test_params = torch.repeat_interleave(test_params, n_steps, dim=0)
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
    train_loss, train_KLD, best_model = train_vae_tb(model, criterion, optimizer, train_loader, val_loader, VAE_config, num_epochs=num_epochs, tb=tb)
    model.cpu() # move the model to the cpu
    torch.save(best_model, model_name)
    model.eval()
    #model.load_state_dict(torch.load('model_vae_lin_p.pth', map_location=device))
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



