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
import argparse
from torch.utils.tensorboard import SummaryWriter
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = "runs/vae_"+args.data_type+"_data"
##Hyperparameters:
part_class = VAE_config.part_class
batch_size= VAE_config.batch_size
num_epochs = VAE_config.num_epochs
model_name = VAE_config.model_name   
parameters = dict(  
    latent_dim = [VAE_config.latent_dim],
    base = [VAE_config.base],
    lr = [ VAE_config.lr]
)
##Data
data_train = VAE_config.data_train_std
data_test = VAE_config.data_test_std
mean = torch.mean(data_train)
##Scale data by mean
data_train = data_train/mean
data_test = data_test/mean

#Use subset of time steps for training
data_train = data_train[:,:VAE_config.n_steps]
data_test = data_test[:,:VAE_config.n_steps]

##combine timesteps and config dim for training of the VAE model
data = torch.reshape(data_train,(data_train.shape[0]*data_train.shape[1],1,50,100))

## Split data into training and validation sets
train_size = int(0.8 * len(data))
val_size = len(data) - train_size

data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
##Corresponds almost to 80/20 split and 
#ensures that the configs are not split within

data_test = (np.reshape(data_test,(data_test.shape[0]*data_test.shape[1],1,50,100)))
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=val_size, shuffle=True)


##Loss function
criterion = nn.MSELoss(reduction='sum')


## hyper parameter loop

for run_id, (latent_dim, base, lr) in enumerate(product(*parameters.values())):

    ## Instantiate the encoder/ decoder layers and the VAE model
    encoder = pro_encoder2d(part_class,base, latent_dim)
    decoder = pro_decoder2d(part_class,base, latent_dim)
    model = VAE(encoder, decoder, latent_dim=latent_dim, mode="train")
    model.to(device)

    ## Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ##number of parameters in the model
    n_params = sum(p.numel() for p in model.parameters())
    print("Number of trainable parameters", n_params)
    comment = f' latent_dim={latent_dim} base={base} lr={lr}'
    tb = SummaryWriter(comment=comment) # tensorboard writer
    model.train()

    ## Train the model
    train_loss, train_KLD, best_model = train_vae_tb(model, criterion, optimizer, train_loader, val_loader,VAE_config, num_epochs=num_epochs, tb=tb)
      
    model.load_state_dict(best_model)
    model.eval()
    model.to(torch.device("cpu"))
    test_loss = 0
    val_loss = 0

    ## Compute the test loss
    for x_test in test_loader:
            # Get the inputs; data is a list of [inputs, labels]
            
            decoded, _,_,_ = model(x_test)
            test_loss += criterion(decoded, x_test)#+ (torch.sum(y_train!=y_pred))
    ## Compute the validation loss
    for x_val in val_loader:
            decoded, _,_,_ = model(x_val)
            val_loss += criterion(decoded, x_val)

    
    test_loss = test_loss/len(test_loader.dataset)
    val_loss = val_loss/len(val_loader.dataset)

    ## Log the train/val/test loss
    tb.add_hparams(
    {"latent_dim": latent_dim, "base": base, "lr": lr},{"Train loss": train_loss, "Train KLD": train_KLD,  "Val loss": val_loss.item()}
    )
tb.close()
## Save the best model
torch.save(model, "models/model_vae_std_final.pt")



