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
log_dir = "runs/vae_std_data"
##Hyperparameters:
part_class = VAE_config.part_class
batch_size= VAE_config.batch_size
num_epochs = VAE_config.num_epochs
model_name = VAE_config.model_name   
parameters = dict(  
    latent_dim = [10],
    base = [4],
    lr = [ 0.0005]
)
##Data
data_train = VAE_config.data_train_std
data_test = VAE_config.data_test_std
mean = torch.mean(data_train)
##Scale data by mean
data_train = data_train/mean
data_test = data_test/mean

#Use subset of data for training
data_train = data_train[:,:VAE_config.n_steps]
data_test = data_test[:,:VAE_config.n_steps]

time = torch.linspace(1e-07,1e-04,1000)
##combine timesteps and config dim for training of the ae model
data = torch.reshape(data_train,(data_train.shape[0]*data_train.shape[1],1,50,100))

train_size = int(0.8 * len(data))
val_size = len(data) - train_size
breakpoint()
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
##Corresponds almost to 80/20 split and 
#ensures that the configs are not split within

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=val_size, shuffle=True)



time_test = torch.repeat_interleave(time, data_test.shape[0], dim=0) #repeat for each config --> 13 in training

data_test = (np.reshape(data_test,(data_test.shape[0]*data_test.shape[1],1,50,100)))
#data_test  =torch.utils.data.TensorDataset(data_test)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)


##Create model


criterion = nn.MSELoss(reduction='sum')


## hyper parameter loop

for run_id, (latent_dim, base, lr) in enumerate(product(*parameters.values())):
    encoder = pro_encoder2d(part_class,base, latent_dim)
    decoder = pro_decoder2d(part_class,base, latent_dim)
    model = VAE(encoder, decoder, latent_dim=latent_dim, mode="train")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ##number of parameters in the model
    n_params = sum(p.numel() for p in model.parameters())
    print("Number of trainable parameters", n_params)
    comment = f' latent_dim={latent_dim} base={base} lr={lr}'
    tb = SummaryWriter(comment=comment) # tensorboard writer
    model.train()
    train_loss, train_KLD, best_model = train_vae_tb(model, criterion, optimizer, train_loader, val_loader,VAE_config, num_epochs=num_epochs, tb=tb)
     # move the model to the cpu
    
    model.load_state_dict(best_model)
    model.eval()
    model.to(torch.device("cpu"))
    test_loss = 0
    val_loss = 0
    for x_test in test_loader:
            # Get the inputs; data is a list of [inputs, labels]
            #data = data.unsqueeze(1)
            #x_test = data.to(x_test)
            decoded, _,_,_ = model(x_test)
            test_loss += criterion(decoded, x_test)#+ (torch.sum(y_train!=y_pred))
    for x_val in val_loader:
            decoded, _,_,_ = model(x_val)
            val_loss += criterion(decoded, x_val)
    test_loss = test_loss/len(test_loader.dataset)
    val_loss = val_loss/len(val_loader.dataset)
    tb.add_hparams(
    {"latent_dim": latent_dim, "base": base, "lr": lr},{"Train loss": train_loss, "Train KLD": train_KLD,  "Val loss": val_loss.item()}
    )
tb.close()
torch.save(model, "models/model_vae_std_final.pt")



