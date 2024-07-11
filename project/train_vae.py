import torch.nn as nn
import numpy as np
import torch
from NNs.autoencoder import *
from lib.train import *
import os

from torch.utils.data import TensorDataset, random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42) # Setting the seed

part_class = 1
batch_size=2000
latent_dim = 40
base = 32
data = torch.tensor(np.load('../../data_kmc/2d_sets/2d_red_5.npy'), dtype=torch.float32)



#combine timesteps and config dim for training of the ae model
data = (np.reshape(data,(data.shape[0]*data.shape[1],1,50,100)))#.squeeze()
x_train, x_test = random_split(data, [int(0.8*data.shape[0]), int(0.2*data.shape[0])], generator=torch.Generator().manual_seed(42))


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
model = VAE(encoder, decoder, latent_dim=latent_dim)
model.to(device)


# TODO: create data loader with training data and batch size
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, 
                                           shuffle=True)
#TODO: choose loss function and optimizer
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#TODO: train the model
model.train()
train_vae(x_train, model, criterion, optimizer, train_loader, num_epochs=1000)
#model.cpu() # move the model to the cpu
torch.save(model.state_dict(), 'model_vae_kmc_red_data_e1000.pth')
model.eval()
model.load_state_dict(torch.load('model_vae_kmc_red_data_e1000.pth'))


test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=True)
loss = 0
for i, data in enumerate(test_loader):
            # Get the inputs; data is a list of [inputs, labels]
            #data = data.unsqueeze(1)
            data = data.to(device)
            decoded, _,_,_ = model(data)
            loss += criterion(decoded, data)#+ (torch.sum(y_train!=y_pred))


loss = loss/(0.2*data.shape[0])  
print('Test loss:', loss.item())