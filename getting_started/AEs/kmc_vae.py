import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import urllib.request as request
import gzip
from cae import *
import torch
from torchvision import datasets, transforms
from train_vae import *
from torch.utils.data import TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42) # Setting the seed

part_class = 3
batch_size=1000
latent_dim = 40
base = 32
x_train = torch.randint(0,2, (2000, part_class, 28, 28)).to(torch.float32)
y_train = torch.argmax(x_train, dim=1)#.to(torch.float32)

train_dataset = TensorDataset(x_train, y_train)


encoder = encoder2d(part_class,base, latent_dim)
decoder = decoder2d(part_class,base, latent_dim)
model = CAE(encoder, decoder, num_hidden=latent_dim)
model.to(device)
# TODO: create data loader with training data and batch size
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, 
                                           shuffle=True)
#TODO: choose loss function and optimizer
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#TODO: train the model
model.train()
train_loop_cae(x_train, model, criterion, optimizer, train_loader, num_epochs=50)
model.cpu() # move the model to the cpu
torch.save(model.state_dict(), 'model_kmc.pth')




    
