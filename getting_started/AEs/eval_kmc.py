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
batch_size=300
latent_dim = 40 
base = 32
x_train = torch.randint(0,2, (1000, part_class, 28, 28)).to(torch.float32)
y_train = torch.argmax(x_train, dim=1)#.to(torch.float32)

train_dataset = TensorDataset(x_train, y_train)


encoder = encoder2d(part_class,base, latent_dim)
decoder = decoder2d(part_class,base, latent_dim)
model = CAE(encoder, decoder, num_hidden=latent_dim)

model.load_state_dict(torch.load('model_kmc.pth'))
model.eval()

x,y = torch.where(y_train[0,...]==1)
#x_train = x_train[1,...].unsqueeze(0)

_,x_decod= model(x_train[:1, ...])
x_decod = x_decod.squeeze(0)
x_decod = torch.softmax(x_decod, dim=0)
y_pred = torch.argmax(x_decod, dim=0)
z,w = torch.where(y_pred==1)


plt.figure()
plt.scatter(x,y)
plt.scatter(z,w)
plt.show()

plt.figure()
plt.sca