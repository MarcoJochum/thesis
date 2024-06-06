import torch
import torch.nn as nn   
from NNs.autoencoder import *
from NNs.ae_layers import * 
from lib.train import *
from NNs.RNN import *
from lib.data import *

data = torch.randn(100, 1)
#data = nn.ConstantPad2d((1,1,1,1), 0)(data)
train_loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)


##Model setup:
X,Y = make_Sequence(data, 10, 1)
print(X[0], Y[0])
model = LSTMs(2, 2, 16, 1, 2, embed = "sin")


##Training

