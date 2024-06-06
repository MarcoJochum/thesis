
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import urllib.request as request
import gzip
from cae import *
import torch
from torchvision import datasets, transforms
from train_vae import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

torch.manual_seed(42) # Setting the seed

# Download the files
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Normalize the pixel values
print( trainset.train_data.max())
X_train = trainset.train_data.to(torch.float32) / 255.0
X_test = testset.test_data.to(torch.float32) / 255.0

# Convert labels to integers
y_train = trainset.train_labels.to(torch.int32)
y_test = testset.test_labels.to(torch.int32)


X_train.to(device)
X_test.to(device)
y_train.to(device)
y_test.to(device)

batch_size = 800 
base = 32
encoder = encoder2d(base_channel_size=base, latent_dim=10)
decoder = decoder2d(base_channel_size=base, latent_dim=10)
model = VAE(encoder, decoder)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='sum')

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, 
                                           shuffle=True)
model.train()
train_loop(X_train, model, criterion, optimizer, train_loader, num_epochs=800)
model.cpu() # move the model to the cpu
torch.save(model.state_dict(), 'model_vae.pth')
