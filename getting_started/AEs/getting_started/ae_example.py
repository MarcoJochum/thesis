import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import urllib.request as request
import gzip
import ae
import torch
from torchvision import datasets, transforms
from cae import *

def show_images(images, labels):
    """
    Display a set of images and their labels using matplotlib.
    The first column of `images` should contain the image indices,
    and the second column should contain the flattened image pixels
    reshaped into 28x28 arrays.
    """
    # Extract the image indices and reshaped pixels
    pixels = images.reshape(-1, 28, 28)

    # Create a figure with subplots for each image
    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(10, 3 * len(images))
    )

    # Loop over the images and display them with their labels
    for i in range(len(images)):
        # Display the image and its label
        axs[i].imshow(pixels[i], cmap="gray")
        axs[i].set_title("Label: {}".format(labels[i]))

        # Remove the tick marks and axis labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("Index: {}".format(i))

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Show the figure
    plt.show()


# Download the files
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Normalize the pixel values
X_train2 = trainset.train_data.to(torch.float32)
X_train = trainset.train_data.to(torch.float32) / 255.0
X_test = testset.test_data.to(torch.float32) / 255.0

# Convert labels to integers
y_train = trainset.train_labels.to(torch.int32)
y_test = testset.test_labels.to(torch.int32)



batch_size = 64 

encoder = encoder()
decoder = decoder()
model = VAE(encoder, decoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(X_train[:10000], batch_size=batch_size, 
                                           shuffle=True)
model.train()
ae.train_loop(X_train[:10000], model, criterion, optimizer, train_loader, num_epochs=200)
torch.save(model.state_dict(), 'model_vae.pth')
model.load_state_dict(torch.load('model_vae.pth'))
model.eval()

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)    
encode, decode, _, _ = model(X_test[20:25])
plt.figure
show_images(decode.detach().numpy(), y_test[20:25])
plt.figure
show_images(X_test[20:25].numpy(), y_test[20:25])