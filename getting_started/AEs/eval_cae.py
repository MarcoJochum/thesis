import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from cae import *
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def show_images(images, labels, figname):
    """
    Display a set of images and their labels using matplotlib.
    The first column of `images` should contain the image indices,
    and the second column should contain the flattened image pixels
    reshaped into 28x28 arrays.
    """
    # Extract the image indices and reshaped pixels
    pixels = images.reshape(-1, 28, 28)
    print(pixels.shape)
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
    plt.savefig(figname)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)

# Normalize the pixel values
X_train = trainset.train_data.to(torch.float32) / 255.0
X_test = testset.test_data.to(torch.float32) / 255.0

# Convert labels to integers
y_train = trainset.train_labels.to(torch.int32)
y_test = testset.test_labels.to(torch.int32)

base  =32

encoder = encoder2d(base_channel_size=base, latent_dim=10)
decoder = decoder2d(base_channel_size=base, latent_dim=10)
model = VAE(encoder, decoder)
model.load_state_dict(torch.load('model_vae.pth'))
model.eval()


X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
encode, decode, _ , _= model(X_train[20:25])
print(decode.shape)
fake = torch.rand_like(X_train[20:25])
_,fake, _ ,_ = model(fake)
plt.figure

z, fake = model.sample(num_samples=5)
show_images(fake.detach().numpy(), y_train[20:25], 'test_decoded_vae2.png')
plt.figure
show_images(X_train[20:25].numpy(), y_train[20:25], 'test_original_vae2.png')

print(count_parameters(model))