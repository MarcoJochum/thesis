import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # input channel size: 1, output channel size: 16
            nn.ReLU(),  # apply the ReLU activation function
            #nn.MaxPool2d(2, stride=2),  # apply max pooling with a kernel size of 2 and a stride of 2
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # input channel size: 16, output channel size: 4
            nn.ReLU(),  # apply the ReLU activation function
            #nn.MaxPool2d(kernel_size=2, stride=2)  # apply max pooling with a kernel size of 2 and a stride of 2
            torch.nn.Flatten(),
            torch.nn.Linear((8*7*7), 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class decoder(nn.Module): 
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(10, 8*7*7),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),  # input channel size: 8, output channel size: 16
            #self.upsample(),
            nn.ReLU(),# apply the ReLU activation function
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # input channel size: 16, output channel size: 8
            nn.Sigmoid()  # apply the sigmoid activation function to compress the output to a range of (0, 1)
            #could be because of normalized inputs
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.shape[0], 8, 7, 7))
        x = self.net(x)
        return x


class CAE(nn.Module):
    def __init__(self, encoder_class: encoder, decoder_class: decoder):
        super().__init__()
        self.num_hidden = 10
        self.encoder = encoder_class
        self.decoder = decoder_class


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class VAE(CAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder_class= encoder, decoder_class=decoder)

        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.logvar = nn.Linear(self.num_hidden, self.num_hidden)  

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std
    
    def forward(self,x):

        encoded = self.encoder(x)
        mu = self.mu(encoded)

        logvar = self.logvar(encoded)

        z = self.reparametrize(mu, logvar)
        
        decoded = self.decoder(z)

        return encoded, decoded, mu, logvar
    
    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.num_hidden)
            samples = self.decoder(z)
        return z, samples