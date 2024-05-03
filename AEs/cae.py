import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class encoder2d(nn.Module):
    def __init__(self,num_input_channels=1, base_channel_size=16, latent_dim=10):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(), 
            #nn.MaxPool2d(2, stride=2),  # apply max pooling with a kernel size of 2 and a stride of 2
            nn.Conv2d(c_hid, 2*c_hid, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),  
            #nn.MaxPool2d(kernel_size=2, stride=2)  # apply max pooling with a kernel size of 2 and a stride of 2
            torch.nn.Flatten(), 
            torch.nn.Linear((2*c_hid*7*7), latent_dim) # 
        )

    def forward(self, x):
        x = self.net(x)
        return x

class decoder2d(nn.Module): 
    def __init__(self,num_input_channels=1 ,base_channel_size=16, latent_dim=10):
        super().__init__()
        c_hid = base_channel_size

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*c_hid*7*7),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, c_hid, 3, stride=2, padding=1, output_padding=1),  #7x7 -> 14x14 
            
            nn.ReLU(),# apply the ReLU activation function
            nn.ConvTranspose2d(c_hid, num_input_channels, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  #because of normalized inputs
            
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.shape[0], -1, 7, 7))
        x = self.net(x)
        return x


class CAE(nn.Module):
    def __init__(self, encoder_class: encoder2d, decoder_class: decoder2d, num_hidden=10):
        super().__init__()
        self.num_hidden = num_hidden
        self.encoder = encoder_class
        self.decoder = decoder_class


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class VAE(CAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder_class= encoder, decoder_class=decoder)

        self.mu = nn.Linear(self.num_hidden, self.num_hidden) #transform the encoded data into the mean of the latent space
        self.logvar = nn.Linear(self.num_hidden, self.num_hidden)  #transform the encoded data into the log variance of the latent space

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
            z = torch.randn(num_samples, self.num_hidden).to(device)
            samples = self.decoder(z)
        return z, samples