import torch
import torch.nn as nn
from .ae_layers import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CAE(nn.Module):
    def __init__(self, encoder_class: encoder2d, decoder_class: decoder2d, latent_dim=20):
        super().__init__()


        '''

        Convolutional Autoencoder base module.

        Args:

            encoder_class (nn.Module): The encoder class.

            decoder_class (nn.Module): The decoder class.

            latent_dim (int): The dimensionality of the latent space.

        '''
        self.latent_dim = latent_dim
        self.encoder = encoder_class
        self.decoder = decoder_class


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return  decoded, encoded
    

class VAE(CAE):
    def __init__(self, encoder, decoder, latent_dim=20, mode='train'):
        super().__init__(encoder_class= encoder, decoder_class=decoder, latent_dim=latent_dim)

        ''' 
        Variational Autoencoder module based on the CAE module.

        Args:

            encoder (nn.Module): The encoder class.

            decoder (nn.Module): The decoder class.

            latent_dim (int): The dimensionality of the latent space.

        '''
        self.mode = mode
        self.mu = nn.Linear(self.latent_dim, self.latent_dim) #transform the encoded data into the mean of the latent space
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)  #transform the encoded data into the log variance of the latent space

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std
    
    def forward(self,x, params=None):

        encoded = self.encoder(x)
        mu = self.mu(encoded)

        logvar = self.logvar(encoded)

        z = self.reparametrize(mu, logvar)
        ##Do this before decoding
        if not self.mode == 'train':
            z[:,:3] = params
        decoded = self.decoder(z)
        
        return  decoded, z, mu, logvar
    
    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return z, samples
