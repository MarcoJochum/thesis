import torch
import torch.nn as nn
from .Embedding import *

class LSTMs(nn.Module):
    def __init__(self,  latent_dim,
                        d_model,
                        hidden_size,
                        num_layers,
                        num_classes,
                        embed=None):
        
        '''
        Module for Long Short-term Memory architecture
        
        Args:
            latent_dim (int): The input size of the LSTM.
            
            d_model (int): The projection dimension.

            hidden_size (int): The number of features in the hidden state h.

            num_layers (int): Number of recurrent layers.
            
            num_classes (int): Number of classes.
        '''
        
        
        super(LSTMs, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True)


        self.fc = nn.Linear(hidden_size, num_classes)


        ## Initialize the weights

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        if embed == "sin":
            self.embed = SineActivation(latent_dim,latent_dim,d_model)
        if embed == "cos":
            self.embed = CosineActivation(latent_dim,latent_dim,d_model)

    def forward(self, x):
        if self.embed:
            x = self.embed(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        #For training I think I want all predictions 
        #out = self.fc(out[:, -1, :])
        return out        
        