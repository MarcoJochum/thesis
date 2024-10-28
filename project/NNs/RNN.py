import torch
import torch.nn as nn
from .Embedding import *

class LSTMs(nn.Module):
    def __init__(self,  latent_dim,
                        d_model,
                        n_mode,
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


        self.fc = nn.Linear(hidden_size, latent_dim)


        ## Initialize the weights

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        if embed == "sin":
            self.embed = SineActivation(n_mode,n_mode,d_model)
        if embed == "cos":
            self.embed = CosineActivation(latent_dim,latent_dim,d_model)
        if embed == None:
            try: d_model == latent_dim
            except: print("INFO: NO Embedding! \nThe input dimension should be same as the d_model")
            self.embed = None
        
    def forward(self, x, hx=None, cx=None, mode="train"):
        
        if self.embed:
            t = self.embed(t)
            t = t.unsqueeze(0)
        #print("X shape:", x.shape)
        #print("T shape:", t.shape)
        #x = torch.cat([x,t],dim=-1)
        #TODO: Are we resetting the hidden state for each batch? IS this correct?
        
        hidden, cell =   self.init_hidden(x.shape[0],device=x.device)
        
        x, (hn,cn) = self.lstm(x,(hidden.detach(),cell.detach()))
        

        
        # 
        #For training I think I want all predictions 
        #out = self.fc(out[:, -1, :])
        out = self.fc(x) #Going from hidden dim to latent dim
        return out#, h, c        
    
    def init_hidden(self,batch_size,device):
        hidden = torch.zeros(self.num_layers,
                                batch_size,
                                self.hidden_size).to(device)
                    
        cell  =  torch.zeros(self.num_layers,
                                batch_size,
                                self.hidden_size).to(device) 
                    
                    
        return hidden, cell
    
class latLNN(nn.Module):
    def __init__(self, VAE, LNN):
        super(latLNN, self).__init__()
        self.VAE = VAE
        self.LNN = LNN


    def forward(self, x, hx, t):
        ##encode with vae
        
        configs= x.shape[0]
        seq=x.shape[1] 
        x_hat = torch.reshape(x, (x.shape[0]*x.shape[1],1,x.shape[3],x.shape[4]))
        breakpoint()
        _,z,mu,log_var = self.VAE(x_hat)
        
        z = torch.reshape(z, (configs,seq,-1))
        
        if hx is None:
            y_lat,hx = self.LNN(z, timespans=t) #predict next time step
        else:
            y_lat,hx = self.LNN(z, hx, timespans=t)
        y_lat = torch.reshape(y_lat, (configs*seq,-1))
        
        y_hat = self.VAE.decoder(y_lat)
        y = torch.reshape(y_hat, (configs,seq,1,x.shape[3],x.shape[4]))
        
        return y,hx, mu, log_var

class Encoder_lstm(nn.Module):
    def __init__(self, latent_dim, hidden, num_layers=1):
        super(Encoder_lstm, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden, num_layers, batch_first=True)



    def forward(self, x):
        output, hidden = self.lstm(x)

        return output, hidden

class Decoder_lstm(nn.Module):
    def __init__(self, latent_dim, hidden, num_layers=1):
        super(Decoder_lstm, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, latent_dim)

    def forward(self, initial_input, encoder_outputs, hidden, targets, tf_prob):

        decoder_seq_length = targets.shape[1]
        batch_size = targets.shape[0]
        batch_vec = torch.linspace(0, batch_size-1, 1)
        outputs = torch.zeros(targets.shape).to(targets.device).unsqueeze(1)
        input = initial_input.unsqueeze(1)

        for t in range(decoder_seq_length):
            #hidden[0] = hidden[0].permute(1,0,2)
            #hidden[1] = hidden[1].permute(1,0,2)
            output, hidden = self.lstm(input, (hidden[0], hidden[1]))
            output = self.fc(output)
            outputs[:, :,t] = output
            ###Need create target tensor that is also encoded with the VAE and matches the latent dim
            ##need to preserve the time dimension which should just be one
            input = output if torch.rand(1) > tf_prob else targets[:, t].unsqueeze(1)
        
        return outputs
    def evaluate(self, initial_input, encoder_outputs, hidden):
        decoder_seq_length = 1 
        
        output, hidden = self.lstm(initial_input.unsqueeze(1),  (hidden[0], hidden[1]))

        output = self.fc(output)
        return output
        
    
       

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, encoder_inputs, targets,  tf_prob):
        encoder_outputs, hidden = self.encoder(encoder_inputs)
        outputs = self.decoder(encoder_inputs[:,-1], encoder_outputs, hidden, targets, tf_prob)
        return  outputs
    def evaluate(self, encoder_inputs):
        encoder_outputs, hidden = self.encoder(encoder_inputs)
        outputs = self.decoder.evaluate(encoder_inputs[:,-1], encoder_outputs, hidden)
        return outputs