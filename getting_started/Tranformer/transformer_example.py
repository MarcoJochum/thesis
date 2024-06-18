import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, act_proj= nn.ReLU()):
        
        '''
        nn.Module for transformer Encoder layer
        
        Args:
            d_model     :   (Int) The embedding dimension 
            
            num_heads   :   (Int) The number of heads used in attention module
            
            d_ff        :   (Int) Projection dimension used in Feed-Forward network
            
            dropout     :   (Float) The dropout value to prevent from overfitting
            
            act_proj    :   (Str)   The activation function used in the FFD
            
        '''
        
       
        super(EncoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.activation = act_proj
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            self.activation,
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        ##  Self-Attention  ##
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ##  MLP ##
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.0, act_proj= nn.ReLU()):
        '''
        nn.Module for transformer Encoder
        
        Args:
            num_layers  :   (Int) The number of encoder layers
            
            d_model     :   (Int) The embedding dimension

            num_heads   :   (Int) The number of heads used in attention module

            d_ff        :   (Int) Projection dimension used in Feed-Forward network

            dropout     :   (Float) The dropout value to prevent from overfitting

            act_proj    :   (Str)   The activation function used in the FFD

            
        '''
        
        super(TransformerEncoder, self).__init__()


        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, 
                                                  dropout=0.0, act_proj= nn.ReLU()) for _ in range(num_layers)])
        
        ##initialize embedding layer here
        
    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)
        return x