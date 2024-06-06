import torch
import torch.nn as nn 
from torch.nn import Module

######################################################
#### Time2Vec architecture
######################################################

def t2v(tau, f, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)

    
    v2 = torch.matmul(tau, w0) + b0
  
    return torch.cat([v1, v2], -1)
class SineActivation(Module):
    def __init__(self, in_features,nmodes, out_features):
        super(SineActivation, self).__init__()
       
        self.w0 = nn.parameter.Parameter(torch.randn(in_features,nmodes))
        self.b0 = nn.parameter.Parameter(torch.randn(nmodes))
        
        self.w = nn.parameter.Parameter(torch.randn(in_features,out_features-nmodes))
        self.b = nn.parameter.Parameter(torch.randn(out_features-nmodes))
        
        self.f = torch.sin

        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w)
    

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)

class CosineActivation(Module):
    def __init__(self, in_features,nmodes, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features,nmodes))
        self.b0 = nn.parameter.Parameter(torch.randn(nmodes))
        
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-nmodes))
        self.b = nn.parameter.Parameter(torch.randn(out_features-nmodes))
        
        self.f = torch.cos
    
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w)

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)



