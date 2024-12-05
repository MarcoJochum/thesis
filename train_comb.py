from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
import ncps.torch as ncps
from ncps.wirings import AutoNCP 
from lib.train import *
from lib.data import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from config.comb import *
import random 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)
model_name = Comb_config.model_name
###
n_epochs = Comb_config.n_epochs
batch_size = Comb_config.batch_size
KLD_weight = Comb_config.KLD_weight
lr = Comb_config.lr 
#VAE params
latent_dim = Comb_config.latent_dim
base = Comb_config.base
part_class = Comb_config.part_class
#LNN params
units = Comb_config.units
backbone_layers = Comb_config.backbone_layers
backbone_units = Comb_config.backbone_units
backbone_dropout = Comb_config.backbone_dropout

encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
VAE = VAE(encoder, decoder, latent_dim=latent_dim)
 
if Comb_config.pretrained:
    VAE.load_state_dict(torch.load(Comb_config.VAE_model_pretrained, map_location=torch.device("cpu")))

LNN =  ncps.CfC(latent_dim,units=units,proj_size=latent_dim, mode="default", batch_first=True, backbone_layers=backbone_layers, 
                backbone_units=backbone_units) 
LNN.to(device)


VAE.to(device)
vLNN = latLNN(VAE, LNN)
vLNN.to(device)

x_train = Comb_config.data_train
x_test = Comb_config.data_test

y_train = x_train[:, 1:51]
x_train = x_train[:, :50]

y_test = x_test[:, 1:51]	
x_test = x_test[:, :50]

time = torch.linspace(1e-07,1e-04,1000).unsqueeze(0)

train_data = torch.utils.data.TensorDataset(x_train[:-10], y_train[:-10])
val_data = torch.utils.data.TensorDataset(x_train[-10:], y_train[-10:])
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)  
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


print("Number of parameters in combined Model:",sum(p.numel() for p in vLNN.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(vLNN.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='mean')
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=20)

best_model = train_comb(vLNN, criterion, optimizer,  train_loader, val_loader, time, KLD_weight, n_epochs, scheduler)
torch.save(vLNN, model_name)