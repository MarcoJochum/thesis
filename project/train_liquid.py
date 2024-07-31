import torch 
import ncps.torch as ncps
from ncps.wirings import AutoNCP  
from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

n_epochs = 500
batch_size = 2000
latent_dim = 50
base = 8
part_class = 1
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
##Initialize VAE for encodings
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)

vae = VAE(encoder, decoder, latent_dim=latent_dim)
vae.load_state_dict(torch.load('model_vae_500_act.pth', map_location=torch.device(device)))
vae.eval()
## LOad trainig data
x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
x_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)
x_train = x_train[:,200:]   
x_test = x_test[:,200:]
list_x_train = []
list_x_test = []  

##Here i use the pretrained AE to encode the data from 51x1000x50x100 to 51x1000x25
with torch.no_grad():
    for x in x_train:
        _, x_lat, _, _ = vae(x)
        list_x_train.append(x_lat)
    for y in x_test: 
        _, y_lat, _ ,_ = vae(y)
        list_x_test.append(y_lat)
x_train_lat =torch.stack(list_x_train)
x_test_lat = torch.stack(list_x_test)
time = torch.logspace(-8, -4, 1000).unsqueeze(0)
#time = time/torch.mean(time)


#time = torch.stack([time[200:] for i in range(51)])

in_features = latent_dim
##I just choose this like they did in the tutorial so if this is not the optimal setting i can change it
wiring = AutoNCP(in_features+10, in_features )
model =  ncps.CfC(in_features,units=60,proj_size=in_features, mode="default", batch_first=True, backbone_layers=3, backbone_units=60, backbone_dropout=0.2) #
#unit size is the dimension of the hidden state
model.to(device)
print("Number of parameters in LNN:",sum(p.numel() for p in model.parameters() if p.requires_grad))
##data dim: 51x1000x10
#take whole trajectory
# y is just one step ahead
#999 training points
#just predict the next one from the current one
#x_train_lat = torch.reshape(x_train_lat, (x_train_lat.shape[0]*x_train_lat.shape[1],
                                #x_train_lat.shape[2]))
#x_train_liq = torch.zeros((51,999,latent_dim))
#y_train_liq = torch.zeros((51,999,latent_dim))
y_train_liq = x_train_lat[:,1:] ##Dim 51x999x10
x_train_liq = x_train_lat[:,:-1] ##Dim 51x999x10

#test data
y_test_liq = x_test_lat[:,1:] ##Dim 51x999x10
x_test_liq = x_test_lat[:,:-1] ##Dim 51x999x10
## This way i predict from the current time step to  the next 
## At least that is my intention
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_data = torch.utils.data.TensorDataset(x_train_liq[:-10], y_train_liq[:-10])
val_data = torch.utils.data.TensorDataset(x_train_liq[-10:], y_test_liq[-10:])
test_data = torch.utils.data.TensorDataset(x_test_liq, y_test_liq)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=False)  
train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, shuffle=False)
best_loss = 1e8
criterion = torch.nn.MSELoss(reduction="mean")
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10, verbose=True)
for i in range(n_epochs):
    model.train()
    total_loss = 0
    total_loss_test = 0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        print("x shape", x.shape)
        
        y_pred,hx = model(x, timespans=time)
        print("hx shape", hx.shape)
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()/len(train_loader)

    val_loss = 0
    for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_pred,_ = model(x_val, timespans=time)
                val_loss += criterion(y_val_pred, y_val)
                print('Validation loss:', val_loss.item()/len(val_loader))
                if val_loss.item()/len(val_loader) < best_loss:
                    print('Model updated:', i)
                    
                    best_loss = val_loss.item()/len(val_loader)
                    best_model=copy.deepcopy(model)
                    torch.save(best_model, 'model_liq_default_1.pth')

    scheduler.step(val_loss.item()/len(val_loader))
                    
            
    print(f"Epoch {i}, Loss: {total_loss}")
for x_test_liq, y_test_liq in test_loader:
        x_test_liq = x_test_liq.to(device)
        y_test_liq = y_test_liq.to(device)
        y_test_pred,_ = model(x_test_liq)
        
        loss_test = criterion(y_test_liq, y_test_pred)
        total_loss_test += loss_test.item()/len(test_loader)   
print(f"Test Loss: {total_loss_test}")

torch.save(best_model, 'model_liq_default_1.pth')