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
from lib.data import *
from config.liquid import Liquid_config
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_epochs = Liquid_config.n_epochs
lr = Liquid_config.lr
batch_size = Liquid_config.batch_size
latent_dim = Liquid_config.latent_dim
base = Liquid_config.base
part_class = Liquid_config.part_class

units = Liquid_config.units
proj_size = Liquid_config.proj_size
backbone_layers = Liquid_config.backbone_layers
backbone_units = Liquid_config.backbone_units
backbone_dropout = Liquid_config.backbone_dropout

model_name = Liquid_config.model_name
vae_name = "models/model_vae_lin.pth"

##Initialize VAE for encodings
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)

vae = VAE(encoder, decoder, latent_dim=latent_dim)
vae.load_state_dict(torch.load(vae_name, map_location=torch.device(device)))
vae.eval()

## LOad trainig data
x_train = Liquid_config.data_train
x_test = Liquid_config.data_test
x_train = x_train/torch.mean(x_train)
x_test = x_test/torch.mean(x_train)
x_train = x_train[:,:50]   
x_test = x_test[:,:50]
list_x_train = []
list_x_test = []  
print("x_train shape:", x_train.shape)
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
#time = torch.logspace(-8, -4, 1000).unsqueeze(0)
time = torch.linspace(1e-07,1e-04,1000).unsqueeze(0)
#time = time/torch.mean(time)
vae.to(device)


#time = torch.stack([time[200:] for i in range(51)])


##I just choose this like they did in the tutorial so if this is not the optimal setting i can change it
wiring = AutoNCP(latent_dim+10, latent_dim )
model =  ncps.CfC(latent_dim,units=units,proj_size=proj_size, mode="default", batch_first=True,
                   backbone_layers=backbone_layers, backbone_units=backbone_units) 
#unit size is the dimension of the hidden state
model.to(device)
print("Number of parameters in LNN:",sum(p.numel() for p in model.parameters() if p.requires_grad))
n_params= sum(p.numel() for p in model.parameters() if p.requires_grad)



x_train_liq, y_train_liq = make_sequence(x_train_lat, 20,1)
_, y_train_liq = make_sequence(x_train.squeeze(), 20,1)

print("y_train_liq shape:", y_train_liq.shape)
x_train_liq = torch.reshape(x_train_liq, (x_train_liq.shape[0]*x_train_liq.shape[1],x_train_liq.shape[2], x_train_liq.shape[3]))
y_train_liq = torch.reshape(y_train_liq, (y_train_liq.shape[0]*y_train_liq.shape[1],y_train_liq.shape[2], y_train_liq.shape[3], y_train_liq.shape[4]))
#test data
x_test_liq, y_test_liq = make_sequence(x_test_lat, 10,1)
# x_train_liq = x_train_lat[:,:-1]
# y_train_liq = x_train_lat[:,1:]
# x_test_liq = x_test_lat[:,:-1]
# y_test_liq = x_test_lat[:,1:]
x_test_liq = torch.reshape(x_test_liq, (x_test_liq.shape[0]*x_test_liq.shape[1],x_test_liq.shape[2], x_test_liq.shape[3]))
y_test_liq = torch.reshape(y_test_liq, (y_test_liq.shape[0]*y_test_liq.shape[1],y_test_liq.shape[2], y_test_liq.shape[3]))
print("x_train_liq shape:", x_train_liq.shape)  
## This way i predict from the current time step to  the next 
## At least that is my intention
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_data = torch.utils.data.TensorDataset(x_train_liq[:-100], y_train_liq[:-100])
val_data = torch.utils.data.TensorDataset(x_train_liq[-100:], y_train_liq[-100:])
test_data = torch.utils.data.TensorDataset(x_test_liq, y_test_liq)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=x_test_liq.shape[0], shuffle=False)  
train_loader = torch.utils.data.DataLoader(train_data, batch_size=Liquid_config.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=x_train_liq[-100:].shape[0], shuffle=True)


best_loss = 1e8
criterion = torch.nn.MSELoss(reduction="mean")
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=20, verbose=True)
gradient_mean = torch.zeros(n_epochs).to(device)
gradient_std = torch.zeros(n_epochs).to(device)
for i in range(n_epochs):
    model.train()
    total_loss = 0
    total_loss_test = 0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        
        y_pred_lat,hx = model(x)
        
        y_pred_lat = torch.reshape(y_pred_lat, (y_pred_lat.shape[0]* y_pred_lat.shape[1],latent_dim))
        y_pred = vae.decoder(y_pred_lat)
        y_pred = torch.reshape(y_pred, (y.shape[0], y.shape[1],  50, 100))
        
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()/len(train_loader)
    for name, param in model.named_parameters():
        
        gradient_mean[i] +=torch.mean(torch.abs(torch.flatten(param.grad)))
        gradient_std[i] += torch.mean(torch.std(torch.flatten(param.grad)))

    print("Gradient mean:", gradient_mean[i])   
    print("Gradient std:", gradient_std[i])

    val_loss = 0
    for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_pred_lat,_ = model(x_val)
                y_val_pred_lat = torch.reshape(y_val_pred_lat, (y_val_pred_lat.shape[0]* y_val_pred_lat.shape[1],latent_dim))
                y_val_pred = vae.decoder(y_val_pred_lat)
                y_val_pred = torch.reshape(y_val_pred, (y_val.shape[0], y_val.shape[1],  50, 100))


                val_loss += criterion(y_val_pred, y_val)
    print('Validation loss:', val_loss.item()/len(val_loader))
    if val_loss.item()/len(val_loader) < best_loss:
        print('Model updated:', i)
                    
        best_loss = val_loss.item()/len(val_loader)
        best_model=copy.deepcopy(model)
        torch.save(best_model, model_name)

    scheduler.step(val_loss.item()/len(val_loader))
                    
            
    print(f"Epoch {i}, Loss: {total_loss}")
for x_test_liq, y_test_liq in test_loader:
        x_test_liq = x_test_liq.to(device)
        y_test_liq = y_test_liq.to(device)
        y_test_pred,_ = model(x_test_liq)
        
        loss_test = criterion(y_test_liq, y_test_pred)
        total_loss_test += loss_test.item()/len(test_loader)   
print(f"Test Loss: {total_loss_test}")
torch.save(model, "final_model.pth")    