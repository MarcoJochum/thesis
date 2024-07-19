import torch 
import ncps.torch as ncps
from ncps.wirings import AutoNCP  
from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
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
vae.load_state_dict(torch.load('model_vae_500-1.pth', map_location=torch.device(device)))
vae.eval()
## LOad trainig data
x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
x_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)
# x_train = x_train[:,:500]   
# x_test = x_test[:,:500]
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

in_features = latent_dim
##I just choose this like they did in the tutorial so if this is not the optimal setting i can change it
wiring = AutoNCP(in_features+10, in_features )
model =  ncps.CfC(in_features,units=25,proj_size=in_features, mode="default", batch_first=True)
#unit size is the dimension of the hidden state
model.to(device)
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
train_data = torch.utils.data.TensorDataset(x_train_liq, y_train_liq)
test_data = torch.utils.data.TensorDataset(x_test_liq, y_test_liq)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=51, shuffle=False)  
train_loader = torch.utils.data.DataLoader(train_data, batch_size=51, shuffle=False)
criterion = torch.nn.MSELoss(reduction="sum")
for i in range(n_epochs):
    model.train()
    total_loss = 0
    total_loss_test = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred,_ = model(x)
        
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()/len(train_loader)

    for x_test_liq, y_test_liq in test_loader:
        x_test_liq = x_test_liq.to(device)
        y_test_liq = y_test_liq.to(device)
        y_test_pred,_ = model(x_test_liq)
        
        loss_test = criterion(y_test_liq, y_test_pred)
        total_loss_test += loss_test.item()/len(test_loader)

    print(f"Epoch {i}, Loss: {total_loss}, Test Loss: {total_loss_test}")
    
torch.save(model, 'model_liq.pth')