from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
import torch
import matplotlib.pyplot as plt 
n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000
lookback = 64
###
n_epochs = 50
batch_size = 200
latent_dim = 50
base = 8
part_class = 1  
data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"
#data = DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z, suffix)
#data.kmc_data='../../data_kmc/'
#data.avg_trj("1400_1e+20_2.0")


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
VAE = VAE(encoder, decoder, latent_dim=latent_dim)
 
VAE.load_state_dict(torch.load('model_vae_500.pth', map_location=torch.device("cpu")))
VAE.eval()



lnn = torch.load("model_liq.pth", map_location=torch.device("cpu"))
lnn.eval()
##Load data

x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
x_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)

list_x_train = []
list_x_test = []  
with torch.no_grad():
    for x in x_train:
        _, x_lat, _, _ = VAE(x)
        list_x_train.append(x_lat)
    for y in x_test: 
        _, y_lat, _ ,_ = VAE(y)
        list_x_test.append(y_lat)
x_train_lat =torch.stack(list_x_train)
x_test_lat = torch.stack(list_x_test)

##Make sequences
test_trj =torch.tensor(np.load("../../data_kmc/1400_1e+20_2.0/avg_trj.npy"), dtype=torch.float32)   #x_train[50]
test_trj = torch.reshape(test_trj, (1000,1,50,100))

with torch.no_grad():
    _,test_trj_lat,_,_ = VAE(test_trj)

##Prediction horizon
t = 1000
##Lookback  

y_pred_lat = torch.zeros((t,latent_dim))
#hx = test_trj_lat[0]
#time = torch.logspace(-8, -4, 1000)   
#print("Time shape:", time.shape)     
hx = torch.zeros((25))

for i in range(0,t-1):
    print("test_trj_lat shape:", test_trj_lat[i].shape)
   
    y_pred_lat[i], hx = lnn(test_trj_lat[i].unsqueeze(0), hx)
    print("hx shape:", hx.shape)

with torch.no_grad():
    y_pred = VAE.decoder(y_pred_lat)


##Average across x dim
test_trj = torch.mean(test_trj, dim=2).squeeze()
y_pred = torch.mean(y_pred, dim=2).squeeze()
print("y_pred shape:", y_pred.shape)
print("test_plot shape:", test_trj.shape)

time = np.logspace(-8, -4, 1000, 'o')
labels = [] 
with torch.no_grad():
      
    fig,axs = plt.subplots(1,2, figsize=(15,5))
    for i in range(0, 700,100):
        
        
        axs[0].plot(np.linspace(0,100,100), y_pred[i].detach().numpy())
        labels.append([f"t = {time[i-1]}"])
        axs[0].set_ylim(0, 5)
        axs[0].set_title("Prediction")
        #axs[0].legend(labels)

        axs[1].plot(np.linspace(0,100,100), test_trj[i].detach().numpy())
        #labels.append([f"t = {time[i-1]}"])
        axs[1].set_ylim(0, 5)
        axs[1].set_title("Ground truth")
        #axs[1].legend(labels)

 
plt.savefig("liquid_compare_test_1e20.png")
print("Number of parameters in LNN:",sum(p.numel() for p in lnn.parameters() if p.requires_grad))