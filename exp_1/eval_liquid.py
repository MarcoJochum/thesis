from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
import torch
import matplotlib.pyplot as plt 
import random 
from config.liquid import Liquid_config
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
# set_seed(42)




###
n_epochs = 50
batch_size = Liquid_config.batch_size
latent_dim = Liquid_config.latent_dim
base = Liquid_config.base
part_class = Liquid_config.part_class
vae_name = "models/model_vae_lin.pth"

model_name = "model_liq_loss.pth"#Liquid_config.model_name
#data = DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z, suffix)
#data.kmc_data='../../data_kmc/'
#data.avg_trj("1400_1e+20_2.0")


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
VAE = VAE(encoder, decoder, latent_dim=latent_dim)
 
VAE.load_state_dict(torch.load(vae_name, map_location=torch.device("cpu")))
VAE.eval()


lnn = torch.load(model_name, map_location=torch.device("cpu"))
lnn.eval()
##Load data

x_train = Liquid_config.data_train
x_test = Liquid_config.data_test
x_train = x_train/torch.mean(x_train)
x_test = x_test/torch.mean(x_train)

list_x_train = []
list_x_test = []  
with torch.no_grad():
    for x in x_train:
        _, _, x_lat, _ = VAE(x)
        list_x_train.append(x_lat)
    for y in x_test: 
        _, _, y_lat ,_ = VAE(y)
        list_x_test.append(y_lat)
x_train_lat =torch.stack(list_x_train)
x_test_lat = torch.stack(list_x_test)

##Make sequences
#test_trj =torch.tensor(np.load("../../data_kmc/1400_1e+20_2.0/avg_trj.npy"), dtype=torch.float32)   #x_train[50]
#test_trj = torch.reshape(test_trj, (1000,1,50,100))

# with torch.no_grad():
#     _,test_trj_lat,_,_ = VAE(test_trj)
test_trj = x_train[:8]
test_trj_lat = x_train_lat[:8]
with open("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names[:8]   
##Prediction horizon
t = 1000
##Lookback  

y_pred_lat = torch.zeros((8,t,latent_dim))
time = torch.logspace(-8, -4, 1000)

hx = torch.zeros((Liquid_config.units))
  
y_pred_lat[:,0] = test_trj_lat[:,0]
y_pred = torch.zeros((8,t,50,100))
with torch.no_grad():
   for j in range(6): 
    for i in range(0,t-1):
        if i<10 :
        
            y_pred_lat[j,i+1] , hx = lnn(test_trj_lat[j,i].unsqueeze(0), hx , timespans=time[i].unsqueeze(0))
            
        else:
            y_pred_lat[j,i+1], hx = lnn(y_pred_lat[j,i].unsqueeze(0), hx, timespans=time[i].unsqueeze(0))
             

    test = VAE.decoder(y_pred_lat[j])
    print(test.shape)
    y_pred[j] = torch.reshape(test, (t,50,100))
        

##1000x50 shape for decoder
#y_pred = VAE.decoder(y_pred_lat)
    


##Average across x dim
  
test_trj = torch.mean(test_trj.squeeze(), dim=2).squeeze()
y_pred = torch.mean(y_pred, dim=2).squeeze()

time = np.logspace(-8, -4, 1000, 'o')
labels = [] 
with torch.no_grad():
      
    fig,axs = plt.subplots(6,2, figsize=(6,20))
    for j in range(6):
        for i in range(0, 100,10):
            
        
            
            axs[j,0].plot(np.linspace(0,100,100), y_pred[j, i].detach().numpy())
            labels.append([f"t = {time[i]}"])
            axs[j,0].set_ylim(0, 5)
            axs[j,0].set_title("Prediction"+ names[j])
            #axs[0].legend(labels)

            axs[j,1].plot(np.linspace(0,100,100), test_trj[j,i].detach().numpy())
            #labels.append([f"t = {time[i-1]}"])
            axs[j,1].set_ylim(0, 5)
            axs[j,1].set_title("Ground truth"+ names[j])
            #axs[1].legend(labels)

 
plt.savefig("liq_loss_realspace_train_0_8.png")
print("Number of parameters in LNN:",sum(p.numel() for p in lnn.parameters() if p.requires_grad))