from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
import torch
import matplotlib.pyplot as plt 
from config.vae import VAE_config
from config.lstm import LSTM_config

n_time = 1000
seq_len = LSTM_config.seq_len
###

latent_dim = LSTM_config.latent_dim
base = LSTM_config.base
part_class = LSTM_config.part_class 
mode = 'train'  
## Configuration to visualize
num_configs = 8


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim, mode=mode)
 
vae.load_state_dict(torch.load("models/model_vae_lin.pth", map_location=torch.device("cpu")))
vae.eval()



lstm = LSTMs(latent_dim, d_model=latent_dim, n_mode=1, hidden_size=50, num_layers=1, num_classes=1, embed=None)
lstm.load_state_dict(torch.load(LSTM_config.model_name,map_location=torch.device('cpu')))
lstm.eval()


##Load data

x_train = LSTM_config.data_train
x_test = LSTM_config.data_test

test_params = LSTM_config.test_params
train_params = LSTM_config.train_params##epsr, cbulk, v_bias

##Rescale parameters with the mean
params_scale = torch.mean(train_params, dim=0)
train_params = train_params/params_scale
train_params_non_stack = train_params

test_params = test_params/params_scale
test_params_non_stack = test_params
test_params = torch.repeat_interleave(test_params, 1000, dim=0)
train_params = torch.repeat_interleave(train_params, 1000, dim=0)

train_params = torch.reshape(train_params, (x_train.shape[0],x_train.shape[1], train_params.shape[-1] ) )
test_params = torch.reshape(test_params, (x_test.shape[0],x_test.shape[1], test_params.shape[-1] ) )

x_train = x_train/torch.mean(x_train)   
x_test = x_test/torch.mean(x_train)

list_x_train = []
list_x_test = []  

## add params to encoding
with torch.no_grad():
    for i in range(len(x_train)):
        x = x_train[i]  # Add batch dimension
        params = train_params[i]  # Add batch dimension
        _, x_lat, _, _ = vae(x, params)
        list_x_train.append(x_lat)
    for i in range(len(x_test)): 
        y = x_test[i]  # Add batch dimension
        params = test_params[i]  # Add batch dimension
        _, y_lat, _, _ = vae(y, params)
        list_x_test.append(y_lat)

x_train_lat =torch.stack(list_x_train)
x_test_lat = torch.stack(list_x_test)
##Make sequences
test_trj = x_train[:num_configs]
test_trj_lat = x_train_lat[:num_configs]
with open("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
names = names[:num_configs]   
##Prediction horizon
t = 150
##seq_len  

y_pred_lat = torch.zeros((num_configs,t+seq_len+1,latent_dim))
y_pred_lat[:,:seq_len] = test_trj_lat[:,:seq_len]
#time = torch.logspace(-8, -4, 1000)   
#print("Time shape:", time.shape)     
h= torch.zeros(num_configs, 50)
c = torch.zeros(num_configs, 50)#
with torch.no_grad():

        for i in range(seq_len,t+seq_len):
            
            out  = lstm(y_pred_lat[:,i-seq_len:i], mode = "eval")
            
            y_pred_lat[:,i+1] = out[:,-1,:]
            if mode != 'train':
                y_pred_lat[:,i+1,:3] = test_params_non_stack[:num_configs] #test_params_non_stack[n_config]
            #Enforce parameters on first 3 elements
            
    


y_pred_lat = torch.reshape(y_pred_lat, (y_pred_lat.shape[0]*y_pred_lat.shape [1],latent_dim))
y_pred = vae.decoder(y_pred_lat)
y_pred = torch.reshape(y_pred, (num_configs,t+seq_len+1,50,100))

##Average across x dim
test_trj = torch.mean(test_trj.squeeze(), dim=2)
y_pred = torch.mean(y_pred, dim=2).squeeze()

print("tsts trj ", test_trj.shape)
time = np.logspace(-8, -4, 1000, 'o')
labels = [] 
with torch.no_grad():
      
    fig,axs = plt.subplots(num_configs,2, figsize=(6,20))
    for j in range(num_configs):

        for i in range(0, 149,10):
            
           
            axs[j,0].plot(np.linspace(0,100,100), y_pred[j, i].numpy())
            labels.append([f"t = {time[i]}"])
            axs[j,0].set_ylim(0, 5)
            axs[j,0].set_title("Prediction"+ names[j])
            #axs[0].legend(labels)

            axs[j,1].plot(np.linspace(0,100,100), test_trj[j,i].numpy())
            #labels.append([f"t = {time[i-1]}"])
            axs[j,1].set_ylim(0, 5)
            axs[j,1].set_title("Ground truth"+ names[j])
            #axs[1].legend(labels)


 
plt.savefig("lstm_lin_test_"+str(num_configs)+".png")
print("Number of parameters in lstm:",sum(p.numel() for p in lstm.parameters() if p.requires_grad))