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

## Configuration to visualize
n_config = 10


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim, mode='eval')
 
vae.load_state_dict(torch.load('model_vae_p.pth', map_location=torch.device("cpu")))
vae.eval()



lstm = LSTMs(latent_dim, d_model=latent_dim, n_mode=1, hidden_size=50, num_layers=1, num_classes=1, embed=None)
lstm.load_state_dict(torch.load('model_lstm_p24.pth',map_location=torch.device('cpu')))
lstm.eval()


##Load data

x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
x_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)

test_params = get_config("../../data_kmc/2d_sets/test_set_80_20_list.txt")
train_params = get_config("../../data_kmc/2d_sets/train_set_80_20_list.txt")##epsr, cbulk, v_bias

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
test_trj = x_train[n_config]
test_trj_lat = x_train_lat[n_config]

##Prediction horizon
t = 100
##Lookback  
lookback = 1
y_pred_lat = torch.zeros((t+lookback+1,latent_dim))
y_pred_lat[:lookback] = test_trj_lat[:lookback]
#time = torch.logspace(-8, -4, 1000)   
#print("Time shape:", time.shape)     
h= torch.zeros(1, 1, 50)
c = torch.zeros(1, 1, 50)

for i in range(lookback,t+lookback):
    out  = lstm(y_pred_lat[i-lookback:i].unsqueeze(0), mode = "train")
    y_pred_lat[i+1] = out[:,-1,:]
    #Enforce parameters on first 3 elements
    y_pred_lat[i+1,:3] = train_params_non_stack[n_config] #test_params_non_stack[n_config]
    


y_pred = vae.decoder(y_pred_lat)


##Average across x dim
test_trj = torch.mean(test_trj, dim=2).squeeze()
y_pred = torch.mean(y_pred, dim=2).squeeze()
print("y_pred shape:", y_pred.shape)
print("test_plot shape:", test_trj.shape)

time = np.logspace(-8, -4, 1000, 'o')
labels = [] 
with torch.no_grad():
      
    fig,axs = plt.subplots(1,2, figsize=(15,5))
    for i in range(0, 10,1):
        
        
        axs[0].plot(np.linspace(0,100,100), y_pred[i].detach().numpy())
        labels.append([f"t = {time[i]}"])
        axs[0].set_ylim(0, 5)
        axs[0].set_title("Prediction")
        #axs[0].legend(labels)

        axs[1].plot(np.linspace(0,100,100), test_trj[i].detach().numpy())
        #labels.append([f"t = {time[i-1]}"])
        axs[1].set_ylim(0, 5)
        axs[1].set_title("Ground truth")
        #axs[1].legend(labels)

 
plt.savefig("lstm_0_10_train_"+str(n_config)+".png")
print("Number of parameters in lstm:",sum(p.numel() for p in lstm.parameters() if p.requires_grad))