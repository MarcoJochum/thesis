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
latent_dim = 40
base = 32
part_class = 1  
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
vae = VAE(encoder, decoder, latent_dim=latent_dim)
 
vae.load_state_dict(torch.load('model_vae_kmc_red_data_e1000.pth', map_location=torch.device('cpu')))
vae.eval()


model = LSTMs(latent_dim, d_model=latent_dim, hidden_size=80, num_layers=2, num_classes=1, embed=None)
model.load_state_dict(torch.load('model_lstm.pth',map_location=torch.device('cpu')))
model.eval()
data_folder = '../../data_kmc/2d_encoded/' 
encodings = loading_encodings(data_folder)
data = torch.tensor(np.load('../../data_kmc/2d_sets/2d_red_5.npy'), dtype=torch.float32)

holdout = encodings[-5]
encodings = encodings[:-5]
train_size = int(encodings.shape[1] * 0.67)    
test_size = encodings.shape[1] - train_size
train, test = encodings[:,0:train_size], encodings[:,train_size:]
x_train, y_train = make_sequence(train, 64, 10)
x_test, y_test = make_sequence(test, 64, 10)
timeseries = np.zeros((n_time, n_x, n_z))
print(x_train[1].shape)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
time = np.logspace(-8, -4, 1000, 'o')
labels = [] 
with torch.no_grad():
    
    train_plot = np.ones_like(timeseries)*np.nan
    y_pred_encode = model(x_train[1])[:,-1,:]
    y_pred_encode = y_pred_encode.unsqueeze(1)
    print("y_pred_encode shape:",y_pred_encode.shape)
    y_pred = vae.decoder(y_pred_encode)
    print("Decoded y pred shape:", y_pred.shape)
    train_plot[lookback:train_size] = y_pred.squeeze().numpy()
    train_plot = np.mean(train_plot, axis=1)
    print("train plot shape:", train_plot.shape)
    test_plot = np.ones_like(timeseries)*np.nan
    y_test_encode = model(x_test[1])[:,-1,:]
    y_test_encode = y_test_encode.unsqueeze(1)
    print("y_test_encode shape:",y_test_encode.shape)
    
    y_test = vae.decoder(y_test_encode)
    print("y test decode shape:", y_test.shape)
    test_plot[train_size+lookback:] = y_test.squeeze().numpy()
    test_plot = np.mean(test_plot, axis=1)
    print("Y test values", y_test.mean())
    t_end = 1000
    t_space = 100
    t_start = 600
    num_plots=3
    fig,axs = plt.subplots(1,num_plots, figsize=(15,5))
    
    y_true = torch.mean(data[1], dim=1)
    print("y_true shape:", y_true.shape)

    for i in range(t_start, t_end,t_space):
        
        axs[0].plot(np.linspace(0,100,100), train_plot[i, :])
        labels.append([f"t = {time[i-1]}"])
        #axs[0].set_ylim(0, 5) 
        axs[0].set_title("Train")
        axs[0].legend(labels)

        axs[1].plot(np.linspace(0,100,100), test_plot[i, :], linestyle="-.")
        labels.append([f"t = {time[i-1]}"])
        #axs[1].set_ylim(0, 5)
        axs[1].set_title("Test")
        axs[1].legend(labels)

        axs[2].plot(np.linspace(0,100,100), y_true[i, 0,:].detach().numpy())
        labels.append([f"t = {time[i-1]}"])
        #axs[1].set_ylim(0, 5)
        axs[2].set_title("Ground truth")
        axs[2].legend(labels)

 
    plt.savefig("lstm_compare.png")