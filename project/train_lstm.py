from lib.train import *
from lib.data_loading import *
from NNs.RNN import *
from lib.data import *
from NNs.autoencoder import *
import torch.utils.data as data
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

##Call ae model to encode all the time series for the different configs
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000
###
n_epochs = 400
batch_size = 4000
latent_dim = 50
base = 8
part_class = 1

##Initialize VAE for encodings
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
##Switch to enforcing params in the vae
vae = VAE(encoder, decoder, latent_dim=latent_dim, mode='eval')
vae.load_state_dict(torch.load('model_vae_p.pth', map_location=torch.device(device)))
vae.eval()

## LOad trainig data
x_train = torch.tensor(np.load('../../data_kmc/2d_sets/train_set_80_20.npy'), dtype=torch.float32)
x_test = torch.tensor(np.load('../../data_kmc/2d_sets/test_set_80_20.npy'), dtype=torch.float32)

test_params = get_config("../../data_kmc/2d_sets/test_set_80_20_list.txt")
train_params = get_config("../../data_kmc/2d_sets/train_set_80_20_list.txt")##epsr, cbulk, v_bias

x_train = x_train[:,::10]/torch.mean(x_train)   
x_test = x_test/torch.mean(x_train)
##Rescale parameters with the mean
params_scale = torch.mean(train_params, dim=0)
train_params = train_params/params_scale
test_params = test_params/params_scale

test_params = torch.repeat_interleave(test_params, 1000, dim=0)
train_params = torch.repeat_interleave(train_params, 100, dim=0)

train_params = torch.reshape(train_params, (x_train.shape[0],x_train.shape[1], train_params.shape[-1] ) )
test_params = torch.reshape(test_params, (x_test.shape[0],x_test.shape[1], test_params.shape[-1] ) )



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




lstm = LSTMs(latent_dim, d_model=latent_dim, n_mode=1, hidden_size=50, num_layers=1, num_classes=1, embed=None)
lstm.to(device)
print("LSTM parameter count:", sum(p.numel() for p in lstm.parameters() if p.requires_grad))
x_train_seq, y_train_seq = make_sequence(x_train_lat, 10, 1)
x_test_seq, y_test_seq = make_sequence(x_test_lat, 10, 1)

# time = torch.tensor(torch.logspace(-8, -4, 1000))
# time = time[:500]
# time = time.reshape(time.shape[0], 1)
# time ,_ = make_sequence(time, 64, 1)
# time_test = torch.stack([time for i in range( x_test_seq.shape[0])]).to(device) 
# time = torch.stack([time for i in range( x_train_seq.shape[0])]).to(device)

# time = time.reshape(time.shape[0]*time.shape[2], 64,1)
# time_test = time_test.reshape(time_test.shape[0]*time_test.shape[2], 64,1)

x_train_seq = torch.reshape(x_train_seq, (x_train_seq.shape[0]*x_train_seq.shape[1], x_train_seq.shape[2], x_train_seq.shape[3])).to(device)
y_train_seq = torch.reshape(y_train_seq, (y_train_seq.shape[0]*y_train_seq.shape[1], y_train_seq.shape[2], y_train_seq.shape[3])).to(device)
x_test_seq = torch.reshape(x_test_seq, (x_test_seq.shape[0]*x_test_seq.shape[1], x_test_seq.shape[2], x_test_seq.shape[3])).to(device)
y_test_seq = torch.reshape(y_test_seq, (y_test_seq.shape[0]*y_test_seq.shape[1], y_test_seq.shape[2], y_test_seq.shape[3])).to(device)


train_loader = torch.utils.data.DataLoader(data.TensorDataset(x_train_seq[:-10], y_train_seq[:-10]), batch_size=batch_size, 
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(data.TensorDataset(x_train_seq[-10:], y_train_seq[-10:]), batch_size=10000, 
                                         shuffle=True)

#time_val = time[-10000:]
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

best_model=train_lstm(lstm, criterion, optimizer, train_loader, n_epochs, val_loader)
#lstm.cpu() # move the model to the cpu
torch.save(lstm.state_dict(), 'model_lstm_p24.pth')
torch.save(best_model, 'model_lstm_val_p24.pth')

with torch.no_grad():
                y_pred = lstm(x_test_seq)[:,-1,:]
                loss_test = criterion(y_pred, y_test_seq[:,-1,:])
                print('Test loss:', loss_test.item()/len(y_test_seq))