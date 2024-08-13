from lib.train import *
from lib.data_loading import *
from NNs.RNN import *
from lib.data import *
from NNs.autoencoder import *
import torch.utils.data as data
import random
from config.lstm import LSTM_config

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

###
n_epochs = LSTM_config.n_epochs
batch_size = LSTM_config.batch_size
lr = LSTM_config.lr
latent_dim = LSTM_config.latent_dim
base = LSTM_config.base
part_class = LSTM_config.part_class
model_name = LSTM_config.model_name
vae_name = "models/model_vae_lin.pth"
hidden_size = LSTM_config.hidden_size
num_layers = LSTM_config.num_layers
num_classes = LSTM_config.num_classes
seq_len = LSTM_config.seq_len
n_steps = 50
##
lstm = LSTMs(latent_dim, d_model=latent_dim, n_mode=1, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, embed=None)
lstm.to(device)
##Initialize VAE for encodings
encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
##Switch to enforcing params in the vae
vae = VAE(encoder, decoder, latent_dim=latent_dim)
vae.load_state_dict(torch.load(vae_name, map_location=torch.device(device)))
vae.eval()

## LOad trainig data
x_train =LSTM_config.data_train
x_test = LSTM_config.data_test

test_params = LSTM_config.test_params
train_params = LSTM_config.train_params##epsr, cbulk, v_bias

x_train = x_train[:,:n_steps]/torch.mean(x_train)   
x_test = x_test[:,:n_steps]/torch.mean(x_train)
##Rescale parameters with the mean
params_scale = torch.mean(train_params, dim=0)
train_params = train_params/params_scale
test_params = test_params/params_scale

test_params = torch.repeat_interleave(test_params, n_steps, dim=0)
train_params = torch.repeat_interleave(train_params, n_steps, dim=0)

train_params = torch.reshape(train_params, (x_train.shape[0],x_train.shape[1], train_params.shape[-1] ) )
test_params = torch.reshape(test_params, (x_test.shape[0],x_test.shape[1], test_params.shape[-1] ) )



list_x_train = []
list_x_test = []  

## add params to encoding
with torch.no_grad():
    for i in range(len(x_train)):
        x = x_train[i]  # Add batch dimension
        params = train_params[i]  # Add batch dimension
        _, x_lat, _, _ = vae(x)
        list_x_train.append(x_lat)
    for i in range(len(x_test)): 
        y = x_test[i]  # Add batch dimension
        params = test_params[i]  # Add batch dimension
        _, y_lat, _, _ = vae(y)
        list_x_test.append(y_lat)

x_train_lat =torch.stack(list_x_train)
x_test_lat = torch.stack(list_x_test)





print("LSTM parameter count:", sum(p.numel() for p in lstm.parameters() if p.requires_grad))
x_train_seq, y_train_seq = make_sequence(x_train_lat, seq_len, 1)
x_test_seq, y_test_seq = make_sequence(x_test_lat, seq_len, 1)

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

train_size = int(0.8 * len(x_train_seq))
val_size = len(x_train_seq) - train_size

train_loader = torch.utils.data.DataLoader(data.TensorDataset(x_train_seq[:train_size], y_train_seq[:train_size]), batch_size=batch_size, 
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(data.TensorDataset(x_train_seq[train_size:], y_train_seq[train_size:]), batch_size=10000, 
                                         shuffle=True)


criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

best_model=train_lstm(lstm, criterion, optimizer, train_loader, n_epochs, val_loader)
#lstm.cpu() # move the model to the cpu

torch.save(best_model, model_name)

with torch.no_grad():
                y_pred = lstm(x_test_seq)[:,-1,:]
                loss_test = criterion(y_pred, y_test_seq[:,-1,:])
                print('Test loss:', loss_test.item()/len(y_test_seq))