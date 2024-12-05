import torch
import torch.utils

from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
from lib.data_loading import *
from lib.train import *
from config.seq import Seq_config
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #if torch.cuda.is_available():
        #torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
seed(42)
model_name = Seq_config.model_name

x_train = Seq_config.data_train
x_test = Seq_config.data_test

n_steps = Seq_config.n_steps
x_train = x_train/torch.mean(x_train)
x_test = x_test/torch.mean(x_train)
x_train = x_train[:, :n_steps]
x_test = x_test[:, :n_steps]

latent_dim = Seq_config.latent_dim
hidden = Seq_config.hidden
num_layers = Seq_config.num_layers
part_class = 1
base = 4
seq_length = Seq_config.seq_length
pred_length = Seq_config.pred_length
n_seq = n_steps - seq_length - pred_length+1


encoder = pro_encoder2d(part_class,base, latent_dim)
decoder = pro_decoder2d(part_class,base, latent_dim)
VAE = VAE(encoder, decoder, latent_dim=latent_dim)
VAE.load_state_dict(torch.load("models/model_vae_lin.pth", map_location=torch.device(device)))


encoder_lstm = Encoder_lstm(latent_dim, hidden, num_layers)
decoder_lstm = Decoder_lstm(latent_dim, hidden, num_layers)
model = Seq2seq(encoder_lstm, decoder_lstm)
model.to(device)
model.train()
print("Number of parameters in lstm:",sum(p.numel() for p in model.parameters() if p.requires_grad))
x_train_seq, y_train_seq = make_sequence2(x_train, seq_length, pred_length)
x_test_seq, y_test_seq = make_sequence2(x_test, seq_length, pred_length)




train_size = int(0.8*x_train_seq.shape[0])
val_size = x_train_seq.shape[0] - train_size
test_size = x_test_seq.shape[0]
indices = torch.randperm(x_train_seq.shape[0])
train_indices = indices[:train_size]
val_indices = indices[train_size:]
target_train= torch.reshape(y_train_seq[:train_size], (train_size* y_train_seq.shape[1]*y_train_seq.shape[2], 1, y_train_seq.shape[4], y_train_seq.shape[5]))

target_val = torch.reshape(y_train_seq[train_size:], (val_size* y_train_seq.shape[1]*y_train_seq.shape[2], 1,
                                                       y_train_seq.shape[4], y_train_seq.shape[5]))
y_val_seq = torch.reshape(y_train_seq[train_size:], (val_size*y_train_seq.shape[1], y_train_seq.shape[2], 1,
                                                      y_train_seq.shape[4], y_train_seq.shape[5]))
y_train_seq = torch.reshape(y_train_seq[:train_size], (train_size*y_train_seq.shape[1], y_train_seq.shape[2],1,
                                                        y_train_seq.shape[4], y_train_seq.shape[5]))
target_test = torch.reshape(y_test_seq, (y_test_seq.shape[0]* y_test_seq.shape[1]*y_test_seq.shape[2], 1,
                                          y_test_seq.shape[4], y_test_seq.shape[5]))
y_test_seq = torch.reshape(y_test_seq, (test_size*y_test_seq.shape[1], y_test_seq.shape[2], 1,
                                         y_test_seq.shape[4], y_test_seq.shape[5]))

val_seq = torch.reshape(x_train_seq[train_size:], (val_size*x_train_seq.shape[1]*x_train_seq.shape[2],
                                                        1, x_train_seq.shape[4], x_train_seq.shape[5]))

x_train_seq = torch.reshape(x_train_seq[:train_size], (train_size*x_train_seq.shape[1]* x_train_seq.shape[2], 
                                                       1, x_train_seq.shape[4], x_train_seq.shape[5]))



x_test_seq = torch.reshape(x_test_seq, (x_test_seq.shape[0]*x_test_seq.shape[1]* x_test_seq.shape[2],
                                        x_test_seq.shape[3], x_test_seq.shape[4], x_test_seq.shape[5]))
#encode x_train already using the vae
with torch.no_grad():
    x_train_seq = VAE(x_train_seq)[2] ##saving mu here
    val_seq = VAE(val_seq)[2]
    x_test_seq = VAE(x_test_seq)[2]
    target_train = VAE(target_train)[2]
    target_val = VAE(target_val)[2]
    target_test = VAE(target_test)[2]
##create data loader


x_train_seq = torch.reshape(x_train_seq, (train_size* n_seq, seq_length, latent_dim))
val_seq = torch.reshape(val_seq, (val_size* n_seq, seq_length, latent_dim))
x_test_seq = torch.reshape(x_test_seq, (test_size* n_seq, seq_length, latent_dim))
target_train = torch.reshape(target_train, (train_size* n_seq, pred_length, latent_dim))
target_val = torch.reshape(target_val, (val_size* n_seq, pred_length, latent_dim))
target_test = torch.reshape(target_test, (test_size* n_seq, pred_length, latent_dim))


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_seq, y_train_seq, target_train), batch_size=Seq_config.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_seq, y_val_seq, target_val), batch_size=Seq_config.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_seq, y_test_seq), batch_size=Seq_config.batch_size, shuffle=False)


optimizer = torch.optim.Adam(model.parameters(), lr=Seq_config.lr)
criterion = nn.MSELoss()

best_model = train_seq(model, VAE, criterion ,optimizer , train_loader, val_loader, Seq_config.n_epochs, model_name)

torch.save(model, model_name)