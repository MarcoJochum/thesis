from lib.train import *
from lib.data_loading import *
from NNs.RNN import *
from lib.data import *
import torch.utils.data as data
##Call ae model to encode all the time series for the different configs
torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = '../../data_kmc/'
suffix = "local_density_li+.txt"

n_prod = 50
n_x = 50
n_y = 1
n_z= 100
n_time = 1000
###
n_epochs = 50
batch_size = 200
latent_dim = 40
# train_data= DataLoading(data_folder, n_prod, n_time, n_x, n_y, n_z,suffix)
# data = train_data.make_data_set()
# folder_list = train_data.make_folder_list()

lstm = LSTMs(latent_dim, d_model=latent_dim, hidden_size=80, num_layers=2, num_classes=1, embed=None)
folder_name ="../../data_kmc/2d_encoded/"
lstm.to(device)
encodings = loading_encodings(folder_name)
print(encodings[1].shape)
holdout = encodings[-5]
encodings = encodings[:-5]

train_size = int(encodings.shape[1] * 0.67)    
test_size = encodings.shape[1] - train_size
train, test = encodings[:,0:train_size], encodings[:,train_size:]
print(train.shape)
print(test.shape)
x_train, y_train = make_sequence(train, 64, 1)
x_test, y_test = make_sequence(test, 64, 1)

x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
x_train = torch.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3]))
y_train = torch.reshape(y_train, (y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3]))
x_test = torch.reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2], x_test.shape[3]))
y_test = torch.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3]))
print("x_train shape:", x_train.shape)
print("y_train shape:",y_train.shape)
train_loader = torch.utils.data.DataLoader(data.TensorDataset(x_train, y_train), batch_size=batch_size, 
                                           shuffle=True)

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
train_lstm(lstm, criterion, optimizer, train_loader, n_epochs, x_test, y_test)
lstm.cpu() # move the model to the cpu
torch.save(lstm.state_dict(), 'model_lstm.pth')
lstm.eval()
lstm.load_state_dict(torch.load('model_lstm.pth'))