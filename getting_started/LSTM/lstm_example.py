import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data
class Airmodel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Airmodel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x , _ = self.lstm(x)
        x= self.fc(x)
        return x  #returning the last timestep output

df = pd.read_csv('airline-passengers.csv')
timeseries = df['Passengers'].values.astype(float)

#plt.plot(timeseries)
#plt.show()

#train test split

train_size = int(len(timeseries) * 0.67)    
test_size = len(timeseries) - train_size
train, test = timeseries[0:train_size], timeseries[train_size:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+1:i + look_back+1])
    return torch.tensor(dataX).unsqueeze(-1), torch.tensor(dataY).unsqueeze(-1)

lookback = 4

X_train, y_train = create_dataset(train, lookback)  
X_test, y_test = create_dataset(test, lookback)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train[0], y_train[0])

model = Airmodel(1, 50, 1, 1)
model.double()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

n_epochs = 2000 
for epoch in range(n_epochs):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss_train = criterion(y_pred, y)
        loss_train.backward()
        optimizer.step()
    if epoch % 100 == 0:
        
        continue

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        loss_test = np.sqrt(criterion(y_pred, y_test))
        loss_train = np.sqrt(criterion(model(X_train), y_train))
        print(f'Epoch: {epoch}, Loss Train: {loss_train.item()}, Loss Test: {loss_test.item()}')
        


with torch.no_grad():

    train_plot = np.ones_like(timeseries)*np.nan
    y_pred = model(X_train)[:,-1,:]
    train_plot[lookback:train_size] = y_pred.squeeze().numpy()

    test_plot = np.ones_like(timeseries)*np.nan
    test_plot[train_size+lookback:] = model(X_test)[:,-1,:].squeeze().numpy()

plt.plot(timeseries, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()