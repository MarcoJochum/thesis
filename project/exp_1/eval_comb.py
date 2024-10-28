from lib.data_loading import *
from NNs.autoencoder import *
from NNs.RNN import *
from lib.data import *
import torch
import matplotlib.pyplot as plt 
import random 
from config.comb import Comb_config
x_train = Comb_config.data_train
x_test = Comb_config.data_test



vLNN = torch.load(Comb_config.model_name, map_location=torch.device("cpu"))
vLNN.eval()
test_trj = x_train[:8]

##Prediction horizon
t = 1000
time = torch.linspace(1e-07,1e-04,t).unsqueeze(0).repeat(60,1)    
print(time.shape)
hx = torch.zeros((8,60))
  

y_pred = torch.zeros((8,t,1,50,100))
y_pred[:, 0] = test_trj[:8, 0]
with torch.no_grad():
   #for j in range(6): 
    for i in range(0,t-1):
        if i<10 :
            print(test_trj[:8,i,:].unsqueeze(2).shape)
            y_pred[:,i+1,:] = vLNN(test_trj[:8,i,:].unsqueeze(2),hx, time[:,i].unsqueeze(1))[0].squeeze(2)
        else:
            y_pred[:,i+1,:] = vLNN(y_pred[:,i,:].unsqueeze(2),hx, time[:,i].unsqueeze(1))[0].squeeze(2)
             


##Average across x dim
print("test trj",test_trj.shape)  
test_trj = torch.mean(test_trj, dim=3).squeeze()
print("test trj",test_trj.shape)    
print("y pred", y_pred.shape)

y_pred = torch.mean(y_pred, dim=3).squeeze()
print("y pred", y_pred.shape)
with open("../../data_kmc/2d_sets/train_set_lin_80_20_list.txt", "r") as f:
    names= f.readlines()
names = [line.strip() for line in names]
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

 
plt.savefig("comb_lin_train_10.png")
print("Number of parameters in combined:",sum(p.numel() for p in vLNN.parameters() if p.requires_grad))