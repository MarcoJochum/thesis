import torch
import copy
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
# Define the autoencoder architecture
torch.manual_seed(42) 

def train_vae(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    # Loop over the dataset multiple times
    model.train()
    best_loss = 1e8
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        running_kld = 0.0
        # Loop over the data loader
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            #train_loader is iterable, returns a batch of data at each iteration
            #data = data.unsqueeze(1)
            data = data.to(device)
            decoded, encoded, mu, log_var  = model(data)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, data) + 1* KLD

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()#*data.size(0)
            running_kld += KLD.item()#*data.size(0)
        val_loss = 0
        i=0
        for x_val in val_loader:
            x_val = x_val.to(device)
            model.eval()
            val_pred, _, mu, log_var = model(x_val)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            val_loss += criterion(val_pred, x_val) + 1.0*KLD
            i+=1
            
        model.train()   

        if val_loss/len(val_loader.dataset) < best_loss:
                print('Model updated:', epoch)
                print('Validation loss:', val_loss.item()/len(val_loader.dataset))
                print("length of validation loader", len(val_loader.dataset))
                best_loss = val_loss/len(val_loader.dataset)
                torch.save(model.state_dict(), "model_vae_500.pth")
                           
        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_KLD = running_kld / len(train_loader.dataset)
        
        print('Epoch {}/{} Loss: {:.4f}  Epoch KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss, epoch_KLD)) 
        scheduler.step()         

def train_vae_tb(model, criterion, optimizer, train_loader, val_loader, num_epochs, tb):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    # Loop over the dataset multiple times
    model.train()
    best_loss = 1e8
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        running_kld = 0.0
        # Loop over the data loader
        for x_train, params_train, time in train_loader:
            # Get the inputs; data is a list of [inputs, labels]
            #train_loader is iterable, returns a batch of data at each iteration
            #data = data.unsqueeze(1)
            x_train = x_train.to(device)
            
            decoded, encoded, mu, log_var  = model(x_train)
            
            #Force the first 3 elements to be the parameters
            L_sup = criterion(encoded[:,:3], params_train)
            
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, x_train) + 1* KLD + 0.01*L_sup

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()#*data.size(0)
            running_kld += KLD.item()#*data.size(0)
        
        ##Validation loss
        val_loss = 0
        for x_val, params_val, time in val_loader:
            x_val = x_val.to(device)
            model.eval()
            val_pred, _, mu, log_var = model(x_val)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            val_loss += criterion(val_pred, x_val) + 1.0*KLD
            
          
        model.train()   

        if val_loss/len(val_loader.dataset) < best_loss:
                print('Model updated:', epoch)
                print('Validation loss:', val_loss.item()/len(val_loader.dataset))
                best_loss = val_loss/len(val_loader.dataset)
                best_model=copy.deepcopy(model.state_dict())
                           
        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_KLD = running_kld / len(train_loader.dataset)
        tb.add_scalar('Loss', epoch_loss, epoch)
        tb.add_scalar('KLD', epoch_KLD, epoch)
        tb.add_scalar('Validation Loss', val_loss.item()/len(val_loader.dataset), epoch)      
        tb.add_scalar('Learning rate', scheduler.get_lr()[0] , epoch)

        
        print('Epoch {}/{} Loss: {:.4f}  Epoch KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss, epoch_KLD)) 
        scheduler.step() 
    return epoch_loss, epoch_KLD, best_model







def train_lstm(model, criterion, optimizer, data_loader, num_epochs, val_loader):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    best_loss = 1e6
    for epoch in range(num_epochs):
        running_loss = 0.0 
        model.train()
        for x,  y in data_loader:
            
            
            y_pred = model(x)[:, -1, :]
        
            loss_train = criterion(y_pred, y[:, -1, :])

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()    
        if epoch % 10 == 0:
            epoch_loss = running_loss / len(data_loader.dataset)
            print('Epoch {}/{} Loss: {:.4f} '.format(epoch+1,num_epochs, epoch_loss, ))
            model.eval()
            val_loss = 0        
            for x_val, y_val in val_loader:
                y_val_pred = model(x_val)[:, -1, :]
                val_loss += criterion(y_val_pred, y_val[:,-1,:] )
                if val_loss/len(val_loader.dataset) < best_loss:
                    print('Model updated:', epoch)
                    print('Validation loss:', val_loss.item()/len(val_loader.dataset))
                    best_loss = val_loss/len(val_loader.dataset)
                    best_model=copy.deepcopy(model.state_dict())
            

        
        

        
        
        scheduler.step()    

    return best_model


def train_cae(model, criterion, optimizer, train_loader, num_epochs):
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.001)
    # Loop over the dataset multiple times
    model.train()
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        KLD = 0
        # Loop over the data loader
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            #data = data.unsqueeze(1)
            data = data.to(device)
            decoded, encoded  = model(data)
            
            loss = criterion(decoded, data)#+ (torch.sum(y_train!=y_pred))

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()#*data.size(0)
        
        #scheduler.step()
        # Print the average loss for the epoche
        epoch_loss = running_loss / len(train_loader.dataset)

        print('Epoch {}/{} Loss: {:.4f} '.format(epoch+1,num_epochs, epoch_loss, )) 