import torch
import copy
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
# Define the autoencoder architecture
torch.manual_seed(42) 

def train_vae(model, criterion, optimizer, train_loader, val_loader, num_epochs, model_name):
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
                best_loss = val_loss/len(val_loader.dataset)
                best_model=copy.deepcopy(model.state_dict())
                torch.save(best_model, model_name)
                           
        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_KLD = running_kld / len(train_loader.dataset)
        
        print('Epoch {}/{} Loss: {:.4f}  Epoch KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss, epoch_KLD)) 
        scheduler.step()         

def train_vae_tb(model, criterion, optimizer, train_loader, val_loader, VAE_config, num_epochs, tb):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    # Loop over the dataset multiple times
    model.train()
    best_loss = 1e8
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        running_kld = 0.0
        running_sup = 0.0
        # Loop over the data loader
        for x_train, params_train, time in train_loader:
            
            x_train = x_train.to(device)
            params_train = params_train.to(device)  
            decoded, encoded, mu, log_var  = model(x_train)
            
            #Force the first 3 elements to be the parameters
            L_sup = criterion(encoded[:,:3], params_train)
            
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, x_train) + VAE_config.KLD_weight * KLD + VAE_config.L_sup_weight *L_sup

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()#*data.size(0)
            running_kld += KLD.item()#*data.size(0)
            running_sup += L_sup.item()
        
        ##Validation loss
        val_loss = 0
        for x_val, params_val, time in val_loader:
            x_val = x_val.to(device)
            params_val = params_val.to(device)
            model.eval()
            val_pred, val_encoded, mu, log_var = model(x_val)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            L_sup = criterion(val_encoded[:,:3], params_val)
            val_loss += criterion(val_pred, x_val) + 1.0*KLD + 0.01*L_sup
            
          
        model.train()   

        if val_loss/len(val_loader.dataset) < best_loss:
                print('Model updated:', epoch)
                print('Validation loss:', val_loss.item()/len(val_loader.dataset))
                best_loss = val_loss/len(val_loader.dataset)
                best_model=copy.deepcopy(model.state_dict())
                           
        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_KLD = running_kld / len(train_loader.dataset)
        epoch_sup = running_sup / len(train_loader.dataset)
        tb.add_scalar('Loss', epoch_loss, epoch)
        tb.add_scalar('KLD', epoch_KLD, epoch)
        tb.add_scalar('Validation Loss', val_loss.item()/len(val_loader.dataset), epoch)      
        tb.add_scalar('Learning rate', scheduler.get_lr()[0] , epoch)

        
        print('Epoch {}/{} Loss: {:.4f}  Epoch KLD: {:.4f}  Epoch Parameter loss {:.4f}'
                        .format(epoch+1,num_epochs, epoch_loss, epoch_KLD, epoch_sup)) 
        scheduler.step() 
    return epoch_loss, epoch_KLD, best_model







def train_lstm(model, criterion, optimizer, data_loader, num_epochs, val_loader):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    best_loss = 1e6
    for epoch in range(num_epochs):
        running_loss = 0.0 
        model.train()
        for x,  y in data_loader:
            optimizer.zero_grad()
            
            y_pred = model(x)[:, -1, :]
        
            loss_train = criterion(y_pred, y[:, -1, :])

            
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

def train_comb(model, criterion, optimizer, train_loader, val_loader, time, KLD_weight, num_epochs, scheduler):
    best_loss = 1e8
    model.train()
    for epoch in range(num_epochs):

        running_loss = 0.0
        
        KLD_loss = 0

        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred, hx, mu, log_var = model(x,None,t=time)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            breakpoint()
            rec_loss = criterion(y_pred, y)
            loss =  rec_loss + KLD * KLD_weight
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            KLD_loss += KLD.item()
        epoch_loss = running_loss/len(train_loader)
        epoch_KLD = KLD_weight*KLD_loss/len(train_loader)
        print('Epoch {}/{} Rec Loss: {:.4f}  Epoch KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss-epoch_KLD, epoch_KLD)) 
        val_loss = 0
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_val_pred, hx, mu, log_var = model(x_val, None,t=time)
            KLD_val = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            val_loss += criterion(y_val_pred, y_val) + KLD_weight * KLD_val

            print('Validation loss:', val_loss.item()/len(val_loader))

        if val_loss.item()/len(val_loader) < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model, 'model_comb.pth')
            print('Model updated:', epoch)
        scheduler.step(val_loss.item()/len(val_loader))
    return best_model


def train_seq(model, VAE, criterion, optimizer, train_loader, val_loader, num_epochs, model_name,
               tf_prob_max=0.6, tf_prob_min=0.1): 
    VAE.to(device)
    #needs to be recomputed at each iteration over the dataset
    

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    best_loss = 1e8 
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        val_loss = 0 
        tf_prob_t = max(tf_prob_max - epoch*0.01, tf_prob_min)
        for i,(x, y, target) in enumerate(train_loader):
            
             
            x = x.to(device)
            y = y.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            
            
            y_pred_lat = model(x, target, tf_prob_t)
            
            y_pred_lat = torch.reshape(y_pred_lat,(y_pred_lat.shape[0]*y_pred_lat.shape[2],1
                                                   ,y_pred_lat.shape[3]))
            
            y_pred = VAE.decoder(y_pred_lat)
            
            
            y_pred = torch.reshape(y_pred, (y.shape))
            y_pred = torch.mean(y_pred, dim=-2)
            y = torch.mean(y, dim=-2)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        
        for x_val, y_val, target in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            target = target.to(device)
            y_val_pred_lat = model(x_val, target, tf_prob=0.0)
            
            y_val_pred_lat = torch.reshape(y_val_pred_lat,(y_val_pred_lat.shape[0]*y_val_pred_lat.shape[2],1,
                                                           y_val_pred_lat.shape[3]))
            y_val_pred = VAE.decoder(y_val_pred_lat)
            y_val_pred = torch.reshape(y_val_pred, (y_val.shape))
            val_loss =+ criterion(y_val_pred, y_val)
        if val_loss.item()/len(val_loader.dataset) < best_loss:
                best_loss = val_loss/len(val_loader.dataset)
                print('Model updated:', epoch)
                best_model = copy.deepcopy(model)
                torch.save(best_model, model_name)

        print('Epoch {}/{} Loss: {:.4f} '.format(epoch+1,num_epochs, epoch_loss/len(train_loader.dataset)))
        print('Validation loss:', val_loss.item()/len(val_loader.dataset))
        scheduler.step(val_loss.item()/len(val_loader.dataset))
    return best_model

