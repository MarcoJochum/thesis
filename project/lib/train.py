import torch
from torch.optim import lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
# Define the autoencoder architecture
torch.manual_seed(42) 

def train_vae(train_x, model, criterion, optimizer, train_loader, num_epochs):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    # Loop over the dataset multiple times
    model.train()
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
        
        
        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(len(train_loader.dataset))
        epoch_KLD = running_kld / len(train_loader.dataset)
        scheduler.step()
        print('Epoch {}/{} Loss: {:.4f}  Epoch KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss, epoch_KLD))          


def train_cae(train_x, model, criterion, optimizer, train_loader, num_epochs):
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



def train_lstm(model, criterion, optimizer, data_loader, num_epochs, x_test, y_test):
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for x, y in data_loader:
            optimizer.zero_grad()
            
            y_pred = model(x)
        
            loss_train = criterion(y_pred, y)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()    
        if epoch % 10 == 0:
            epoch_loss = running_loss / len(data_loader.dataset)
            print('Epoch {}/{} Loss: {:.4f} '.format(epoch+1,num_epochs, epoch_loss, ))

            model.eval()
            with torch.no_grad():
                y_pred = model(x_test)
                loss_test = criterion(y_pred, y_test)
                print('Test loss:', loss_test.item())
    