import torch.nn as nn
import torch
# Define the autoencoder architecture
class AutoEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        
        # Set the number of hidden units
        self.num_hidden = hidden
        
        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),  # input size: 784, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
            nn.Linear(256, self.num_hidden),  # input size: 256, output size: num_hidden
            nn.ReLU(),  # apply the ReLU activation function
        )
        
        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),  # input size: num_hidden, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
            nn.Linear(256, 784),  # input size: 256, output size: 784
            nn.Sigmoid(),  # apply the sigmoid activation function to compress the output to a range of (0, 1)
        )

    def forward(self, x):
        # Pass the input through the encoder
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 28, 28)  # reshape the output to the same dimensions as the input
        # Return both the encoded representation and the reconstructed output
        return encoded, decoded
    

def train_loop(train_x, model, criterion, optimizer, train_loader, num_epochs):
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        
        # Loop over the data loader
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            data = data.unsqueeze(1)
            encoded, decoded, mu, log_var  = model(data)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, data) + 3* KLD

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*train_x.size(0)

        # Print the average loss for the epoche
        epoch_loss = running_loss / len(train_loader.dataset)
        print('Epoch {}/{} Loss: {:.4f}'.format(epoch+1,num_epochs, epoch_loss))          


class VAE(AutoEncoder):
    def __init__(self):
        super().__init__()

        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.logvar = nn.Linear(self.num_hidden, self.num_hidden)  

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std
    
    def forward(self,x):

        encoded = self.encoder(x)
        mu = self.mu(encoded)

        logvar = self.logvar(encoded)

        z = self.reparametrize(mu, logvar)

        decoded = self.decoder(z)

        return encoded, decoded, mu, logvar
    
    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.num_hidden)
            samples = self.decoder(z)
        return samples        
    


# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD