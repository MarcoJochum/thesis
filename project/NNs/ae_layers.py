import torch
import torch.nn as nn

class encoder2d(nn.Module):
    def __init__(self,num_input_channels=1, base_channel_size=16, latent_dim=10):
        super().__init__()


        '''
        Simple encoder module for a 2D convolutional autoencoder.

        Args:
            num_input_channels (int): The number of input channels in the input data.

            base_channel_size (int): The number of channels in the first convolutional layer.

            latent_dim (int): The dimensionality of the latent space.

        
        '''



        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(), 
            #nn.MaxPool2d(2, stride=2),  # apply max pooling with a kernel size of 2 and a stride of 2
            nn.Conv2d(c_hid, 2*c_hid, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),  
            #nn.MaxPool2d(kernel_size=2, stride=2)  # apply max pooling with a kernel size of 2 and a stride of 2
            torch.nn.Flatten(), 
            torch.nn.Linear((2*c_hid*7*7), latent_dim) # 
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
class decoder2d(nn.Module): 
    def __init__(self,num_input_channels=1 ,base_channel_size=16, latent_dim=10):
        super().__init__()


        '''

        Simple decoder module for a 2D convolutional autoencoder.

        Args:

            num_input_channels (int): The number of input channels in the input data.

            base_channel_size (int): The number of channels in the first convolutional layer.

            latent_dim (int): The dimensionality of the latent space.

        '''
        c_hid = base_channel_size

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*c_hid*7*7),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, c_hid, 3, stride=2, padding=1, output_padding=1),  #7x7 -> 14x14 
            
            nn.ReLU(),# apply the ReLU activation function
            nn.ConvTranspose2d(c_hid, num_input_channels, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  #because of normalized inputs
            
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.shape[0], -1, 7, 7))
        x = self.net(x)
        return x
    

class pro_encoder2d(nn.Module):
    def __init__(self, num_input_channnels=2, base_channel_size=16, latent_dim=10, activation=nn.ELU()):
        super().__init__()

        '''
        More complex encoder module for a 2D convolutional autoencoder.

        Args:  

            num_input_channels (int): The number of input channels in the input data.

            base_channel_size (int): The number of channels in the first convolutional layer.

            latent_dim (int): The dimensionality of the latent space.

            activation (torch.nn.Module): The activation function to use in the encoder.
        
        
        '''
        c_hid = base_channel_size
        self.activation = activation
        self.net = nn.Sequential(

            nn.Conv2d(num_input_channnels, c_hid, kernel_size=3, stride=2, padding=1),#25x25
            self.activation,

            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, stride=2, padding=1), #13x13
            self.activation,

            #nn.ConstantPad3d((0, 1, 0, 0), 0),
            nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, stride=2, padding=1),#7x7
            self.activation,

            # #nn.ConstantPad3d((0, 0, 0, 1), 0),
            # nn.Conv2d(4*c_hid, 8*c_hid, kernel_size=3, stride=2, padding=1),
            # self.activation,

            # #nn.ConstantPad3d((0, 1, 0, 0), 0),
            # nn.Conv2d(8*c_hid, 16*c_hid, kernel_size=3, stride=2, padding=1),
            # self.activation,

            # #nn.ConstantPad3d((0, 0, 0, 1), 0),
            # nn.Conv2d(16*c_hid, 32*c_hid, kernel_size=3, stride=2, padding=1),
            # self.activation,

            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(4*c_hid*7*7, 256),
            self.activation,
#TODO: Adapt the input to the linear layer depending on the input data size
            nn.Linear(256, latent_dim ), 
        )
    def forward(self, x):
        x = self.net(x)
        return x
    

class pro_decoder2d(nn.Module):
    def __init__(self, num_input_channels=2, base_channel_size=16, latent_dim=10, activation=nn.ELU()):
        super().__init__()



        '''
        More complex decoder module for a 2D convolutional autoencoder.

        Args:

            num_input_channels (int): The number of input channels in the input data.

            base_channel_size (int): The number of channels in the first convolutional layer.

            latent_dim (int): The dimensionality of the latent space.

            activation (torch.nn.Module): The activation function to use in the decoder.

        '''
        c_hid = base_channel_size
        self.activation = nn.ELU()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 4*c_hid*7*7)
        )

        self.net = nn.Sequential(
            # nn.ConvTranspose2d(32*c_hid, 16*c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),
            # #nn.ConstantPad2d((0, 0, 0, -1), 0),
            # self.activation,

            # nn.ConvTranspose2d(16*c_hid, 8*c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),
            # #nn.ConstantPad2d((0, -1, 0, 0), 0),
            # self.activation,

            # nn.ConvTranspose2d(8*c_hid, 4*c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),
            # #nn.ConstantPad2d((0, 0, 0, -1), 0),
            # self.activation,

            nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),#7x7 to 13x13
            #nn.ConstantPad2d((0, -1, 0, 0), 0),
            self.activation,

            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),#13x13 to 25x25
            #nn.ConstantPad2d((0, 0, 0, 0), 0),
            self.activation,

            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),#25x25 to 50x50
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.linear(x)
        #TODO: Adapt the reshaping operation to the dimensions needed
        x = torch.reshape(x, (x.shape[0], -1, 7, 7))
        x = self.net(x)
        return x