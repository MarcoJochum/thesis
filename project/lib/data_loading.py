import pandas as pd
import os 
import numpy as np
import warnings
import re
import torch


class DataLoading:
    def __init__(self, data_folder, n_prod, n_time, n_x, n_y, n_z, suffix, run_type="2d"):
        self.data_root = data_folder
        self.kmc_data = data_folder + run_type + "/"
        self.suffix = suffix
        self.n_prod = n_prod
        self.n_time = n_time
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        #self.config_list = makelis
    def load_files(self, file_path):


        ''' 
        Load kMC data from output file and return as a 3D grid corresponding 
        to the x, y, z dimensions of the simulation grid.

        Parameters:
            file_path (str): Path to the kMC output file
        
            n_x (int): Length of the x dimension of the simulation grid
        
            n_y (int): Length of the y dimension of the simulation grid
        
            n_z (int): Length of the z dimension of the simulation grid without contacts

            Compute index in original configuration: coord = n_y*n_z*x_0  + z_0+ y_0*n_z
        '''
        with warnings.catch_warnings():#Ignore dtype warning because of Nan at beginning of file
            warnings.simplefilter("ignore")
            data = pd.read_csv(file_path, sep="\s+", header=None, on_bad_lines="skip")
            # try:
            #     data = pd.read_csv(file_path, sep="\s+", header=None, on_bad_lines="skip")
            # except pd.errors.ParserError as e:
            #     print(f"Error parsing {file_path}: {e}")
            #     return None, None
    
        # Convert all data to numeric, replacing non-numeric values with NaN
        #Replacing initial 3 NAN values with Nan
        data = data.apply(pd.to_numeric, errors='coerce')
        

        data = data.values

        assert data.shape[1] -1 == self.n_x*self.n_y*self.n_z, "Data dimension does not match grid dimensions. Check n_x, n_y, n_z."
        #store time data
        time = data[3:,0]
        
        grid=np.ones((time.shape[0],self.n_x,self.n_y,self.n_z))  
    

        for i in range(time.shape[0]):

            grid[i] = data[i+3, 1:].reshape(self.n_x,self.n_y,self.n_z)#+3 needed to skip the nan values at the beginning that contain the coordinates

        
        return grid, time

    def avg_trj(self, config_folder):
        '''
        Average the trajectories of the kMC data and save the averaged trajectory data in 
        the same directory as the original data.

        Parameters:
            
            config_folder (str): Name of the folder containing a specific parameter configuration.
            
        '''

        
        mean_config = np.zeros((self.n_time,self.n_x,self.n_y,self.n_z)) 
        #Needs to be defined because of += operation
        

            
        full_path = self.kmc_data + config_folder

        if os.path.isdir(full_path): ##Only read directories
            print("Currently loading:", config_folder)
            c_bulk = self.find_c(config_folder) #Extract concentration from folder name

            for i in range(self.n_prod):
                prod_path = "prod_" + str((i+1)) + "/"
                file_path = full_path+ "/" + prod_path + self.suffix
                if not os.path.exists(file_path): #Catch non-existing production runs
                    print("File does not exist: ", file_path)
                    continue
                grid, time = self.load_files(file_path)
                if grid is None or time is None:
                    print("Failed to load files.")
                    return
                grid = self.padding(grid)
                mean_config += grid/c_bulk
                if i % 10 == 0:
                    print("Progress: ", i/self.n_prod*100, "%")
                

            mean_config = mean_config/self.n_prod
            #mean_config = mean_config.squeeze(2)
            np.save(self.kmc_data + config_folder + "/avg_trj", mean_config)
            

        #return mean_config

    def make_data_set(self, file_name=None, list_configs=None):
        '''
        Creates data set from the previously averaged and safed trajectories for different configs.

        Parameters:

            file_name (str): Name of the file to save the data set in. 
            Default is None, which does not save the data set but returns the data set.
            
            list_configs (list): List of configuration folders to load. 
            Default is None, which loads all folders in the data folder.
        '''
        grids = []
        files = []
        counter = 0
        if list_configs:
            
            for config_folder in list_configs:
                counter += 1
                print("Currently loading from list:", config_folder)
                full_path = self.kmc_data + config_folder
                if os.path.isdir(full_path): ##Only read directories
                    file_path = full_path + "/avg_trj.npy"
                    ##check if file path exists

                    if not os.path.exists(file_path):
                        print("File does not exist: ", file_path)
                        continue
                    grid = np.load(file_path)
                    grid = np.reshape(grid, (grid.shape[0],grid.shape[2],grid.shape[1],grid.shape[3]))                     
                    grids.append(grid)
                    files.append(config_folder)

            
        else:
            for config_folder in os.listdir(self.kmc_data):
                counter += 1
                print("Currently loading:", config_folder)
                full_path = self.kmc_data + config_folder
                if os.path.isdir(full_path): ##Only read directories
                    file_path = full_path + "/avg_trj.npy"
                    grid = np.load(file_path) 
                    grid = np.reshape(grid, (grid.shape[0],grid.shape[2],grid.shape[1],grid.shape[3]))               
                    grids.append(grid)
                    files.append(config_folder)
        data =np.stack(grids)
        print("Data set shape: ", data.shape, "Number of files combined", counter)
        with open(self.data_root + "2d_sets/" + file_name +"_list.txt", "w") as f:
            for name in files:
                f.write(name + "\n")
        if file_name:
            np.save(self.data_root+"2d_sets/"+file_name, data)
            return torch.tensor(data, dtype=torch.float32)
        else:
            return torch.tensor(data, dtype=torch.float32)

        
    def padding(self, grid, pad_value=0):

        '''
        Pad the grid with zeros to match the time dimension of the grid with the maximum time dimension. 
        Sometimes kMC simulations save different number of timesteps for different configurations.

        Parameters:

            grid (np.array): Grid to be padded with zeros.

            pad_value (int): Value to pad the grid with. Default is 0.
        '''
        
        t_padding = 1000 - grid.shape[0]
        if t_padding > 0:
            print("Padding with zeros. Padding size: ", t_padding, " rows.")
        pad_width = ((0,t_padding), (0,0), (0,0), (0,0))
        padded_grid = np.pad(grid, pad_width, 'constant', constant_values=pad_value)
        return padded_grid
            
    def find_c(self, config_folder):
        '''
        Extract c_bulk from folder name.

        Parameters:
            config_folder (str): Name of the folder containing a specific parameter configuration. Needs to 
            have Format: epsr_"c_bulk"_"v_bias" 
        '''
        pattern = r'_(.*?)_' 
        finds = re.search(pattern, config_folder)

        if finds:
            c = finds.group(1)
            print("Concentration extracted: ", float(c))
            return float(c)
        else:
            print("No concentration found in folder name.")
            exit()
    def make_folder_list(self, file_name="folder_names.txt"):

        '''
        Create a file with all folder names in the data folder. 
        Use this to create specific data sets with make_data_set.

        Parameters:

            file_name (str): Name of the file to save the folder names in. Default is "foilder_names.txt".
        
        '''
        folder_names=[]
        for config_folder in os.listdir(self.kmc_data):
            if os.path.isdir(self.kmc_data + config_folder):
                folder_names.append(config_folder)

        with open(self.data_root + file_name, "w") as f:
            for folder in folder_names:
                f.write(folder + "\n")

        return folder_names
        
def saving_encodings(model, data, config_list):

    '''
    Save the encoded data for the different configurations in the data folder.
    This is the training data for the LSTM model.

    Parameters:

        model (torch.nn.Module): Autoencoder model to encode the data.

        data (np.array): Data set to encode.

        config_list (list): List of all the configuration names 


    '''

    assert data.shape[0] == len(config_list), "Data and config list do not match in length."
    model.eval()
    for i in range(data.shape[0]):
        print("Currently encoding config:", config_list[i])
        x_config = torch.tensor(data[i], dtype=torch.float32)  
        encoding = model.encoder(x_config)
        np.save("../../data_kmc/2d_encoded/" + config_list[i], encoding.detach().numpy())
        print("Encoded shape:", encoding.shape)

    with open("../../data_kmc/2d_encoded/encoding_model.txt", "w") as f:
        f.write("Model used for encoding: " + model.__class__.__name__ + "\n")

def loading_encodings(folder_name, config_list):
    '''
    Load the encoded data for the different configurations in the data folder.
    This is the training data for the LSTM model.

    Parameters:

        folder_name (str): Name of the folder containing the encoded data.

       
    '''
    encodings = []
    for config in config_list:
        encoding = np.load(folder_name + config)
        encodings.append(encoding)
    encodings = np.stack(encodings)
    return encodings

def get_config(config_file_name):
    '''
    Load the configuration file and return the configuration dictionary.

    Parameters:

        config_file_name (str): Name of the configuration file to load.
    '''
    lines_array = []

    # Open the file and read the lines
    with open(config_file_name, 'r') as file:
        lines_array = [line.strip() for line in file.readlines()]
    # Now lines_array contains each line of the file as an element
    split_lines = []
    # Iterate over each line in the array
    for line in lines_array:
        # Split the line by underscore and add the resulting list to split_lines
        split_lines.append(line.split("_"))
    # Now split_lines contains each line of the file split into parts
    
    lines_array= np.array(split_lines)
    lines_array = lines_array.astype(np.float32)
    return torch.tensor(lines_array)