import pandas as pd
import os 
import numpy as np

class DataLoading:
    def __init__(self, data_folder, n_prod, n_time, n_x, n_y, n_z, suffix, config_folder=None):
        self.data_folder = data_folder
        self.suffix = suffix
        self.n_prod = n_prod
        self.n_time = n_time
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

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

        data = pd.read_csv(file_path, sep="\s+", header=None)

        # Convert all data to numeric, replacing non-numeric values with NaN
        #Replacing initial 3 NAN values with Nan
        data = data.apply(pd.to_numeric, errors='coerce')
        

        data = data.values

        assert data.shape[1] -1 == self.n_x*self.n_y*self.n_z, "Data dimension does not match grid dimensions. Check n_x, n_y, n_z."
        #store time data
        time = data[3:,0]
        
        grid=np.ones((time.shape[0],self.n_x,self.n_y,self.n_z))  
    

        for i in range(time.shape[0]):

            grid[i] = data[i+3, 1:].reshape(self.n_x,self.n_y,self.n_z)

        
        return grid, time

    def avg_trj(self):
        '''
        Load kMC data from output file and return as a 3D grid corresponding 
        to the x, y, z dimensions of the simulation grid.

        Parameters:
            data_folder (str): Path to the kMC data
        
            config_folder (str): Name of the folder containing the configuration files
        
            suffix (str): SUffix of trajectory files
        
            n_prod (int): Number of trajectories to average

            n_time (int): Number of time steps in the simulation

            n_x (int): Length of the x dimension of the simulation grid
        
            n_y (int): Length of the y dimension of the simulation grid
        
            n_z (int): Length of the z dimension of the simulation grid without contacts

            
        '''

        c_limit = 1e21
        mean_config = np.zeros((self.n_time,self.n_x,self.n_y,self.n_z)) 

        for config_folder in os.listdir(self.data_folder):
            print("Currently loading:", config_folder)
            full_path = self.data_folder + config_folder
            if os.path.isdir(full_path): ##Only read directories
            
                for i in range(self.n_prod):
                    prod_path = "prod_" + str(i) + "/"
                    file_path = full_path+ "/" + prod_path + self.suffix
                    grid, time = self.load_files(file_path)
                    mean_config += grid/c_limit
                    if i % 10 == 0:
                        print("Progress: ", i/self.n_prod*100, "%")
                

            mean_config = mean_config/self.n_prod
            #mean_config = mean_config.squeeze(2)
            np.save(self.data_folder + config_folder + "/avg_trj", mean_config)
            

        #return mean_config

    def load_data(self):
        '''
        Load the averaged trajectory data from a directory containing the averaged trajectory data.
        '''
        grids = []
        for config_folder in os.listdir(self.data_folder):
            print("Currently loading:", config_folder)
            full_path = self.data_folder + config_folder
            if os.path.isdir(full_path): ##Only read directories
                file_path = full_path + "/avg_trj.npy"
                grid = np.load(file_path)
                print(grid.shape)
                
                #return grid
            grids.append(grid)
        data =np.stack(grids)

        return data
            


