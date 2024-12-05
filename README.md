# kMC surrogate documentation
## Introduction

This is the work of my master's thesis on improving the efficiency of an existing C++ kMC code for modeling Solid State Electrolytes. The data was generated using the openkmc code, which is being developed at the TUM group for Simulation of Nanosystems for Energy Conversion. This code will be published by the chair at a later point. However, its results have been discussed in several papers:
- [Modeling of Space-Charge Layers in Solid-State Electrolytes: A Kinetic Monte Carlo Approach and Its Validation](https://pubs.acs.org/doi/10.1021/acs.jpcc.2c02481)
- [Local Temporal Acceleration Scheme to Couple Transport and Reaction Dynamics in Kinetic Monte Carlo Models of Electrochemical Systems](https://pubs.acs.org/doi/10.1021/acs.jctc.1c01010)

In the following a detailed explanation of the surrogate model workflow will be given.

## Rough workflow
1. Start openkmc simulation to obtain data to train ML model
2. Postprocess data from kMC and create data sets for ML model
3. Train VAE with data sets
4. Train TFT with encoded configs from VAE
5. Postprocess for visualization



## 1. Openkmc
### General info
- To run the kMC code 3 files are needed
    1. `grid.json`
    2. `input.json`
    3. `pair_potential.json`
- `openkmc/materials` contains a collection of base materials for the electrode and electrolyte material

### Step-by-step walk-through
1. Move to `openkmc/build`
   - Two scripts to create input files for kmc
     1. `solid_electrolytes.py`: Standard script to create input files for one parameter configuration
     2.  `solid_electrolytes_many.py`: Automatically creates a range of input files for the permutation of specified input parameters
         - Currently only works for one $\varepsilon_r$ at a time, here the material file needs to be change specifically
   - Script needs to be run twice so all three input files are created (see above)
   - Creates input files in `openkmc/input/bo/2d/epsr_cbulk_Vbias` e.g. for $\varepsilon_r=1400, c_{\text{bulk}}=1e19, \phi_{\text{bias}}=1.5$ the folder name would be `openkmc/input/bo/2d/1400_1e+19_1.5`
   - Running `solid_electrolytes_many.py` also creates a file in `openkmc/input/` containing a list of all the configuration names
2. Move to `openkmc/` and run `create_input_files.py`
   - This creates `n_runs` number of seperate input files for all the parallel kMC runs with the same configuration
   - The input files differ only in the output folder so that the kMC outputs are saved to seperate trajetories
   - After running this each `openkmc/input/bo/2d/epsr_cbulk_Vbias` should look something like this 
  ```plaintext
     openkmc/input/bo/2d/1400_1e+19_1.5/
     ├── prod_1
        └── input.json  
     ├── prod_2
        └── input.json  
     ├── ...
     └── prod_n
        └── input.json
```
3. Now we can go into the `run.sh` file to start the kMC runs
    - Here we specify at the botton the text file containing our list of input parameter combinations that was created in **Step 1** 
    - when running this bash script we also need to specify the number of different parameter combinations we want to simulate
    - This bash script then calls the sbatch script `parallel_trj.sh` to actually send the kMC simulation to the cluster
    - The output files should then start to appear in `openkmc/output/bo/2d/epsr_cbulk_Vbias` like so:
  ```plaintext
     openkmc/output/bo/2d/1400_1e+19_1.5/
     └──prod_1
        ├──average_density_an-_block.txt
        ├──average_density_an-.txt
        ├──average_density_Li+_block.txt
        ├──average_density_Li+.txt
        ├──average_potential.txt
        ├──event_counters.txt
        ├──local_density_an-_block.txt
        ├──local_density_an-.txt
        ├──local_density_Li+_block.txt
        ├──local_density_Li+.txt
        ├──local_potential_block.txt
        └──local_potential.txt
    ...
```

## 2. Data postprocessing

### General info
- To use the kMC data for our ML surrogate we want to reshape the output data and save it in a format that is easily procesable in python/pytorch
- When fitting on the mean and standard deviation field we also need to compute these from the `n_runs` trajectories
- All of the standard data loading and handling should be done with the custom `DataLoading` class that can be found in `lib/data_loading.py`

### Step-by-step walktrough

1. Move the data from `openkmc/output/bo/2d` to `data_kmc/folder_name`
2. Now we need to load all the trajectories for each configuration, rehape, and combine them
   - For this we use the `lib/dl_submitit.py` script which allows us to load and proces the files in parallel on the cluster using [Submitit](https://github.com/facebookincubator/submitit)
   - Here we need to set the `data_folder`, `suffix`, and `run_type` variables to specify where the data that we want to load is stored
   - This should take around 5 minutes to complete
   - If everything worked corretly in every configuration folder `data_kmc/folder_name/epsr_cbulk_Vbias/` a `avg_trj.npy` and a `std_trj.npy` file should appear
   - These files have the shape `n_time_steps x X_dim x Z_dim`
3. After reshaping and combining the data now we want to create data sets which we can easily use to train our surrogate model
   - For this we use the `lib/data_set_split.py` script
   - We split the different parameter combinations by shuffling them randomly and then splitting them according to our desired train/test ratio i.e. 80/20
   - we use the `make_data_set()` method of our `DataLoading` class to create the data sets which are usually stored in `data_kmc/2d_sets`
   - Now a data set has the shape `n_configs x n_time_steps x X_dim x Z_dim` i.e. for the standard case that is implemented `84 x 1000 x 50 x 100` (for the train set)
  

  ## 3. Create and train VAE model
  ### General info
  - VAE class is created from CAE base class in `NNs/autoencoder.py`. CAE is a autoencoder with convolutional layers
  - The layers are defined in `NNs/ae_layers.py`
    - current model uses `pro_encoder2d` and `pro_decoder2d`
    - custom encoder/decoder classes can be created easily by following the structure in the existing ones and then just instatiating them when creating the VAE class
  - the `train_vae.py` script is designed for hyperparameter tuning and therefore runs the training in a loop over different hyperparameter combinations, however just specifying on combination also works to train a single VAE model
  - `config/vae.py` contains a class with all the parameters of the VAE model
    - before training set the desired parameters here

### Step-by-step walktrough

1. Set the desired VAE parameters in the `config/vae.py` file
2. Set the file name for the model at the bottom of the `train_vae.py` script
3. Run the `train_vae.py` script on a GPU to train a VAE on the kMC data with the `--data_type` argument specify wether you want to train on the mean (`avg`) or standard deviation (`std`) field 



## 4. Create and train TFT model

### General info
- The [Darts library](https://unit8co.github.io/darts/) is used for creating this model as it contains a ready-to-use implementation of the TFT algorithm
- our data is converted into the `TimeSeries` class from Darts

### Step-by-step walktrough
1. Run the `train_tft.py` script on the GPU specifying the again the `--data_type` argument


## 5. Postprocess for visualization

### General info

- All model evaluation and visualization scripts are stored in two folders: `exp_1` and `eval_01`
  - `exp_1/eval_tft.py`: performs inference using the specified `tft_model`
    - predictions are stored generally in `data_kmc/2d_results/lin_time/`
    - inference takes around 80 seconds therefore it is easier to do it once and then used the stored prediction for further evaluation
  - VAE evaluation is fast and always done in place
- `eval_01/` contains a number of evaluation/ plotting scripts used for my master thesis
  - `eval_01/eval_error.py` computes the MAPE, SMAPE and MSE for the prediction relative to the ground trouth
  - `eval_01/latent_analysis.py` performs PCA and [t-SNE](https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html) on the encoded (both predicted and ground truth) data and saves it in `eval_01/data` 
  - `eval_01/loss_plot.py` plots validation and train loss of the tft model
  - `eval_01/plot_data_3d.py` visualizes the 3D parameters space spanned by $\varepsilon_r, c_{\text{bulk}}, \phi_{\text{bias}}$. This shows the distribution of training/test paramters used in this 3D space
  - `eval_01/plot_error.py` visualizes the error of the surrogate model over time across the SSE cell
  - `eval_01/plot_latent.py` uses the results of `eval_01/latent_analysis.py` to visualize a 2D projection of the trajectories with both PCA and t-SNE
