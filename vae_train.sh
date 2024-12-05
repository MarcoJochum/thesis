#!/bin/bash
#SBATCH --partition=wom
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=10       # Requested CPUs per task. Default=1
#SBATCH --ntasks-per-node=1     # Requested tasks per node. Default=1
#SBatch --nodes=1
#SBATCH --gpus=1                # Number of GPUs to use
#SBATCH --mem-per-cpu=5G      # Memory limit per CPU. [1-999][K|M|G|T]
#SBATCH --time=4-12:00:00         # Time limit. [[days-]hh:]mm[:ss]


### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=logs/vae_std_hyper.log

### configure email notifications
### mail types: BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80
#SBATCH --mail-user=marco.jochum@tum.de
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,TIME_LIMIT_90

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name="vae_std_hyper"
#SBATCH --comment="VAE hyperparameter search"




# Define the new output path


# Use jq to modify the output_path in the input.json file


### load environment modules
source ../../thesis_env/bin/activate
module load python/3.9-torch2-cuda12
module load gcc/13.2.0
module load oneapi/tbb/2021.7.0
module load openmpi/4.1.4

srun python train_vae.py