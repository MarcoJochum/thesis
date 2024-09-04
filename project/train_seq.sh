#!/bin/bash

#SBATCH --partition=gpucloud
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=10       # Requested CPUs per task. Default=1
#SBATCH --mem=50G                # Memory limit. [1-999][K|M|G|T]
#SBATCH --mem-per-cpu=10G         # Memory limit per CPU. [1-999][K|M|G|T]
#SBATCH --time=12:00:00         # Time limit. [[days-]hh:]mm[:ss]
#SBATCH --gpus-per-task=1
### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=seq.log

### configure email notifications
### mail types: BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80
#SBATCH --mail-user=marco.jochum@tum.de
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name=seq_train
#SBATCH --comment="Training seq2seq on kmc data"

### load environment modules


source ../../thesis_env/bin/activate    

### run your program...
srun python train_seq.py
