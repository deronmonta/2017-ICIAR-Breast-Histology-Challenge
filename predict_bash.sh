#!/bin/bash

#SBATCH --job-name=deep_learn
#SBATCH --output=gpu_job.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:2
#SBATCH --partition=gpu
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=10:00

module load GPU/Cuda/8.0
module load GPU/cuDNN/8.0-v5.1
# using your anaconda environment
source activate tensorflow
python train_hpc.py