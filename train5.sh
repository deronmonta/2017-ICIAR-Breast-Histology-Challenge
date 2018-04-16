#!/bin/bash
#SBATCH --job-name=hy294_train5
#SBATCH --output=gpu_job5txt
#SBATCH --mail-type all --mail-user hao-yu.yang@yale.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=5g
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding

export PATH=$HOME/anaconda2/bin:$PATH
module load GPU/Cuda/8.0
module load GPU/cuDNN/8.0-v5.1
# using your anaconda environment
source activate tensorflow
python train_hpc.py