#!/bin/bash
#SBATCH --job-name=train_3d_surr
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:A100_SXM4"
#SBATCH --mem=32G

module load python/3.9
module load cuda/12.1.1
module load cudnn/8.9.0.131
module load py-pytorch/1.11.0_py39
module load py-numpy/1.24.2_py39
module load py-h5py/3.7.0_py39

# Launch training or hyperopt
#python3 -u main.py train
# OR for hyperopt:
python3 -u main.py train

nvidia-smi