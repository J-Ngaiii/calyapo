#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_finetune_savio 
#SBATCH --account=fc_hartmanl2
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1 # keep as 1

# Processors per task:
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=8

#Number of GPUs
#SBATCH --gres=gpu:A40:1 
#SBATCH --qos=a40_gpu3_normal

# Wall clock limit:
#SBATCH --time=60:00:00

#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd /global/home/users/jonathanngai/calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not change directory. Exiting."
  exit 1
fi

source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not activate virtual environment. Exiting."
  exit 1
fi

echo nvidia-smi call:
nvidia-smi