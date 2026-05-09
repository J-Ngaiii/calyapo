#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_gen_geoplot
#SBATCH --account=ic_datah195
#SBATCH --partition=savio2_1080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Processors per task:
# Eight times the number for A40 in savio3_gpu
# Four times the number of GPUs for A500 in savio4_gpu
# Two times the number of GPUs for 1080ti in savio2_1080ti
#SBATCH --cpus-per-task=2

#Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --qos=savio_normal

# Wall clock limit:
#SBATCH --time=1:00:00

#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

echo "nvidia-smi check:"
nvidia-smi

# --- Environment Setup ---
# Create the directory specifically named 'slurm' for the #SBATCH output logs
mkdir -p slurm
mkdir -p logs

# don't exhaust ur memory
export HF_HOME="/global/scratch/users/jonathanngai/hf_cache"
export TORCH_HOME="/global/scratch/users/jonathanngai/torch_cache"
mkdir -p $HF_HOME $TORCH_HOME

# Navigate to your project directory
cd /global/home/users/jonathanngai/calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not change directory. Exiting."
  exit 1
fi

# Activate your virtual environment
source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not activate virtual environment. Exiting."
  exit 1
fi

TRAIN_PLAN='presidents_to_abortion'
SPLIT='val'
python scripts/data_eval/report.py --train_plan=${TRAIN_PLAN} \
    --split=${SPLIT} \
    --only_geo