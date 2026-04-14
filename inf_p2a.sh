#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_finetune_savio 
#SBATCH --account=fc_hartmanl2
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2

# Processors per task:
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=8

#Number of GPUs
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal

# Wall clock limit:
#SBATCH --time=10:00:00

#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# --- Environment Setup ---
# Create the directory specifically named 'slurm' for the #SBATCH output logs
mkdir -p slurm
mkdir -p logs

# Navigate to your project directory
cd /global/home/users/jonathanngai/calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not change directory. Exiting."
  exit 1
fi

# Activate your virtual environment
source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate calypo
if [ $? -ne 0 ]; then
  echo "Error: Could not activate virtual environment. Exiting."
  exit 1
fi

# Export API keys
if [ -f .env ]; then 
  export $(grep -v '^#' .env | xargs)
fi

# Distributed Setup
NPROC_PER_NODE=1                     
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) # Random port to avoid collisions

# Model/Data Params for opinion_school
TRAIN_PLAN="presidents_to_abortion"
MODEL_NAME="meta-llama/Llama-3.1-8B"
MODEL_NICKNAME="llama3.1-8b"
ADAPTER_FOLDER="llama3.1-8b_wd0.1_gam0.85_lr1e-05_2026-04-14-01-53-07AM"

# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NICKNAME="llama3.1-8b-Instruct" 
# ADAPTER_FOLDER="llama3.1-8b-Instruct_wd0.1_gam0.85_lr1e-05_2026-04-14-05-26-04AM"

# MODEL_NAME="meta-llama/Llama-3.2-3B"
# MODEL_NICKNAME="llama3.2-3b"
# ADAPTER_FOLDER="llama3.2-3b_wd0.1_gam0.85_lr1e-05_2026-04-13-04-39-12PM"

# MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NICKNAME="llama3.2-3b-Instruct"
# ADAPTER_FOLDER="llama3.2-3b-Instruct_wd0.1_gam0.85_lr1e-05_2026-04-13-04-38-50PM"

# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NICKNAME="llama2-7b" 

python scripts/llm/offline_inf.py --train_plan=${TRAIN_PLAN} \
    --adapter_folder=${ADAPTER_FOLDER} \
    --model_name=${MODEL_NAME} \
    --model_nickname=${MODEL_NICKNAME} \