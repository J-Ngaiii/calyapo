#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_inference_os 
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

# --- Storage Redirection ---
# Redirecting cache to scratch to avoid Home quota issues
export HF_HOME="/global/scratch/users/jonathanngai/hf_cache"
export TORCH_HOME="/global/scratch/users/jonathanngai/torch_cache"
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

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
TRAIN_PLAN="opinion_school"
# MODEL_NAME="meta-llama/Llama-3.1-8B"
# MODEL_NICKNAME="llama3.1-8b"
# ADAPTER_FOLDER="llama3.1-8b_wd0.1_gam0.85_lr1e-05_2026-04-12-04-58-33PM"

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NICKNAME="llama3.1-8b-Instruct" 
ADAPTER_FOLDER="llama3.1-8b-Instruct_wd0.1_gam0.85_lr1e-05_2026-04-12-04-41-42PM"

# MODEL_NAME="meta-llama/Llama-3.2-3B"
# MODEL_NICKNAME="llama3.2-3b"
# ADAPTER_FOLDER="llama3.2-3b_wd0.1_gam0.85_lr1e-05_2026-04-13-02-04-15AM"

# MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NICKNAME="llama3.2-3b-Instruct"
# ADAPTER_FOLDER="llama3.2-3b-Instruct_wd0.1_gam0.85_lr1e-05_2026-04-13-02-03-23AM"

# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NICKNAME="llama2-7b" 

MODEL_TYPE="base" 
# MODEL_TYPE="lora" 
SPLIT="train"
# SPLIT="val"
# SPLIT="test"

NUM_GPUS=1
CHUNK_SIZE=2000

python scripts/llm/offline_inf.py --train_plan=${TRAIN_PLAN} \
    --model_name=${MODEL_NAME} \
    --adapter_folder=${ADAPTER_FOLDER} \
    --model_type=${MODEL_TYPE} \
    --split=${SPLIT} \
    --num_gpus=${NUM_GPUS} \
    --chunk_size=${CHUNK_SIZE} \