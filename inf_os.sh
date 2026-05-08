#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_inference_os 
#SBATCH --account=fc_hartmanl2
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Processors per task:
# Eight times the number for A40 in savio3_gpu
# Four times the number of GPUs for A500 in savio4_gpu
#SBATCH --cpus-per-task=4

#Number of GPUs
#SBATCH --gres=gpu:A5000:1
#SBATCH --qos=a5k_gpu4_normal

# Wall clock limit:
#SBATCH --time=10:00:00

#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

echo "nvidia-smi check:"
nvidia-smi

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
MODEL_NAME="meta-llama/Llama-3.1-8B"
MODEL_NICKNAME="llama3.1-8b"
ADAPTER_FOLDER="wdllama3.1-8b_wd0.1_gam0.85_lr1e-05_2026-05-03-06-46-46AM"

# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NICKNAME="llama3.1-8b-Instruct" 
# ADAPTER_FOLDER="wdllama3.1-8b-Instruct_wd0.1_gam0.85_lr1e-05_2026-05-03-04-12-29AM"

# MODEL_NAME="meta-llama/Llama-3.2-3B"
# MODEL_NICKNAME="llama3.2-3b"
# ADAPTER_FOLDER="wdllama3.2-3b_wd0.1_gam0.85_lr1e-05_2026-05-03-02-15-01AM"

# MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NICKNAME="llama3.2-3b-Instruct"
# ADAPTER_FOLDER="wdllama3.2-3b-Instruct_wd0.1_gam0.85_lr1e-05_2026-05-03-12-27-51AM"

# MODEL_NAME="Qwen/Qwen2.5-14B"
# MODEL_NICKNAME="qwen2.5-14b"
# ADAPTER_FOLDER="wdqwen2.5-14b_wd0.1_gam0.85_lr1e-05_2026-05-03-12-43-16AM"

# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# MODEL_NICKNAME="qwen2.5-14b-Instruct"
# ADAPTER_FOLDER="wdqwen2.5-14b-Instruct_wd0.1_gam0.85_lr1e-05_2026-05-03-12-43-51AM"

# MODEL_TYPE="base" 
MODEL_TYPE="lora" 
SPLIT="train"
# SPLIT="val"
# SPLIT="test"

RUN_KEYWORD="archon"
NUM_GPUS=1
CHUNK_SIZE=2000

python scripts/llm/offline_inf.py --train_plan=${TRAIN_PLAN} \
    --model_name=${MODEL_NAME} \
    --run_keyword=${RUN_KEYWORD} \
    --adapter_folder=${ADAPTER_FOLDER} \
    --model_type=${MODEL_TYPE} \
    --split=${SPLIT} \
    --num_gpus=${NUM_GPUS} \
    --chunk_size=${CHUNK_SIZE} \