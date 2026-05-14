#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_inference_os 
#SBATCH --account=fc_hartmanl2
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Processors per task:
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=8

#Number of GPUs
#SBATCH --gres=gpu:A40:2
#SBATCH --qos=a40_gpu3_normal

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
conda activate calyapo
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

MODEL_NAME="socratesft/socrates-qwen2.5-14b-dpo"
MODEL_NICKNAME="socrates-qwen2.5-14b-dpo"
MODEL_TYPE="socrates_qwen" 

# MODEL_NAME="socratesft/socrates-llama3-8b-sft"
# MODEL_NICKNAME="socrates-llama3-8b-sft"
# MODEL_TYPE="socrates_llama" 

# SPLIT="train"
# SPLIT="val"
SPLIT="test"

RUN_KEYWORD="archon"
NUM_GPUS=2
CHUNK_SIZE=2000

python scripts/llm/offline_inf.py --train_plan=${TRAIN_PLAN} \
    --model_name=${MODEL_NAME} \
    --run_keyword=${RUN_KEYWORD} \
    --model_type=${MODEL_TYPE} \
    --split=${SPLIT} \
    --num_gpus=${NUM_GPUS} \
    --chunk_size=${CHUNK_SIZE} \