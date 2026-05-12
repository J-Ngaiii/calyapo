#!/bin/bash
# from https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/
#SBATCH --job-name=calyapo_inference_p2a
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

echo "nvidia-smi check:"
nvidia-smi

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
MODEL_NAME="gemini-2.5-flash"
MODEL_NICKNAME="gemini-2.5-flash"
MODEL_TYPE="gemini" 
# SPLIT="train"
# SPLIT="val"
SPLIT="test"

RUN_KEYWORD="archon"
NUM_GPUS=1
CHUNK_SIZE=2000

python scripts/llm/offline_inf.py --train_plan=${TRAIN_PLAN} \
    --model_name=${MODEL_NAME} \
    --run_keyword=${RUN_KEYWORD} \
    --model_type=${MODEL_TYPE} \
    --split=${SPLIT} \
    --num_gpus=${NUM_GPUS} \
    --chunk_size=${CHUNK_SIZE} \