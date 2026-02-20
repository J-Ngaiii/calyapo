#!/bin/bash
#SBATCH --job-name=finetune_calyapo
#SBATCH --output=slurm/calyapo_train_%j.out
#SBATCH --error=slurm/calyapo_train_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                      
#SBATCH --gres=gpu:1                    
#SBATCH --time=12:00:00

# --- Environment Setup ---
# 1. Create the directory specifically named 'slurm' for the #SBATCH output logs
mkdir -p slurm

# 2. Navigate to your project directory
cd /nas/ucb/jngai/calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not change directory. Exiting."
  exit 1
fi

# 3. Activate your virtual environment
source /nas/ucb/jngai/miniconda3/etc/profile.d/conda.sh
conda activate calyapo
if [ $? -ne 0 ]; then
  echo "Error: Could not activate virtual environment. Exiting."
  exit 1
fi

# --- Define Training Parameters ---
# Joseph's SubPOP version: https://github.com/JosephJeesungSuh/subpop/tree/main
# Llama Cookbook version: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning
if [ -f .env ]; then # store API keys in a local .env file then search for the variables
  export $(grep -v '^#' .env | xargs)
fi
export TOKENIZERS_PARALLELISM=false # for debugging we wanna just use one gpu with batch size 1

# Distributed Setup
NPROC_PER_NODE=1                      # Match this to your --gres=gpu count
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) # Random port to avoid collisions

# Model/Data Params
DATASET="ideology_to_ideology"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="calyapo/training/checkpoints/${DATASET}"
USE_PEFT=True
BATCH_SIZE_TRAINING=1
BATCH_SIZE_VALIDATION=1
GRADIENT_ACCUMULATION_STEPS=4
DIST_CHECKPOINT_ROOT_FOLDER="/nas/ucb/jngai/calyapo/training/model_checkpointing"
DIST_CHECKPOINT_FOLDER="fine-tuned"
NUM_WORKERS_DATALOADER=1
ONE_GPU=True
WEIGHT_DECAY=0.1
GAMMA=0.85
LR=1e-5
NUM_EPOCHS=3
DATASET_PATH="calyapo/training/datasets"
MODEL_NICKNAME="llama7b"

print_header() {
    echo "------------------------------------------------"
    echo "Starting Calyapo Finetuning Job"
    echo "Date:       $(date)"
    echo "Dataset:    ${DATASET}"
    echo "Model:      ${MODEL_NAME}"
    echo "Output:     ${OUTPUT_DIR}"
    echo "------------------------------------------------"
}

print_header

# --- Run Training with torchrun ---
# NO SPACES AFTER THE BACKSLACH
# DISABLE FSDP FOR DEBUGGING
torchrun --nnodes=1 \
    --nproc-per-node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    scripts/experiment/run_finetune.py \
    --enable_fsdp False \
    --low_cpu_fsdp False \
    --fsdp_config.pure_bf16 \
    --use_peft=${USE_PEFT} \
    --use_fast_kernels \
    --checkpoint_type StateDictType.FULL_STATE_DICT \
    --peft_method='lora' \
    --use_fp16 \
    --mixed_precision \
    --batch_size_training ${BATCH_SIZE_TRAINING} \
    --val_batch_size ${BATCH_SIZE_VALIDATION} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --dist_checkpoint_root_folder ${DIST_CHECKPOINT_ROOT_FOLDER} \
    --dist_checkpoint_folder ${DIST_CHECKPOINT_FOLDER} \
    --batching_strategy='padding' \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --dataset_path ${DATASET_PATH} \
    --model_name ${MODEL_NAME} \
    --model_nickname ${MODEL_NICKNAME} \
    --num_workers_dataloader ${NUM_WORKERS_DATALOADER} \
    --lr ${LR} \
    --num_epochs ${NUM_EPOCHS} \
    --weight_decay ${WEIGHT_DECAY} \
    --gamma ${GAMMA} \
    --seed 42 \
    --one_gpu ${ONE_GPU} \
    --use_wandb \
    --save_model \
    --save_metrics \
    --save_optimizer