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
#SBATCH --gres=gpu:A40:2
#SBATCH --qos=a40_gpu3_normal

# Wall clock limit:
#SBATCH --time=60:00:00

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

# --- Define Training Parameters ---
# Joseph's SubPOP version: https://github.com/JosephJeesungSuh/subpop/tree/main
# Llama Cookbook version: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning
if [ -f .env ]; then # store API keys in a local .env file then search for the variables
  export $(grep -v '^#' .env | xargs)
fi
export TOKENIZERS_PARALLELISM=true

# Set longer time for GPUs to wait for each other because it takes time to load weights from rank 0
# export NCCL_BLOCKING_WAIT=1
# export NCCL_TIMEOUT=180000

# more flags
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Or the specific interface Savio uses
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed Setup
NPROC_PER_NODE=2                      # Match this to your --gres=gpu count
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) # Random port to avoid collisions

# Model/Data Params
DATASET="opinion_school_dataset"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="calyapo/training/checkpoints/${DATASET}"
USE_PEFT=True
BATCH_SIZE_TRAINING=16
BATCH_SIZE_VALIDATION=32
GRADIENT_ACCUMULATION_STEPS=4
DIST_CHECKPOINT_ROOT_FOLDER="/global/home/users/jonathanngai/calyapo/calyapo/training/checkpoints/${DATASET}"
DIST_CHECKPOINT_FOLDER="fine-tuned"
NUM_WORKERS_DATALOADER=4
ONE_GPU=False
WEIGHT_DECAY=0.1
GAMMA=0.85
LR=1e-5
NUM_EPOCHS=3
MODEL_NICKNAME="llama2-7b"
ENABLE_FSDP=True
LOW_CPU_FSDP=True
LOW_CPU_MEM_USAGE=True
PURE_BF16=True

print_header() {
    echo "------------------------------------------------"
    echo "Starting Calyapo Finetuning Job"
    echo "Date:               $(date)"
    echo "Dataset:            ${DATASET}"
    echo "Model:              ${MODEL_NAME}"
    echo "Output:             ${OUTPUT_DIR}"
    echo ""
    echo "Distributed Computing Checks"
    echo "FSDP:               ${ENABLE_FSDP}"
    echo "Low CPU FSDP:       ${LOW_CPU_FSDP}"
    echo "Low CPU Memory:     ${LOW_CPU_MEM_USAGE}"
    echo "One GPU:            ${ONE_GPU}"
    echo ""
    echo "Training Batch Checks"
    echo "Batch Size Train:   ${BATCH_SIZE_TRAINING}"
    echo "Batch Size Val:     ${BATCH_SIZE_VALIDATION}"
    echo "Grad Acc:           ${GRADIENT_ACCUMULATION_STEPS}"
    echo ""
    echo "Training Hyperparameter Checks"
    echo "Weight Decay:       ${WEIGHT_DECAY}"
    echo "Gamma:              ${GAMMA}"
    echo "Learning Rate:      ${LR}"
    echo "Epochs:             ${NUM_EPOCHS}"
    echo "------------------------------------------------"
}

print_header

# --- Run Training with torchrun ---
# NO SPACES AFTER THE BACKSLACH
# DISABLE FSDP FOR DEBUGGING
# 1080 cannot handle --fsdp_config.pure_bf16

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 \
    --nproc-per-node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    scripts/experiment/run_finetune.py \
    --enable_fsdp ${ENABLE_FSDP} \
    --low_cpu_fsdp ${LOW_CPU_FSDP} \
    --fsdp_config.pure_bf16 ${PURE_BF16} \
    --use_peft=${USE_PEFT} \
    --quantization "4bit" \
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
    --model_name ${MODEL_NAME} \
    --model_nickname ${MODEL_NICKNAME} \
    --num_workers_dataloader ${NUM_WORKERS_DATALOADER} \
    --lr ${LR} \
    --num_epochs ${NUM_EPOCHS} \
    --weight_decay ${WEIGHT_DECAY} \
    --gamma ${GAMMA} \
    --seed 42 \
    --one_gpu ${ONE_GPU} \
    --use_wandb True \
    --save_model True \
    --save_metrics True \
    --save_optimizer True \
    --low_cpu_mem_usage True