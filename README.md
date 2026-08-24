# CalYAPo

CalYAPo is a research repository for finetuning large language models (LLMs) on California sub-national public opinion data, developed as part of a Data Science Honors Thesis at UC Berkeley. The project investigates whether LLMs finetuned with Low-Rank Adaptation (LoRA) on state-level survey data can accurately predict individual-level survey responses and if such individual-level predictive performance produces trades-offs against aggregate distributional alignment.

The dataset (~19,000 individuals) is constructed from California-specific survey data collected by the [Berkeley Institute of Governmental Studies (IGS)](https://igs.berkeley.edu/).

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Data Access](#data-access)
- [Data Pipeline](#data-pipeline)
- [Finetuning](#finetuning)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Adding New Datasets](#adding-new-datasets)
- [Codebase Walkthrough](#codebase-walkthrough)

---

## Hardware Requirements

Finetuning was conducted on the [UC Berkeley Savio HPC cluster](https://research-it.berkeley.edu/services-projects/high-performance-computing-savio) using a single NVIDIA A40 GPU (`savio3_gpu` partition). Inference runs were conducted on a single NVIDIA A5000 GPU (`savio4_gpu` partition).

The following models were evaluated:

| Model | Parameters | Type |
|-------|-----------|------|
| `meta-llama/Llama-3.1-8B` | 8B | Base |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Instruction-tuned |
| `meta-llama/Llama-3.2-3B` | 3B | Base |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Instruction-tuned |
| `Qwen/Qwen2.5-7B` | 7B | Base |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Instruction-tuned |
| `Qwen/Qwen2.5-14B` | 14B | Base |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Instruction-tuned |

All finetuning runs use 4-bit quantization and mixed precision (bf16) to fit within a single GPU's memory budget. The wall-clock time limit per finetuning job is 60 hours; inference jobs are capped at 10 hours.

---

## Installation

```bash
conda create -n calyapo python=3.10 -y
conda activate calyapo
pip install -e .
```

---

## Data Access

The CalYAPo dataset is derived from California public opinion surveys administered by the Berkeley Institute of Governmental Studies (IGS). Raw IGS poll data is publicly available for download from the [IGS Poll website](https://igs.berkeley.edu/research/berkeley-igs-poll). Once downloaded, place the raw files under `data/raw/igs/`.

Expected raw data sources and their locations:

| Source | Format | Path |
|--------|--------|------|
| IGS (California) | `.sav` / `.dta` | `data/raw/igs/` |

---

## Data Pipeline

Data moves through five stages from raw survey files to model-ready prompts:

| Stage | Format | Description |
|-------|--------|-------------|
| **Raw** | `.csv`, `.dta`, `.sav` | Unprocessed survey files |
| **Intermediate** | `.csv` | Cleaned by `raw_cleaners`; all columns are mapped to readable variable labels (e.g. `harris_opinion`) |
| **Processed** | `.json` | One JSON per dataset–time-period combination; each entry is an individual with train/val/test questions, responses, and demographics |
| **Penultimate** | `.json` | All processed JSONs compiled across time periods per dataset |
| **Final** | `.json` | Prompt-completion formatted JSONs ready for finetuning |

**Architecture note:** Individual cleaning functions operate on a per-dataframe basis. Handler functions manage file I/O and in-memory data passing between stages.

### Quick commands

```bash
# Stage 1–2: Clean raw survey data to intermediate
python calyapo/data_preprocessing/clean_datasets.py

# Stage 3–5: Combine and format into train/val/test splits
python calyapo/data_preprocessing/data_combiner.py

# Test prompt tokenization
python calyapo/training/datasets/calyapo_dataset.py
```

Data paths are defined in `calyapo/configurations/config.py`.

---

## Finetuning

Finetuning uses the [LLaMA Cookbook](https://github.com/meta-llama/llama-cookbook) framework with LoRA adapters. Jobs are submitted to Savio via SLURM:

```bash
sbatch scripts/experiment/finetune.slurm
```

The sbatch script sets training parameters as environment variables and launches training via `torchrun`:

```bash
torchrun --nnodes=1 \
    --nproc-per-node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    scripts/experiment/run_finetune.py \
    --enable_fsdp False \
    --use_peft True \
    --quantization "4bit" \
    --use_fast_kernels \
    --peft_method='lora' \
    --use_fp16 \
    --mixed_precision \
    --batch_size_training 4 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --model_nickname ${MODEL_NICKNAME} \
    --output_dir ${OUTPUT_DIR} \
    --lr 1e-5 \
    --num_epochs 3 \
    --weight_decay 0.1 \
    --gamma 0.85 \
    --seed 42 \
    --save_model True
```

Key parameters to configure in the sbatch script before submission:

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model identifier | `meta-llama/Llama-3.1-8B` |
| `MODEL_NICKNAME` | Short name used for checkpoint naming | `llama3.1-8b` |
| `DATASET` | Dataset config name | `opinion_school_dataset` |
| `OUTPUT_DIR` | Checkpoint output path | `calyapo/training/checkpoints/${DATASET}` |

Model and training hyperparameters are configured in `training/configs/training.py` and can be overridden via sbatch arguments. See [Codebase Walkthrough](#codebase-walkthrough) for a detailed trace of the finetuning execution.

---

## Inference

Inference jobs are submitted to Savio via SLURM:

```bash
sbatch scripts/experiment/inference.slurm
```

Or run directly:

```bash
python scripts/llm/offline_inf.py \
    --train_plan=${TRAIN_PLAN} \
    --model_name=${MODEL_NAME} \
    --model_type=${MODEL_TYPE} \
    --adapter_folder=${ADAPTER_FOLDER} \
    --split=${SPLIT} \
    --run_keyword=${RUN_KEYWORD} \
    --num_gpus=1 \
    --chunk_size=2000
```

Key parameters:

| Variable | Description | Example |
|----------|-------------|---------|
| `TRAIN_PLAN` | Name of the training plan to run inference on | `opinion_school` |
| `MODEL_NAME` | HuggingFace model identifier | `meta-llama/Llama-3.1-8B` |
| `MODEL_TYPE` | Whether to load the base or LoRA-finetuned model | `lora` or `base` |
| `ADAPTER_FOLDER` | Name of the checkpoint folder under `training/checkpoints/` | `wdllama3.1-8b_wd0.1_gam0.85_lr1e-05_...` |
| `SPLIT` | Dataset split to run inference on | `train`, `val`, or `test` |
| `RUN_KEYWORD` | Tag for naming the output inference run | `archon` |

---

## Evaluation

```bash
# Generate summary tables
python scripts/data_eval/table.py

# Generate full evaluation report
python scripts/data_eval/report.py
```

---

## Repository Structure

```
calyapo/
├── configurations/
│   ├── config.py                  # Data path configuration
│   ├── data_map_config.py
│   └── data_mappings.py           # Variable label mappings
├── data_preprocessing/
│   ├── clean_datasets.py          # Raw → Intermediate
│   ├── data_combiner.py           # Intermediate → Final splits
│   └── generate_steering_prompts.py
└── training/                      # LLaMA Cookbook clone
    ├── configs/                   # Training and dataset configs
    ├── datasets/                  # Dataset loader definitions
    │   └── calyapo_dataset.py
    ├── inference/
    ├── model_checkpoints/
    ├── utils/
    │   ├── dataset_utils.py
    │   └── train_utils.py
    └── finetuning.py              # Main finetuning entrypoint

data/
├── raw/
│   ├── anes/
│   ├── ppic/
│   └── igs/
```

---

## Training Plans

A Training Plan is a configuration object that specifies which demographic variables to include, which survey questions to train on, and how to split the data. Two Training Plans were used in this work:

**`Opinion_School`** — used for Training Setting 1 (generalization to unseen individuals). Trains and validates on the same three favorability questions (Kamala Harris, Joe Biden, Donald Trump). Composed of 26,516 training, 7,576 validation, and 3,790 test datapoints.

**`Presidents_to_Abortion`** — used for Training Setting 2 (generalization to unseen questions). Trains on Biden/Trump favorability responses and validates on abortion access opinion questions. Composed of 29,606 training, 3,792 validation, and 1,896 test datapoints.

Training Plans are defined in `training/configs/datasets.py`. Each individual-question pair is encoded as a separate datapoint, so a single respondent who answers multiple survey questions contributes multiple entries to the dataset.

---

## Adding New Datasets

1. Register the dataset config in `training/configs/datasets.py`
2. Add the dataset loader mapping in `training/datasets/__init__.py`
3. Implement a `get_<dataset>_dataset` function following the pattern in `training/datasets/calyapo_dataset.py`
4. Place raw data files under the appropriate `data/raw/<source>/` directory and update path references in `calyapo/configurations/config.py`

---

## Codebase Walkthrough

### Finetuning logic (`training/finetuning.py`)

The finetuning script is invoked by `scripts/experiment/run_finetune.py` via a `fire` call from the sbatch script.

Key execution steps:

- **Lines 152**: A `config` object is instantiated from `train_config.model_name`; `train_config` is defined in `training/configs/training.py` and is partially overridden by sbatch arguments.
- **Lines 153–197**: The `model` object is constructed based on `config.model_type`.
- **Lines 328–342**: `get_preprocessed_dataset()` is called and returns the train/validation datasets.
  - Defined in `training/utils/dataset_utils.py`
  - Accepts a `dataset_config` (e.g. from `training/configs/datasets.py`)
  - Uses `DATASET_PREPROC` in `training/datasets/__init__.py` to look up the appropriate loader by `dataset_config.dataset`
  - Calls `get_calyapo_dataset()` (defined in `training/datasets/calyapo_dataset.py`) with the dataset config, tokenizer, and split path from `get_split()`
- **Line 416**: `train()` is called, defined in `training/utils/train_utils.py`
  - **Line 241 of `train_utils.py`**: `evaluation()` is called during the training loop

---

## Citation

If you use CalYAPo in your research, please cite:

```bibtex
@thesis{ngai2026calyapo,
  author = {Ngai, Jonathan},
  title  = {CalYAPo: Simulating California Political Opinion with LoRA Finetuned Large Language Models},
  school = {University of California, Berkeley},
  year   = {2026}
}
```
