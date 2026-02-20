# CalYAPo

CalyAPO is Jonathan Ngai's Data Science Honors Thesis repository for analyzing California public opinion data and training neural net models.

# Steps
- load raw data including as jsons
- define a your_dataset.py file in the training/datasets folder than tells the repo how to load your data
- modify training/datasets/__init__.py so that the dictionary handles for the new dataset you added AND update configs/datasets.py
- in dataset_utils define the data collator, loader and getter 
- modify train_utils.py in training/utils with relevant config and different modes
- modify the finetuning.py script

# Idea
- Clean and store steering prompts into jsons (can be done seperately outside of GPU memory)

# Pipeline design
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py are never supposed to interact with the Orchestrator class. The Orchestrator acts upon and manipulates functions in those files. 
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py should not be pulling from paths directly but rather passing data in memory. The Orchestrator class can handle pulling. 
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py need not have rigorous checks. The Individual class handles that. 

# To Do
- Update IGS Raw with utils file loader
- Implement CES Raw
- Handle for pasing configs in the Orchestrator
- Update combiner to allow for in memory passing
- Check implementation of build dataset in clean_datasets.py

# repo structure
\data

--\raw

----\anes

----\ppic

----\igs



\calyapo

--\configurations

----\config.py

----\data_map_config.py

----\data_mappings.py

--\data_preprocessing

----\clean_datasets.py

----\data_combiner.py

----\generate_steering_prompts.py

--\training (llama cookbook clone)

----\configs

----\data

----\datasets

----\inference

----\model_checkpoints

----\policies

----\tools

----\utils

----\finetuning.py

# Data flow
Data in the \data repo but its path is instantiated in the \calyapo\configurations\config repo. 

clean_datasets --> turns it into jsons binding 

# Quick Commands
- Clean Dataset: python calyapo/data_preprocessing/clean_datasets.py
- Split Data: python calyapo/data_preprocessing/data_combiner.py
- Test Prompt Tokenizer: python calyapo/training/datasets/calyapo_dataset.py
- Finetune: python -m calyapo.training.finetuning \
    --dataset "ideology_to_trump_dataset" \
    --run_validation True \
    --save_model True \
    --output_dir "calyapo/training/model_checkpoints"

- Local Windows Finetuning:
    python -m calyapo.training.finetuning `
    --dataset "ideology_to_trump_dataset" `
    --model_name "meta-llama/Llama-2-7b-hf" `
    --output_dir "calyapo/training/checkpoints/ideology_to_trump" `
    --batch_size 1 `
    --micro_batch_size 1 `
    --gradient_accumulation_steps 4 `
    --num_epochs 3 `
    --run_validation True `
    --save_model True `
    --use_peft `
    --peft_method lora `
    --quantization 4bit `
    --use_fp16 True `
    --dataloader_num_workers 0

    - Also need to change finetuning data loaders to have 0 workers