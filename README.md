# CalYAPo

CalyAPO is Jonathan Ngai's Data Science Honors Thesis repository for analyzing California public opinion data and training neural net models.

# Idea
- If we split each dataset into train/val/test based on predefined ratios then comebine them back together the whole thing => the overall ratios will be preserved for the combined dataset and each split will have a proportionate number of individuals from each dataset

# Pipeline design
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py are never supposed to interact with the Orchestrator class. The Orchestrator acts upon and manipulates functions in those files. 
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py should not be pulling from paths directly but rather passing data in memory. The Orchestrator class can handle pulling. 
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py need not have rigorous checks. The Individual class handles that. 
- The cleaning functions in raw_cleaners.py clean_datasets.py and data_combiner.py get imported into DataSplitter that creates the train, val and tests splits end to end.
- skipping missing data happens in split_combine's llama flatten helper

**overall hierarchy**
- Individual cleaning functions execute operations on a per df batch basis
- Handler funcs handle file pulling and in-memory data passing
- Orchestrator string inputs/outputs from diff handlers together

# Stages of data
- Raw (csv or dta or sav): Completely unprocessed
- Intermediate (csv): Processed by raw_cleaners, all columns are readable by configs to be mapped to variable_labels like 'harris_opinion'
- Processed (json): json form of each dataset-time period combination, basically a list of dictionaries with each dictionary corresponding to an individual with the train/val/test questions and responses and demographics per indiv_map
- penultimate (json): compile all jsons from /processed directory based on dataset across time period
- final (json): contains the actual steering : completion formatted jsons

# Adding new datasets
- add to training/configs/datasets.py
- add to training/datasets/__init__.py

# Finetuning execution
- sbatch script executes caLL
- scripts/experiment/run_finetune.py --> initiates fire call to execute the actual finetuning.py script
- lines 328-342 --> get_preprocessed_dataset is called --> returns datasets
    - get_preprocessed_dataset is defined in training/utils/dataset_utils what it does is:
        - take in datset_config like that defined in training/configs/datasets.py
        - access DATASET_PREPROC in training/datasets/__init__.py using dataset_config.dataset
        - get the 'get_calyapo_dataset' method that's mapped in training/datasets/__init__.py but defined in training/datasets/calyapo_dataset.py
        - initialize and execute the 'get_calyapo_dataset' method with 
            - dataset_config
            - tokenizer
            - output of internally defined get_split() function which outputs dataset_config.train or dataset_config.test path directly based on the split get_preprocessed_dataset got initially
        - get_calyapo_dataset then executes and returns 
        - that recursively goes back yp to be the output of get_preprocessed_dataset

# To Do
- uniqieID for training on multiple questions so the samplers dont need to de-duplicate

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