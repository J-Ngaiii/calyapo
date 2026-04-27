# CalYAPo

CalyAPO is Jonathan Ngai's Data Science Honors Thesis repository for analyzing California public opinion data and training neural net models.

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

# Stages of eval
- make sure intermediate and final copies of calyapo data are available
- run table.py --> creates tabulars from which visualization functions draw
- run report.py --> generate crosstabs, accuracy numbers, kl divergence, wd distance and more

# Adding new models for finetuning
- finetuning.py:
    - import decode layer
    - add in handling to load model (line 257)
    - add in handling for FSDP wrapper (line 341)

# Adding new datasets
- add to training/configs/datasets.py
- add to training/datasets/__init__.py

# Table and Visualization Generation Code
- Config file sets up different tables
- Reader func (model name, base_or_lora, split, train_plan) loads results in as a dataframe then passes it to diff calculation functions
- Calculation functions

# Finetuning execution
- sbatch script executes caLL
- scripts/experiment/run_finetune.py --> initiates fire call to execute the actual finetuning.py script
- lines 152 --> create a config object from train_config.model_name, train_config itself comes from training/configs/training.py and should get overridden partially by arguments in the sbatch
- lines 153-197 we define the 'model' object based on based on config.model_type
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
- lines 416 the train() call actually happens, train() is defined in training/utils/train_utils.py
    - line 241 of train_utils.py: evaluation() gets called

# Adding Metrics (all under train_utils.py)
- for train metrics:
    - line 105: add tracking lists under the 'if train_config.save_metrics' condition
    - line 163: add metrics under train_config.save_metrics
    - line 210: add metrics under wandb.log
    - line 319: save metrics to json
- for val metrics:
    - line 105: add tracking lists under the 'if train_config.save_metrics' condition
    - line 241: handle for evaluation() outputting more statistics then save them accordingly
    - line 363: add validation tracking lists
    - line 389: update validation tracking lists
    - under line 419: remember to also calculate average of the metric across all epochs
    - line 422: wandb update validation metrics
    - line 428: return validation tracking lists
    - line 319: save metrics to json

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
