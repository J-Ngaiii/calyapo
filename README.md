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

# Changes to Llama Cookbook to implement new dataset
[Readme]{https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/README.md}
- training/configs/datasets.py: add in a new dataset class
- training/datasets/: add a igs_dataset.py

# Other files
- Concatenator: no changes needed
- Sampler: no changes needed

# Quick Commands
- Clean Dataset: python calyapo/data_preprocessing/clean_datasets.py
- Split Data: python calyapo/data_preprocessing/data_combiner.py