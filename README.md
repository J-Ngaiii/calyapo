# CalyAPO

CalyAPO is Jonathan Ngai's Data Science Honors Thesis repository for analyzing California public opinion data and training neural net models.

# Steps
- load raw data including as jsons
- define a your_dataset.py file in the training/datasets folder than tells the repo how to load your data
- modify training/datasets/__init__.py so that the dictionary handles for the new dataset you added AND update configs/datasets.py
- in dataset_utils define the data collator, loader and getter 
- modify train_utils.py in training/utils with relevant config and different modes
- modify the finetuning.py script