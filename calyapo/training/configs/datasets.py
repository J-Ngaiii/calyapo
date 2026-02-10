# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER

@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_cookbook/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_cookbook/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_cookbook/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "getting-started/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""

@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class calyapo_dataset_config:
    """
    Parent class wrapper, done for sanity-checking the CalyapoDataset object in calyapo_dataset.py
    """
    dataset: str = "calyapo_dataset"
    file: str = "calyapo/training/datasets/calyapo_dataset.py:get_calyapo_dataset"
    predict_eos: bool = True

@dataclass
class ideology_to_trump_dataset (calyapo_dataset_config):
    dataset: str = "ideo2trump" # override parent class
    
    train_split: str = f"{str(UNIVERSAL_FINAL_FOLDER)}/ideology_to_trump_train.jsonl"
    # map 'test_split' (what Llama expects) to your Validation file (in-sample check)
    test_split: str = f"{str(UNIVERSAL_FINAL_FOLDER)}/ideology_to_trump_val.jsonl"
    # ideology_to_trump_test.jsonl' is NOT included in this config.
    # it stays hidden until you run the final inference script.


@dataclass
class ideology_to_ideology_dataset (calyapo_dataset_config):
    dataset: str = "ideo2ideo" # override parent class
    
    train_split: str = f"{str(UNIVERSAL_FINAL_FOLDER)}/ideology_to_ideology_train.jsonl"
    test_split: str = f"{str(UNIVERSAL_FINAL_FOLDER)}/ideology_to_ideology_val.jsonl"

