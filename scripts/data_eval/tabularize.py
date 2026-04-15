import json
import re
import os
from pathlib import Path
import pandas as pd
from typing import Iterable, Union

import argparse

from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER
from calyapo.utils import report_path, file_saver

def get_data_path(train_plan: Union[str|Path], verbose=False):
    path_conf = {
        'train' : UNIVERSAL_FINAL_FOLDER / Path(f"{args.train_plan}_train"), 
        'val' : UNIVERSAL_FINAL_FOLDER / Path(f"{args.train_plan}_val"), 
        'test' : UNIVERSAL_FINAL_FOLDER / Path(f"{args.train_plan}_test")
    }
    return path_conf
    
def get_inf_path(train_plan: Union[str|Path], time_folder=None, verbose=False):
    base_path = Path(OUTPUT_FOLDER) / train_plan
    if time_folder:
        base_path = base_path / time_folder

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")
    pattern = re.compile(r"(results|config)_(training|train|validation|test)_.*?(lora|base)_(\d{8}_\d{6})\.(jsonl|json)")

    found_files = {}

    # collect files in a dictionary
    for file_path in base_path.iterdir():
        match = pattern.match(file_path.name)
        if match:
            file_type, split, model_type, timestamp, ext = match.groups()
            if 'train' in split:
                split_key = 'train' 
            elif 'val' in split:
                split_key = 'val' 
            elif 'test' in split:
                split_key = 'test'
            else:
                if verbose: 
                    print(f"Split keyword '{split}' not recognized, skipping file.")
                continue
            
            if split_key not in found_files: 
                found_files[split_key] = {}
            if model_type not in found_files[split_key]: 
                found_files[split_key][model_type] = {}
            
            if file_type == 'results':
                key_name = 'results_path'
            elif file_type == 'config':
                key_name = 'config_path'
            else:
                if verbose: 
                    print(f"File type '{file_type}' not recognized, skipping file.")
                continue
            found_files[split_key][model_type][key_name] = file_path

    # index into path collection dictionary to format final output
    out = {}
    for split in ['train', 'val', 'test']:
        for model_type in ['lora', 'base']:
            key: str = f"{split}_{model_type}"
            out[key] = found_files.get(split, {}).get(model_type, {})

    for key, paths in out.items():
        if not paths or 'results_path' not in paths or 'config_path' not in paths:
            if verbose: print(f"Warning: Missing files for {key} in {base_path}")
            
    return out

def parse_calyapo_data(file_path):
    rows = []
    meta_regex = re.compile(r"IGS dataset in (?P<date>\d+)\.\nDemographics: (?P<demogs>.*?)\nQuestion \((?P<topic>.*?)\): (?P<question>.*?)\n", re.DOTALL)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            prompt = data.get("prompt", "")
            completion = data.get("completion", "").strip()
            
            match = meta_regex.search(prompt)
            if match:
                # 1. Start with base fields
                entry = {
                    "dataset_date": match.group("date"),
                    "topic": match.group("topic"),
                    "answer": completion
                }
                
                # 2. Parse Demographics string (e.g., "Age: 50-64, Party: Democrat")
                demog_str = match.group("demogs")
                # Split by comma, but only if followed by a "Key: " pattern
                parts = re.split(r",\s*(?=[A-Za-z\s]+:)", demog_str)
                
                for p in parts:
                    if ":" in p:
                        key, val = p.split(":", 1)
                        entry[key.strip()] = val.strip()
                
                rows.append(entry)
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
# --- Implementation ---
# Replace 'data.jsonl' with your actual filename
# df = parse_calyapo_data('data.jsonl')

# Example of how to do a cross-tab once you have the 'df':
# ct = pd.crosstab(df['Party Identity'], df['answer'])
# print(ct)
parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan to finetune on.")
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

