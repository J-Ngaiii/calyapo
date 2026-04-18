import json
import re
import os
from pathlib import Path
import pandas as pd
from typing import Iterable, Union

import argparse

from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER
from calyapo.utils import report_path, file_saver

def get_base_data_path(train_plan: Union[str|Path], filetype: str = 'jsonl', verbose=False):
    path_conf = {
        'train' : UNIVERSAL_FINAL_FOLDER / Path(f"{args.train_plan}_train.{filetype}"), 
        'val' : UNIVERSAL_FINAL_FOLDER / Path(f"{args.train_plan}_val.{filetype}"), 
        'test' : UNIVERSAL_FINAL_FOLDER / Path(f"{args.train_plan}_test.{filetype}")
    }
    return path_conf
    
def get_inf_data_path(train_plan: Union[str|Path], in_path_folder: Union[str|Path] = "inference_outputs", sub_folder: Union[str|Path] = None, verbose = False):
    base_path = Path(in_path_folder) / train_plan
    if sub_folder:
        base_path = base_path / sub_folder

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

def parse_inference_data(results_path, config_path, verbose: bool = False):
    with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
    engine_params = config_data['engine_params']
    sampling_params = config_data['sampling_params']

    if verbose:
        print(f"model:                   {engine_params.get('model', None)}")
        print(f"quantization:            {engine_params.get('quantization', None)}")
        print(f"max_model_len:           {engine_params.get('max_model_len', None)}")
        print(f"max_num_seqs:            {engine_params.get('max_num_seqs', None)}")
        print(f"gpu_memory_utilization:  {engine_params.get('gpu_memory_utilization', None)}")
        print(f"LoRA Enabled:            {engine_params.get('enable_lora', None)}")
        print(f"seed:                    {engine_params.get('seed', None)}")

        print(f"temperature:             {sampling_params.get('temperature', None)}")
        print(f"max_tokens:              {sampling_params.get('max_tokens', None)}")
        print(f"logprobs:                {sampling_params.get('logprobs', None)}")
        print(f"----------------------------------------------")
    
    ids = []
    predictions = []
    true_values = []
    data_labels = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                pred_correct = int(json_obj['is_correct'])
                pred_val = str(json_obj['prediction'])
                tru_val = str(json_obj['true_label'])
                idx = int(json_obj['index'])

                ids.append(idx)
                data_labels.append(pred_correct)
                predictions.append(pred_val)
                true_values.append(tru_val)

    dataset = pd.DataFrame({
        'index' : ids, 
        'model_prediction' : predictions, 
        'true_value' : true_values, 
        'prediction_correct' : data_labels, 
        'model_name' : [engine_params.get('model', None)]*len(ids)
    })

    return dataset

def main(train_plan: str, out_path: Union[str|Path], sub_folder: str = None, save: bool = False, verbose: bool = False, debug: bool = False):
    base_data_paths = get_base_data_path(train_plan=train_plan, verbose=verbose)
    inf_data_paths = get_inf_data_path(train_plan=train_plan, sub_folder=sub_folder, verbose=verbose)

    calyapo_data = {
        'train' : parse_calyapo_data(base_data_paths['train']), 
        'val' : parse_calyapo_data(base_data_paths['val']), 
        'test' : parse_calyapo_data(base_data_paths['test'])
    }

    for k, v in inf_data_paths.items():
        inf_dat = parse_inference_data(**v, verbose=verbose)
        model_name = str(inf_dat['model_name'][0])
        split, model_type = k.split('_')

        base_dat = calyapo_data[split]
        assert len(base_dat) == len(inf_dat), f"Length mismatch between base and inference datasets."
        assert all(base_dat['answer'] == inf_dat['true_value']), f"True values do not match between base and inference datasets."

        base_dat[f'{model_name}_{model_type}_correct'] = inf_dat['prediction_correct']

        calyapo_data[split] = base_dat
        

    if save:
        file_saver(out_path=Path(out_path) / f"{train_plan}_train_eval_data.csv", data=calyapo_data['train'], data_type='csv', verbose=verbose)
        file_saver(out_path=Path(out_path) / f"{train_plan}_val_eval_data.csv", data=calyapo_data['val'], data_type='csv', verbose=verbose)
        file_saver(out_path=Path(out_path) / f"{train_plan}_test_eval_data.csv", data=calyapo_data['test'], data_type='csv', verbose=verbose)


if __name__ == "__main__":
    # --- Implementation ---
    # Replace 'data.jsonl' with your actual filename
    # df = parse_calyapo_data('data.jsonl')

    # Example of how to do a cross-tab once you have the 'df':
    # ct = pd.crosstab(df['Party Identity'], df['answer'])
    # print(ct)
    parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan to finetune on.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    OUTPATH_FOLDER = Path("evaluation_datasets")
    LLAMA_SUBFOLDER = "meta-llama"
    LLAMA_MODELS_FINETUNED = [
        'Llama-3.1-8B', 
        'Llama-3.1-8B-Instruct', 
        'Llama-3.2-3B', 
        'Llama-3.2-3B-Instruct', 
    ]

    for model_name in LLAMA_MODELS_FINETUNED:
        main(train_plan = args.train_plan, out_path = OUTPATH_FOLDER, sub_folder = Path(LLAMA_SUBFOLDER) / model_name, save = args.save, verbose = args.verbose, debug = args.debug)

