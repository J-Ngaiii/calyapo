import json
from pathlib import Path
import argparse
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable

from calyapo.utils import report_path

OUTPUT_FOLDER = Path("inference_outputs")

def get_path(train_plan, time_folder=None, verbose=False):
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

def calculate_accuracy(results_path, config_path, bootstrap=False, verbose=False):
    if verbose:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        engine_params = config_data['engine_params']
        sampling_params = config_data['sampling_params']
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
    
    data_labels = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                pred_correct = int(json_obj['is_correct'])
                data_labels.append(pred_correct)
    
    labels_array = np.array(data_labels)
    num_datapoints = len(data_labels)
    acc = np.mean(labels_array)

    lower_ci, upper_ci = None, None
    
    bootstrapped_means = []
    if bootstrap:
        rng = np.random.default_rng(seed=42)
        resamples = rng.choice(labels_array, size=(10000, num_datapoints), replace=True)
        bootstrapped_means = np.mean(resamples, axis=1).tolist()
        lower_ci = np.percentile(bootstrapped_means, 2.5)
        upper_ci = np.percentile(bootstrapped_means, 97.5)

    if verbose: 
        print(f"\n----------------------- Metrics -----------------------")
        print(f"Total number of datapoints loaded in: {num_datapoints}")
        print(f"Accuracy: {acc}")
        if bootstrap:
            print(f"95% Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
        print(f"---------------------------------------------------------")
    stat_dict = { # encodes relevant singular-valued statistics
        'Model Accuracy' : acc, 
        'Lower CI' : lower_ci, 
        'Higher CI' : upper_ci
    }
    list_dict = { # encodes relevant multi-valued lists and matrices
        'Bootstrapped Means' : bootstrapped_means
    }
    return tuple([stat_dict, list_dict])

def plot_results(results_dict, train_plan, time_folder = None, verbose = False):
    """Generates a bar chart with error bars representing the 95% CI."""    
    
    summary_rows = []
    bootstrap_dfs= []
    for name, dicts in results_dict.items():
        if dicts:
            stat_dict, list_dict = dicts
            split, model_type = name.split('_')
            split_label, model_label = split.capitalize(), model_type.upper()
            summary_rows.append({
                'Split': split_label,
                'Model': model_label,
                **stat_dict
            })

            bootstrapped_means = list_dict['Bootstrapped Means']
            if len(bootstrapped_means) > 0:
                temp_df = pd.DataFrame({
                'Split': [split_label] * len(bootstrapped_means),
                'Model': [model_label] * len(bootstrapped_means),
                'Sample Accuracies': bootstrapped_means
            })
            bootstrap_dfs.append(temp_df)

    summary_df = pd.DataFrame(summary_rows)
    bootstrapped_df = None
    if len(bootstrap_dfs) > 0:
        bootstrapped_df = pd.concat(bootstrap_dfs, ignore_index=True)
    else:
        print(f"No bootstrapped data found, defaulting to summary df")
        bootstrapped_df = summary_df

    if summary_df.empty:
        print("No data to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    sns.set_style("whitegrid")
    palette = {"LORA": "orange", "BASE": "dodgerblue"}

    # --- AX1: BAR PLOT ---
    bar = sns.barplot(data=summary_df, x="Split", y="Model Accuracy", hue="Model", 
                palette=palette, ax=ax1, alpha=0.8)
    for container in bar.containers:
        ax1.bar_label(container, fmt='%.3f', padding=3)
    ax1.set_title("Model Accuracy Scores")

    # --- AX2: POINT PLOT ---
    # linestyle=None removes the connecting lines
    # dodge=True prevents the points from overlapping vertically
    ax = sns.pointplot(
        data=bootstrapped_df,
        x="Split",
        y="Sample Accuracies",
        hue="Model",
        linestyle='none',
        dodge=0.25, # Separates the points horizontally
        palette={"LORA": "orange", "BASE": "dodgerblue"},
        markers=["D", "o"], # Diamond for LoRA, Circle for Base
        markersize=6, # controls how big the markers are for the different models
        capsize=0.125, # controls how wide/large the CI caps are
        err_kws={'linewidth': 1}, 
        errorbar=("pi", 95), 
        ax=ax2
    )
    ax2.set_title("95% Bootstrap Confidence Intervals for Model Performance")
    plt.ylim(0, 1.0)
    
    # can outsource saving to persistence.py
    report_config = {
        'train_plan' : train_plan, 
        'report_name' : 'plot_package', 
        'save_file_type' : 'png'
    }
    save_path = report_path(report_conf=report_config)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Seaborn plot saved to: {save_path}")
    plt.close()

def main(train_plan, time_folder = None, bootstrap = False, graph = False, verbose = False):
    splits_conf = get_path(train_plan=train_plan, time_folder=time_folder, verbose=verbose)

    plot_data = {}
    for k, v in splits_conf.items():
        print(f"-------------- Processing Split '{k}' --------------\n")
        plot_data[k] = calculate_accuracy(**v, bootstrap=bootstrap, verbose=verbose)

    if graph:
        plot_results(results_dict=plot_data, train_plan=train_plan, time_folder=time_folder, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan to finetune on.")
    parser.add_argument("--time_folder", type=str, nargs='?', default=None, help="Folder corresponding to time period of training run.")
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()
    
    main(args.train_plan, args.time_folder, bootstrap=args.bootstrap, graph=args.graph, verbose=args.verbose)