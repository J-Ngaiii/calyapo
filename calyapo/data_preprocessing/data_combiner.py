import json
import os
from pathlib import Path
from typing import List, Dict, Any

from calyapo.configurations.data_map_config import TRAIN_PLANS, VARLABEL_DESC
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER, UNIVERSAL_NA_FILLER, DATA_PATHS
from calyapo.data_preprocessing.cleaning_objects import DataPackage, Individual
from calyapo.data_preprocessing.clean_datasets import build_steering_dataset
from calyapo.utils.persistence import *

UNIVERSAL_FINAL_FOLDER = Path(UNIVERSAL_FINAL_FOLDER)
UNIVERSAL_FINAL_FOLDER.mkdir(parents=True, exist_ok=True)

def format_demographics(demog_dict: Dict[str, str]) -> str:
    """Converts {'age': '18-29'} -> "Age: 18-29"."""
    parts = []
    for k, v in demog_dict.items():
        if v == UNIVERSAL_NA_FILLER: continue 
        clean_key = VARLABEL_DESC[k]
        parts.append(f"{clean_key}: {v}")
    return ", ".join(parts)

def flatten_data_to_llama_format(raw_data_list: List[Dict], split: str) -> List[Dict[str, str]]:
    """
    Flattens Individuals into MCQ Prompt/Completion pairs.
    """
    flattened_examples = []
    
    for entry in raw_data_list:
        # 1. Base Context
        time_period = entry.get('time', 'Unknown')
        dataset_name = entry.get('dataset', 'Unknown')
        demog_str = format_demographics(entry.get('demog', {}))
        
        narrative = f"This is a respondent from the {dataset_name} dataset in {time_period}."
        
        # 2. Get Section Data
        section_data = entry.get(split, {})
        
        # These maps are parallel
        text_map = section_data.get('var_label2qst_text', {})
        choices_map = section_data.get('var_label2qst_choices', {})
        # This one contains the answer logic {var: {option_text, option_letter}}
        options_map = section_data.get('var_label2qst_option', {})
        
        # 3. Create Examples
        for var_label, answer_data in options_map.items():
            
            # answer_data is a dictionary, e.g. {'option_letter': 'A', 'option_text': 'Yes'}
            # NOTE: eval() might be needed if it was saved as string representation of dict
            if isinstance(answer_data, str):
                try:
                    import ast
                    answer_data = ast.literal_eval(answer_data)
                except:
                    continue

            response_text = answer_data.get('option_text') # text description of chosen option, accurate to survey
            target_letter = answer_data.get('option_letter') # just the corresponding letter
            
            # skip if missing
            if not target_letter or target_letter == UNIVERSAL_NA_FILLER:
                continue
                
            question_text = text_map.get(var_label, "")
            choices_block = choices_map.get(var_label, "")
            question_varlabel_desc = VARLABEL_DESC[var_label]

            # construct P=prompt
            prompt = (
                f"{narrative}\n"
                f"Demographics: {demog_str}.\n"
                f"Question ({question_varlabel_desc}): {question_text}\n"
                f"{choices_block}\n"
                f"Answer:"
            )
            
            # Construct Completion
            completion = f" {target_letter}"
            
            flattened_examples.append({
                "prompt": prompt,
                "completion": completion
            })
            
    return flattened_examples

def save_jsonl(data: List[Dict], filename: str, out_path: str = None):
    if out_path is None:
        out_path = UNIVERSAL_FINAL_FOLDER / filename
    print(f"Saving {len(data)} examples to {out_path}...")
    with open(out_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def split_combine(
                train_plan: str, 
                package: DataPackage = None, 
                in_path: str = None, 
                out_path: str = None, 
                save: bool = True, 
                debug: bool = False, 
                verbose: bool = True):
    if verbose: print(f"--- Running Split & Combine for {train_plan} ---")
    if train_plan not in TRAIN_PLANS:
        raise ValueError(f"Plan {train_plan} not found.")
    
    train_data = []
    val_data = []
    test_data = []
    
    datasets_in_plan = TRAIN_PLANS[train_plan].get('datasets')
    
    for dataset_name in datasets_in_plan:
        if verbose: print(f"Processing {dataset_name}...")
        
        if package is None:
            # first try inputted path, then pull from config
            if in_path:
                processed_dir = in_path
            else:
                processed_dir = DATA_PATHS[dataset_name]['processed']    
            target_json = processed_dir / f"{train_plan}_{dataset_name}_fullpack_processed.json"
            
            if target_json.exists():
                if verbose: print(f"Loading existing package: {target_json.name}")
                raw_json = file_loader(in_path=target_json, data_type='json', verbose=verbose)
                package = DataPackage.from_dict(raw_json) 
            else:
                # if we cannot pull from path generate from scratch
                if verbose: print(f"No processed data found. Building steering dataset for {dataset_name}...")
                package = build_steering_dataset(dataset_name=dataset_name, train_plan=train_plan, save=False, debug=debug)

        if debug:
            print(f"Data Package object: {package}")

        train_individuals: List[Dict] = package.get('train')
        val_individuals: List[Dict] = package.get('val')
        test_individuals: List[Dict] = package.get('test')
        train_data.extend(flatten_data_to_llama_format(train_individuals, 'train'))
        val_data.extend(flatten_data_to_llama_format(val_individuals, 'val'))
        test_data.extend(flatten_data_to_llama_format(test_individuals, 'test'))

    if save:
        save_jsonl(train_data, f"{train_plan}_train.jsonl", out_path)
        save_jsonl(val_data, f"{train_plan}_val.jsonl", out_path)
        save_jsonl(test_data, f"{train_plan}_test.jsonl", out_path)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

def split_ratio():
    pass