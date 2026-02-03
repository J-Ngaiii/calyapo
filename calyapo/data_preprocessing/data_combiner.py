import json
import os
from pathlib import Path
from typing import List, Dict, Any

from calyapo.configurations.data_map_config import ALL_DATA_MAPS, TRAIN_PLANS
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER
from calyapo.data_preprocessing.cleaning_objects import DataPackage
from calyapo.data_preprocessing.clean_datasets import build_steering_dataset, NA_FILLER

UNIVERSAL_FINAL_FOLDER = Path(UNIVERSAL_FINAL_FOLDER)
UNIVERSAL_FINAL_FOLDER.mkdir(parents=True, exist_ok=True)

def format_demographics(demog_dict: Dict[str, str]) -> str:
    """
    Converts {'age': '18-29', 'partyid': 'Democrat'} 
    into "Age: 18-29, Partyid: Democrat"
    """
    parts = []
    for k, v in demog_dict.items():
        if v == NA_FILLER: continue 
        # Clean keys: 'party_id' -> 'Party Id'
        clean_key = k.replace('_', ' ').title()
        parts.append(f"{clean_key}: {v}")
    return ", ".join(parts)

def flatten_data_to_llama_format(raw_data_list: List[Dict], section_key: str) -> List[Dict[str, str]]:
    """
    Takes a list of Individual dictionaries (e.g. all train individuals)
    and flattens them into Llama-compatible Prompt/Completion pairs.
    """
    flattened_examples = []
    
    for entry in raw_data_list:
        # 1. Base Context (Who is this?)
        time_period = entry.get('time', 'Unknown')
        dataset_name = entry.get('dataset', 'Unknown')
        demog_str = format_demographics(entry.get('demog', {}))
        
        narrative = f"This is a respondent from the {dataset_name} dataset in {time_period}."
        
        # 2. Get the specific section (train, val, or test)
        # Structure: entry['train']['var_label2qst_option']...
        section_data = entry.get(section_key, {})
        options_map = section_data.get('var_label2qst_option', {})
        text_map = section_data.get('var_label2qst_text', {})
        
        # 3. Create one example per question
        for var_label, response_text in options_map.items():
            # Skip valid missing data
            if not response_text or response_text == NA_FILLER:
                continue
                
            question_text = text_map.get(var_label, "")
            
            # Construct Prompt
            # We explicitly separate Demographics, Question, and Response for the model.
            prompt = (
                f"{narrative}\n"
                f"Demographics: {demog_str}.\n"
                f"Question ({var_label}): {question_text}\n"
                f"Response:"
            )
            
            # Construct Completion (Leading space is important for tokenization)
            completion = f" {response_text}"
            
            flattened_examples.append({
                "prompt": prompt,
                "completion": completion
            })
            
    return flattened_examples

def save_jsonl(data: List[Dict], filename: str):
    """Helper to save list of dicts to JSONL"""
    output_path = UNIVERSAL_FINAL_FOLDER / filename
    print(f"Saving {len(data)} examples to {output_path}...")
    
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def split_combine(train_plan, save: bool = True):
    if train_plan not in TRAIN_PLANS:
        raise ValueError(f"Plan {train_plan} not found.")
    
    train_data = []
    val_data = []
    test_data = []
    for dataset_name in ALL_DATA_MAPS.keys():
        package: DataPackage = build_steering_dataset(dataset_name, train_plan, save=False)

        _train = flatten_data_to_llama_format(package.get_data('train_data'), 'train')
        _val = flatten_data_to_llama_format(package.get_data('val_data'), 'val')
        _test= flatten_data_to_llama_format(package.get_data('test_data'), 'test')

        train_data.extend(_train)
        val_data.extend(_val)
        test_data.extend(_test)

    if save:
        save_jsonl(train_data, f"{train_plan}_train.jsonl")
        save_jsonl(val_data, f"{train_plan}_val.jsonl")
        save_jsonl(test_data, f"{train_plan}_test.jsonl")

        print("--- Processing Complete ---")
        print(f"Total Training Examples: {len(train_data)}")
        print(f"Total Validation Examples: {len(val_data)}")
        print(f"Total Test Examples: {len(test_data)}")

    # save data onto UNIVERSAL_FINAL_FOLDER > create a train, test and val file

    data_package = DataPackage('all', 'all')
    data_package.add_data('train_data', train_data)
    data_package.add_data('val_data', val_data)
    data_package.add_data('test_data', test_data) 
    return data_package

if __name__ == "__main__":
    split_combine('ideology_to_trump')