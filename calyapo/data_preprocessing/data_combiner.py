import json
import os
from pathlib import Path
from typing import List, Dict, Any

from calyapo.configurations.data_map_config import ALL_DATA_MAPS, TRAIN_PLANS
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER, UNIVERSAL_NA_FILLER
from calyapo.data_preprocessing.cleaning_objects import DataPackage
from calyapo.data_preprocessing.clean_datasets import build_steering_dataset

UNIVERSAL_FINAL_FOLDER = Path(UNIVERSAL_FINAL_FOLDER)
UNIVERSAL_FINAL_FOLDER.mkdir(parents=True, exist_ok=True)

def format_demographics(demog_dict: Dict[str, str]) -> str:
    """Converts {'age': '18-29'} -> "Age: 18-29"."""
    parts = []
    for k, v in demog_dict.items():
        if v == UNIVERSAL_NA_FILLER: continue 
        clean_key = k.replace('_', ' ').title()
        parts.append(f"{clean_key}: {v}")
    return ", ".join(parts)

def flatten_data_to_llama_format(raw_data_list: List[Dict], section_key: str) -> List[Dict[str, str]]:
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
        section_data = entry.get(section_key, {})
        
        # These maps are parallel
        text_map = section_data.get('var_label2qst_text', {})
        choices_map = section_data.get('var_label2qst_choices', {})
        # This one contains the answer logic {var: {option_text, option_letter}}
        options_map = section_data.get('var_label2qst_option', {})
        
        # 3. Create Examples
        for var_label, answer_data in options_map.items():
            
            # answer_data is a dictionary, e.g. {'option_letter': 'A', 'option_text': 'Yes'}
            # Note: eval() might be needed if it was saved as string representation of dict
            if isinstance(answer_data, str):
                try:
                    import ast
                    answer_data = ast.literal_eval(answer_data)
                except:
                    continue

            response_text = answer_data.get('option_text')
            target_letter = answer_data.get('option_letter')
            
            # Skip if missing
            if not target_letter or target_letter == UNIVERSAL_NA_FILLER:
                continue
                
            question_text = text_map.get(var_label, "")
            choices_block = choices_map.get(var_label, "")
            
            # Construct Prompt
            prompt = (
                f"{narrative}\n"
                f"Demographics: {demog_str}.\n"
                f"Question ({var_label}): {question_text}\n"
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

def save_jsonl(data: List[Dict], filename: str):
    output_path = UNIVERSAL_FINAL_FOLDER / filename
    print(f"Saving {len(data)} examples to {output_path}...")
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def split_combine(train_plan: str, save: bool = True):
    print(f"--- Running Split & Combine for {train_plan} ---")
    if train_plan not in TRAIN_PLANS:
        raise ValueError(f"Plan {train_plan} not found.")
    
    train_data = []
    val_data = []
    test_data = []
    
    datasets_in_plan = TRAIN_PLANS[train_plan].keys()
    
    for dataset_name in datasets_in_plan:
        print(f"Processing {dataset_name}...")
        
        # Get Package (Runs cleaning)
        package: DataPackage = build_steering_dataset(dataset_name, train_plan, save=True)

        # Flatten
        train_data.extend(flatten_data_to_llama_format(package.get_data('train'), 'train'))
        val_data.extend(flatten_data_to_llama_format(package.get_data('val'), 'val'))
        test_data.extend(flatten_data_to_llama_format(package.get_data('test'), 'test'))

    if save:
        save_jsonl(train_data, f"{train_plan}_train.jsonl")
        save_jsonl(val_data, f"{train_plan}_val.jsonl")
        save_jsonl(test_data, f"{train_plan}_test.jsonl")

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

if __name__ == "__main__":
    split_combine('ideology_to_trump')