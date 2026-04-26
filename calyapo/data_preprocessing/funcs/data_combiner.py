import json
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable

from calyapo.configurations.data_map_config import TRAIN_PLANS, VARLABEL_DESC
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER, UNIVERSAL_NA_FILLER, IGS_SURVEY_WAVE_DESC, POLLING_FIRM_DESC
from calyapo.data_preprocessing.cleaning_objects import DataPackage, Individual
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
    meta_configs = []
    
    for entry in raw_data_list:
        # track IDS here
        time_label = entry.get('time', 'Unknown')
        polling_date = IGS_SURVEY_WAVE_DESC.get(time_label, 'Unkown')
        dataset_label = entry.get('dataset', 'Unknown')
        polling_firm = POLLING_FIRM_DESC.get(dataset_label, 'Unkown')
        demog_str = format_demographics(entry.get('demog', {}))
        
        narrative = f"You are a survey respondent based in California from the {polling_date} {polling_firm} polling wave."
        
        section_data = entry.get(split, {})
        
        text_map = section_data.get('var_label2qst_text', {})
        choices_map = section_data.get('var_label2qst_choices', {})
        # options map contains the answer logic {var: {option_text, option_letter}}
        options_map = section_data.get('var_label2qst_option', {})
        
        # each datapoint is a unique individual-question pair so now we iterate thru questions
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
            
            #FIX skip if missing 
            if not target_letter or target_letter == UNIVERSAL_NA_FILLER:
                continue
                
            question_text = text_map.get(var_label, "")
            choices_block = choices_map.get(var_label, "")
            question_varlabel_desc = VARLABEL_DESC[var_label]

            # construct prompt
            prompt = (
                f"{narrative}\n"
                f"You have the following demographic profile: {demog_str}.\n"
                f"Answer the following question about {question_varlabel_desc} according to your demographic profile: {question_text}\n"
                f"{choices_block}\n"
                f"Answer:"
            )
            
            # construct completiion target (just the target letter)
            completion = f"{target_letter}"
            
            flattened_examples.append({
                "prompt": prompt,
                "completion": completion
            })
            
            meta_configs.append({
                'id': entry.get('id', 'Unknown'), 
                'uniqueid': entry.get('uniqueid', 'Unknown'), 
                'time_period': entry.get('time', 'Unknown'), 
                'dataset': entry.get('dataset', 'Unknown'), 
            })
    return flattened_examples, meta_configs

def save_jsonl(data: List[Dict], filename: str, out_path: str = None, verbose: bool = False):
    if out_path is None:
        out_path = Path(UNIVERSAL_FINAL_FOLDER) / filename
    elif Path(out_path).is_dir():
        out_path = Path(out_path) / filename
    if verbose: print(f"(split_combine | save_jsonl) Saving {len(data)} examples to {out_path}...")
    with open(out_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def split_combine(
        package: DataPackage,
        out_path: str = None,
        save: bool = True,
        debug: bool = False,
        verbose: bool = True
    ):
    data_dict = {
        'train' : {
            'data' : [], # when function finishes this will be list of {prompt : completion} dictionaries
            'meta' : []
        },
        'val' : {
            'data' : [],
            'meta' : []
        },
        'test' : {
            'data' : [],
            'meta' : []
        }
    }
       
    # needs to be able to take different packages in memory
    if verbose:
        print(f"(split_combine) There are '{len(package['dataset_packages'])}' datasets to process")
    for dataset_name, inpack in package['dataset_packages'].items():
        if verbose: print(f"(split_combine) Processing {dataset_name}...")
        if debug:
            print(f"(split_combine | Debug) Data Package: {package}")

        for split, split_dict in data_dict.items():
            indiv_map: List[Dict] = inpack.get(split)
            data, meta = flatten_data_to_llama_format(indiv_map, split)
            split_dict['data'].extend(data)
            split_dict['meta'].extend(meta)

    if save:
        assert out_path is not None, f"(split_combine | WARNING) Cannot have no out_path if saving."
        for split, split_dict in data_dict.items():
            save_jsonl(split_dict['data'], f"{package.train_plan}_{split}.jsonl", out_path, verbose)
            save_jsonl(split_dict['meta'], f"{package.train_plan}_{split}_meta.jsonl", out_path, verbose)


    out_pack = DataPackage(package.dataset_name, package.train_plan, package.time_period)
    for split, split_dict in data_dict.items():
        out_pack[split] = split_dict['data']
        out_pack[f"{split}_meta"] = split_dict['meta']

    return out_pack

def subdivide_training_set(
        package: DataPackage, 
        subproportions: Iterable[float], 
        out_path: str = None, 
        seed: int = 42, 
        save: bool = True, 
        debug: bool = False, 
        verbose: bool = True
    ):
    out_pack = DataPackage(package.dataset_name, package.train_plan, package.time_period)
    np.random.default_rng(seed)

    base_train_set = package['train']  # list of {prompt: completion}
    base_train_meta = package['train_meta']
    train_size_total = len(base_train_set)
    rng = np.random.default_rng(seed)
    for proportion in subproportions:
        indices = rng.choice(train_size_total, size=int(train_size_total * proportion), replace=False)
        train_subproportion = []
        train_subproportion_meta = []
        for i in indices:
            train_subproportion.append(base_train_set[i])
            train_subproportion_meta.append(base_train_meta[i])
        out_pack[f"train_{str(proportion)}"] = train_subproportion
        out_pack[f"train_{str(proportion)}_meta"] = train_subproportion_meta

        if save:
            assert out_path is not None, f"(subdivide_training_set | WARNING) Cannot have no out_path if saving."
            save_jsonl(train_subproportion, f"{package.train_plan}_train_{str(proportion)}.jsonl", out_path, verbose)
            save_jsonl(train_subproportion_meta, f"{package.train_plan}_train_{str(proportion)}_meta.jsonl", out_path, verbose)

    return out_pack

