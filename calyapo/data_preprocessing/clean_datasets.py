import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import re

from calyapo.configurations.data_map_config import ALL_DATA_MAPS, TRAIN_PLANS
from calyapo.configurations.config import DATA_PATHS
from calyapo.data_preprocessing.cleaning_objects import Individual, DataPackage


NA_FILLER = "not available"

def invert_mapping(opt_map: Dict[str, Any]) -> Dict[int, str]:
    """
    Inverts the mapping schema from {Text: Code} to {Code: Text} for lookup.
    Returns {<raw encoding> : < codebook option text>} 
    - eg. {18 : '18-29', 19 : '18-29', ...}
    """
    inverted = {}
    for label, code in opt_map.items():
        if isinstance(code, list): # handles age
            for c in code:
                inverted[c] = label
        else:
            inverted[code] = label
    return inverted

def get_dataset_decoders(dataset_name: str, time_period: str) -> Dict[str, Dict[int, str]]:
    """
    Generates a dictionary of decoders for a specific dataset.
    Returns: {<variable_label> : <inverted mapping>}
    - eg. { 'partyid': {1: 'Democrat', ...}, 'age': {1: '18-29', ...} }
    """
    if dataset_name not in ALL_DATA_MAPS:
        raise ValueError(f"Unknown dataset {dataset_name} in ALL_DATA_MAPS")
    if time_period not in ALL_DATA_MAPS[dataset_name]:
        raise ValueError(f"Unknown time period {time_period} for dataset {dataset_name}")
    
    opt2label = ALL_DATA_MAPS[dataset_name][time_period].get('label2opt', {})
    
    decoders = {}
    for variable_label, mapping in opt2label.items():
        if mapping: # Only invert if mapping exists
            decoders[variable_label] = invert_mapping(mapping)
        
    return decoders

def build_steering_dataset(dataset_name: str, train_plan: str = "ideology_to_trump", save: bool = True):
    """
    Main ETL function.
    Always saves one JSON per time period. 
    if retain_time = False then instead of saving a dict of each cleaned dataset --> just combine them all
    """
    # validation checks
    if train_plan not in TRAIN_PLANS:
        raise ValueError(f"Unknown plan: {train_plan}. Check TRAIN_PLANS.")
    plan_config = TRAIN_PLANS[train_plan] # plan_config = {'IGS' : IGS_IDEO_IDEO, 'PPIC' : PPIC_IDEO_IDEO, ... }
    if dataset_name not in plan_config:
        print(f"Dataset '{dataset_name}' is not part of plan '{train_plan}'. Skipping.")
        return
    if dataset_name not in DATA_PATHS:
        raise ValueError(f"No path defined for {dataset_name} in DATA_PATHS")
    if 'raw' not in DATA_PATHS[dataset_name]:
        raise ValueError(f"No path for raw folder for {dataset_name} in DATA_PATHS")
    if 'processed' not in DATA_PATHS[dataset_name]:
        raise ValueError(f"No path for processed folder for {dataset_name} in DATA_PATHS")
    if dataset_name not in ALL_DATA_MAPS:
        raise ValueError(f"Unknown dataset {dataset_name} in ALL_DATA_MAPS")
    
    # setting up plans, these should be INVARIANT ACROSS TIME AND DATASET (they are variable labels)
    dataset_plan = plan_config[dataset_name] # dataset_plan = {'demo' : ['age', 'partyid'], 'train_resp' : ['ideology'], 'val_resp' : ['trump_opinion']}
    demo_vars = dataset_plan['demo'] # ['age', 'partyid']
    train_resp_vars = dataset_plan['train_resp'] # ['ideology']
    val_resp_vars = dataset_plan['val_resp'] # ['trump_opinion']
    test_resp_vars = dataset_plan['test_resp'] # ['abortion_senate']
    
    # loading data
    raw_dir = DATA_PATHS[dataset_name]['raw']
    csv_files = list(raw_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    else:
        print(f"Loaded {len(csv_files)} csv files from raw directory")
    
    all_data = []
    all_train = []
    all_val = []
    all_test = []
    for i in range(len(csv_files)):
        csv_path = csv_files[i] # seperate function to do per csv
        df = pd.read_csv(csv_path)
        time_period = csv_path.stem.split('_')[-1] # format is something like raw_ppic_dec2019

        try:
            dataset_maps = ALL_DATA_MAPS[dataset_name][time_period]
        except KeyError:
            print(f"Warning: Config for '{time_period}' not found. Skipping file.")
            continue

        # setting up question text and variable mappers NOT invariant across time
        label2qes = dataset_maps.get('label2qes', {})
        var2label = dataset_maps.get('var2label', {})
        label2var = {v: k for k, v in var2label.items()} # label2var = {'age': 'Q21', 'ideology': 'Q27'}
        opt_decoders = get_dataset_decoders(dataset_name, time_period) # opt_decoders = { 'partyid': {1: 'Democrat', ...}, 'age': {1: '18-29', ...} }

        cleaned_data = []
        train_data = []
        val_data = []
        test_data = []

        # start cleaning
        for idx, row in df.iterrows():
            entry = {
                "id" : idx, # default to row index, can use igs id or can just not
                "time" : time_period, 
                "demog" : {},
                "dataset" : dataset_name,  
                "train" : {
                    "var_label2qst_text": {}, # we keep track of the variable label (`ideology`), question text and question response
                    "var_label2qst_option": {}
                }, 
                "val" : {
                    "var_label2qst_text": {},
                    "var_label2qst_option": {}
                }, 
                "test" : {
                    "var_label2qst_text": {}, 
                    "var_label2qst_option": {}
                }, 
            }
            
            # try to find an explicit ID column
            id_col = label2var['dataset_id'] # try to encode ids as `ID` in data_mappings regardless of dataset
            if id_col and id_col in row:
                indiv_id = row[id_col]
            else:
                indiv_id = idx
            entry = Individual(indiv_id, time_period, dataset_name)

            # populate with demog
            for variable_label in demo_vars:
                if variable_label not in label2var:
                    print(f"Warning: {variable_label} not found var2label")
                    continue 
                csv_col = label2var.get(variable_label)
                
                # Safety check: does column exist in this CSV?
                if csv_col not in row:
                    print(f"Warning: column name {csv_col} not found in dataset")
                    entry['demog'][variable_label] = NA_FILLER
                    continue 
                
                raw_demog_option = str(row[csv_col])
                
                if variable_label in opt_decoders and raw_demog_option in opt_decoders[variable_label]:
                    decoded_demog = opt_decoders[variable_label][raw_demog_option]
                else:
                    decoded_demog = NA_FILLER # missing handling is use NA filler to avoid confusing the LLM 
                
                # Handle NaNs
                if pd.notna(decoded_demog):
                    entry.add_demog(variable_label, decoded_demog)
                else:
                    entry.add_demog(variable_label, NA_FILLER)

            # populate train
            for variable_label in train_resp_vars:
                csv_col = label2var.get(variable_label)
                if not csv_col or csv_col not in row:
                    continue
                    
                raw_train_option = str(row[csv_col]) # have more specific names, I would rather variable error rather than silently training leak 
                
                if variable_label in opt_decoders and raw_train_option in opt_decoders[variable_label]:
                    decoded_train = opt_decoders[variable_label][raw_train_option]
                else:
                    decoded_train = NA_FILLER

                if pd.notna(decoded_train):
                    ques_txt = label2qes.get(variable_label, "Missing Question Text")
                    entry.add_train(variable_label, ques_txt, decoded_train)
                else:
                    entry.add_train(variable_label, ques_txt, NA_FILLER)
            
            # populate val
            for variable_label in val_resp_vars:
                csv_col = label2var.get(variable_label)
                if not csv_col or csv_col not in row:
                    continue
                    
                raw_val_option = str(row[csv_col])
                
                if variable_label in opt_decoders and raw_val_option in opt_decoders[variable_label]:
                    decoded_val = opt_decoders[variable_label][raw_val_option]
                else:
                    decoded_val = NA_FILLER
                    
                if pd.notna(decoded_val):
                    ques_txt = label2qes.get(variable_label, "Missing Question Text")
                    entry.add_val(variable_label, ques_txt, decoded_val)
                else:
                    entry.add_val(variable_label, ques_txt, NA_FILLER)

            # populate test
            for variable_label in test_resp_vars:
                csv_col = label2var.get(variable_label)
                if not csv_col or csv_col not in row:
                    continue
                    
                raw_test_option = str(row[csv_col])
                
                if variable_label in opt_decoders and raw_test_option in opt_decoders[variable_label]:
                    decoded_test = opt_decoders[variable_label][raw_test_option]
                else:
                    decoded_test = NA_FILLER

                if pd.notna(decoded_test):
                    ques_txt = label2qes.get(variable_label, "Missing Question Text")
                    entry.add_test(variable_label, ques_txt, decoded_test)
                else:
                    entry.add_test(variable_label, ques_txt, NA_FILLER)    
                
            cleaned_data.append(entry.return_full_individual())
            train_data.append(entry.return_train_individual())
            val_data.append(entry.return_val_individual())
            test_data.append(entry.return_test_individual())

        # populate `all_data`
        all_data.extend(cleaned_data)
        all_train.extend(train_data)
        all_val.extend(val_data)
        all_test.extend(test_data)
        # --- 5. Save Intermediate Output ---
        print(f"Successfully processed {len(cleaned_data)} samples.")
        if save:
            output_dir = DATA_PATHS[dataset_name]['processed']
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{train_plan}_{dataset_name}_{time_period}_processed.json"
            
            with open(output_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
                
            print(f"Saved to: {output_path}") 
    

    data_package = DataPackage(dataset_name, 'all')
    data_package.add_data('all_data', all_data)
    data_package.add_data('train_data', all_train)
    data_package.add_data('val_data', all_val)
    data_package.add_data('test_data', all_test) 
    return data_package


if __name__ == "__main__":
    build_steering_dataset('IGS', 'ideology_to_trump')