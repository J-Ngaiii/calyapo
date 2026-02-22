import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union

from calyapo.configurations.data_map_config import ALL_DATA_MAPS
from calyapo.configurations.config import DATA_PATHS, UNIVERSAL_NA_FILLER
from calyapo.data_preprocessing.cleaning_objects import Individual, TrainPlanWrapper, DataPackage
from calyapo.utils.persistence import *

def process_csv(data: pd.Dataframe, dataset_name: str, train_plan: str, debug: bool = False, verbose: bool = False) -> DataPackage:
    """
    Process a single CSV file or dataframe and return a DataPackage.
    """
    if 'time_period' in data.columns:
        # previous functions should create this col for in-memory processing
        time_period = str(data['time_period'].iloc[0])
    else:
        time_period = UNIVERSAL_NA_FILLER

    if time_period not in ALL_DATA_MAPS.get(dataset_name, {}):
        if verbose: print(f"No mapping for {dataset_name} in {time_period}. Skipping.")
        return None

    if debug: print(f"Base df:\n{data}")
    
    tp_wrap = TrainPlanWrapper(dataset_name, train_plan) # validates train_plan and dataset_name
    dataset_maps = ALL_DATA_MAPS[dataset_name][time_period]
    var2label = dataset_maps.get('var2label', {})
    label2var = {v: k for k, v in var2label.items()}

    # data specific arrays
    cleaned_data: List[Dict] = []
    train_data: List[Dict] = []
    val_data: List[Dict] = []
    test_data: List[Dict] = []
    for idx, row in data.iterrows():
        # ID Logic
        id_col = label2var.get('dataset_id')
        indiv_id = row[id_col] if (id_col and id_col in row) else idx
        
        entry = Individual(indiv_id, time_period, train_plan, dataset_name)

        # 1. Demographics
        for var_label in tp_wrap.get_var_lst('demo'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_demog(var_label, row[csv_col], debug)
            else:
                entry.add_demog(var_label, UNIVERSAL_NA_FILLER, debug)

        # 2. Train Questions
        for var_label in tp_wrap.get_var_lst('train_resp'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_train(var_label, row[csv_col])

        # 3. Val Questions
        for var_label in tp_wrap.get_var_lst('val_resp'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_val(var_label, row[csv_col])
        
        # 4. Test Questions
        for var_label in tp_wrap.get_var_lst('test_resp'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_test(var_label, row[csv_col])

        cleaned_data.append(entry.return_full_indiv_map())
        train_data.append(entry.return_split_indiv_map('train'))
        val_data.append(entry.return_split_indiv_map('val'))
        test_data.append(entry.return_split_indiv_map('test'))

    pack = DataPackage(dataset_name, train_plan, time_period)
    pack.add_data('full', cleaned_data)
    pack.add_data('train', train_data)
    pack.add_data('val', val_data)
    pack.add_data('test', test_data)
    
    return pack


def build_steering_dataset(
        data_or_path: Union[List[pd.DataFrame]|Path|str], 
        dataset_name: str, 
        train_plan: str = "ideology_to_trump", 
        out_path: str = None, 
        save: bool = True, 
        debug: bool = False, 
        verbose: bool = False):
    """
    Iterates through all CSVs for a dataset, cleans them, saves intermediates,
    and returns a combined DataPackage.
    """
    if isinstance(data_or_path, list) and all([isinstance(df, pd.DataFrame) for df in data_or_path]):
        if verbose: print(f"Processing {len(data_or_path)} DataFrames passed in-memory.")
        data_sources = data_or_path
    else:
        if data_or_path is None: # use default path if no given path
            data_or_path = DATA_PATHS[dataset_name]['intermediate']
        data_sources = list(Path(data_or_path).glob('*.csv'))
        if not data_sources:
            raise FileNotFoundError(f"No CSVs in {data_or_path}")
        if verbose: print(f"Found {len(data_sources)} files for {dataset_name}.")

    # whole-survey arrays
    master_full: List[Dict] = []
    master_train: List[Dict] = []
    master_val: List[Dict] = []
    master_test: List[Dict] = []
    for source in data_sources:
        # use file loader then make process csv just handle data
        # time period is baked into dataframe / csv anyway
        if isinstance(source, Path) or isinstance(source, str):
            data = file_loader(in_path=source, data_type='csv', verbose=verbose) # fixed to csv, any type conversion happens at raw layer
        else:
            data = source
        pack = process_csv(data=data, dataset_name=dataset_name, train_plan=train_plan, debug=debug, verbose=verbose)
        
        if not pack: continue # skip if config missing
        
        full_data = pack.get_data('full')
        master_full.extend(full_data)
        master_train.extend(pack.get_data('train'))
        master_val.extend(pack.get_data('val'))
        master_test.extend(pack.get_data('test'))

        if save:
            if out_path is None:
                out_path = DATA_PATHS[dataset_name]['processed']
            out_path.mkdir(parents=True, exist_ok=True)
            # e.g. ideology_to_trump_IGS_2024_processed.json
            out_name = f"{train_plan}_{dataset_name}_{pack.time_period}_processed.json"
            
            with open(out_path / out_name, 'w') as f:
                json.dump(full_data, f, indent=2)
            if verbose: print(f"Saved {len(full_data)} rows to {out_name}")

    master_pack = DataPackage(dataset_name, train_plan, "all_combined")
    master_pack.add_data('full', master_full)
    master_pack.add_data('train', master_train)
    master_pack.add_data('val', master_val)
    master_pack.add_data('test', master_test)

    if save:
        out_path = out_path or DATA_PATHS[dataset_name]['processed']
        out_path.mkdir(parents=True, exist_ok=True)
        out_name = f"{train_plan}_{dataset_name}_fullpack_processed.json"
        with open(out_path / out_name, 'w') as f:
            json.dump(master_pack.to_dict(), f, indent=2) 
        if verbose: print(f"Saved master data package for {dataset_name} to {out_name}")
    
    return master_pack