import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union

from calyapo.configurations.data_map_config import ALL_DATA_MAPS
from calyapo.configurations.config import DATA_PATHS, UNIVERSAL_NA_FILLER
from calyapo.data_preprocessing.cleaning_objects import Individual, TrainPlanWrapper, DataPackage
from calyapo.utils.persistence import *

def process_csv(data: pd.DataFrame, dataset_name: str, train_plan: str, debug: bool = False, verbose: bool = False) -> DataPackage:
    """
    Process a single CSV file or dataframe and return a DataPackage.
    """

    if 'time_period' in data.columns:
        # previous functions should create this col for in-memory processing
        time_period = str(data['time_period'].iloc[0])
    else:
        time_period = UNIVERSAL_NA_FILLER

    if time_period not in ALL_DATA_MAPS.get(dataset_name, {}):
        if verbose: print(f"(process__csv) No mapping for '{dataset_name}' in '{time_period}'. Skipping.")
        return None

    if debug: print(f"(process_csv | Debug) Base df empty: '{data is None}'")
    
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
        # ID logic
        id_col = label2var.get('dataset_id')
        indiv_id = row[id_col] if (id_col and id_col in row) else idx
        
        entry = Individual(indiv_id, time_period, train_plan, dataset_name)

        # 1. demographics
        for var_label in tp_wrap.get_var_lst('demo'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_demog(var_label, row[csv_col], debug)
            else:
                entry.add_demog(var_label, UNIVERSAL_NA_FILLER, debug)

        # 2. train questions
        for var_label in tp_wrap.get_var_lst('train_resp'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                # activates if given theres an nan value
                entry.add_train(var_label, row[csv_col])
                

        # 3. val questions
        for var_label in tp_wrap.get_var_lst('val_resp'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_val(var_label, row[csv_col])
        
        # 4. test questions
        for var_label in tp_wrap.get_var_lst('test_resp'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_test(var_label, row[csv_col])

        cleaned_data.append(entry.return_full_indiv_map())
        train_data.append(entry.return_split_indiv_map('train'))
        val_data.append(entry.return_split_indiv_map('val'))
        test_data.append(entry.return_split_indiv_map('test'))

    # all are the same length due to missing val handling in Individual class
    pack = DataPackage(
        dataset_name=dataset_name, 
        train_plan=train_plan, 
        time_period=time_period,
    )
    pack.add_data('full', cleaned_data)
    pack.add_data('train', train_data)
    pack.add_data('val', val_data)
    pack.add_data('test', test_data)

    if debug:
        print(pack)
    
    return pack


def split_questions(
        data: List[pd.DataFrame], 
        dataset_name: str, 
        train_plan: str, 
        out_path: str = None, 
        save: bool = True, 
        debug: bool = False, 
        verbose: bool = False
    ) -> DataPackage:
    """
    Iterates through all CSVs for a dataset, cleans them, saves intermediates,
    and returns a combined DataPackage.
    """    
    # whole-survey arrays
    master_full: List[Dict] = []
    master_train: List[Dict] = []
    master_val: List[Dict] = []
    master_test: List[Dict] = []
    for df in data:
        pack = process_csv(data=df, dataset_name=dataset_name, train_plan=train_plan, debug=debug, verbose=verbose)

        
        if not pack: continue # skip if config missing
        
        full_data = pack.get_data('full')
        if debug:
            print(f"(split_question | Debug) full_data empty:{full_data is None}")
        master_full.extend(full_data)
        master_train.extend(pack.get_data('train'))
        master_val.extend(pack.get_data('val'))
        master_test.extend(pack.get_data('test'))

        if save:
            assert out_path is not None, f"(split_questions) Cannot save files without valid out path"
            # e.g. ideology_to_trump_IGS_2024_processed.json
            out_name = f"{train_plan}_{dataset_name}_{pack.time_period}_processed.json"
            
            file_saver(out_path=Path(out_path / out_name), data=full_data, data_type='DataPackage', indnt=2, verbose=verbose)

    master_pack = DataPackage(dataset_name, train_plan, "all_combined")
    master_pack.add_data('full', master_full)
    master_pack.add_data('train', master_train)
    master_pack.add_data('val', master_val)
    master_pack.add_data('test', master_test)

    if debug:
        print(f"(split_question | Debug) master_pack['full'] empty: {master_pack.get('full') is None}")

    if save:
        assert out_path is not None, f"(split_questions) Cannot save files without valid out path"
        out_name = f"{train_plan}_{dataset_name}_fullpack_processed.json"
        file_saver(out_path=Path(out_path / out_name), data=master_pack, data_type='DataPackage', indnt=2, verbose=verbose)
    
    return master_pack