import pandas as pd
import numpy as np
import random
from typing import List, Dict
from collections import defaultdict

from calyapo.data_preprocessing.cleaning_objects import DataPackage, Individual
from calyapo.configurations.config import UNIVERSAL_RANDOM_SEED, UNIVERSAL_NA_FILLER
from calyapo.utils.sampling import *
from calyapo.utils.persistence import *

random.seed(UNIVERSAL_RANDOM_SEED)

def get_unique_id(indiv_map: Dict):
    if 'uniqueid' in indiv_map:
        id = indiv_map.get('uniqueid')
    else:
        id = indiv_map.get('id')

    return id

def indiv_valid_response(indiv_map: Dict, splt: str, check: str = None, debug: str = False):
    if check is None:
        check = 'all'
    
    option_map = indiv_map[splt].get('var_label2qst_option', {})

    # if debug:
        # if indiv_map['time'] != '202402':
        #     print(f"(indiv_valid | Debug) time period '{indiv_map['time']}' option_map: {option_map}")
        
    if check == 'all':
        # need to iterate thru all the label : option_dict values
        # option_map = {
        #   'ideology' : {'option_letter' : 'A', 'option_text' : 'XYZ'}, 'trump' : {'option_letter' : 'B', 'option_text' : 'shdsah'}
        # }
        has_data = all(
            # responses to all questions not NaN
            opt.get('option_letter') != UNIVERSAL_NA_FILLER 
            for opt in option_map.values()
        ) # training plans with multiple train/val/test questions we want ALL questions to be not NA to include
    elif check == 'any':
        has_data = any(
            # any response to a question not NaN
            opt.get('option_letter') != UNIVERSAL_NA_FILLER 
            for opt in option_map.values()
        )

    return has_data

def split_ratio(
        package: DataPackage, 
        target_ratios: Dict[str, float], 
        homogenous_plan: bool, 
        ques_split_varying: bool,
        train_setting: int, 
        valid_indiv_setting: str = None, 
        out_path: str = None, 
        save: bool = False, 
        debug: bool = False, 
        verbose: bool = False 
    ):
    """
    Splits up data according to predefined ratios so the model sees new individuals and questions at validation.
    In the case of validating on multiple questions => don't want model to learn to predict "no response"
    => only select individuals who responded to ALL questions into the validation set. 

    Output must put it each unique value into a particular split (train/val/test) with no overlaps in Train Settings 1 and 2
    """
    if debug:
            test = (
                len(package['train']) == len(package['val'])
                and
                len(package['val']) == len(package['test'])
                and
                len(package['train']) == len(package['test'])
            )
            print(f"(split_ratio | Debug) Length of all splits equal: '{test}'")
            
    if homogenous_plan and not ques_split_varying or train_setting == 1:
        if debug:
            print(f"(split_ratio| Debug) Train Setting 1: Same question, new individual processing active")
        # if you are train, val, testing on the exact same question => exact same variable label

        # remove individuals without a valid response
        all_valid_indivs = []
        seen = set()
        for indiv_map in package['full']:
            id = get_unique_id(indiv_map)
            
            if id in seen:
                continue

            # doesn't matter which split, all the questions will be the same
            if indiv_valid_response(indiv_map=indiv_map, splt='train', check=valid_indiv_setting, debug=debug):
                all_valid_indivs.append(indiv_map)

        if debug:
            print(f"(split_ratio| Debug) num unique individuals: '{len(all_valid_indivs)}'\n(split_ratio| Debug) length of initial package: '{len(package['full'])}'")
        
        indices = np.random.choice(len(all_valid_indivs), size=len(all_valid_indivs), replace=False)

        random.shuffle(indices)
        
        n = len(all_valid_indivs)
        train_end = int(n * target_ratios['train'])
        val_end = train_end + int(n * target_ratios['val'])
        
        outPack = DataPackage(package.dataset_name, package.train_plan, package.time_period)
        outPack['train'] = [all_valid_indivs[i] for i in indices[:train_end]]
        outPack['val'] = [all_valid_indivs[i] for i in indices[train_end:val_end]]
        outPack['test'] = [all_valid_indivs[i] for i in indices[val_end:]]
        
    elif train_setting == 2:
        if debug:
            print(f"(split_ratio| Debug) Train Setting 2: New question, same individual processing active")

        train_valid_indivs = []
        val_valid_indivs = []
        test_valid_indivs = []
        for indiv_map in package['full']:
            if indiv_valid_response(indiv_map=indiv_map, splt='train', check=valid_indiv_setting, debug=debug):
                train_valid_indivs.append(indiv_map)
            if indiv_valid_response(indiv_map=indiv_map, splt='val', check=valid_indiv_setting, debug=debug):
                val_valid_indivs.append(indiv_map)
            if indiv_valid_response(indiv_map=indiv_map, splt='val', check=valid_indiv_setting, debug=debug):
                test_valid_indivs.append(indiv_map)

        if debug:
            print(f"(split_ratio| Debug) num train individuals: '{len(train_valid_indivs)}'\n(split_ratio| Debug) num val individuals: '{len(val_valid_indivs)}'\n(split_ratio| Debug) num val individuals: '{len(test_valid_indivs)}'")
        
        # reduce total num of val and test datapoints
        num_train = len(train_valid_indivs)
        val_size = int(num_train * target_ratios['val'])
        test_size = int(num_train * target_ratios['test'])
        val_indices = np.random.choice(len(val_valid_indivs), size=min(len(val_valid_indivs), val_size), replace=False)
        test_indices = np.random.choice(len(test_valid_indivs), size=min(len(test_valid_indivs), test_size), replace=False)

        outPack = DataPackage(package.dataset_name, package.train_plan, package.time_period)
        outPack['train'] = train_valid_indivs
        outPack['val'] = list(np.array(val_valid_indivs)[val_indices])
        outPack['test'] = list(np.array(test_valid_indivs)[test_indices])
    else:
        # see which split set has the least data points --> check if that is less than the respective ratios
        # if num data points in split set with the least data points is less than its ratio (eg we have less val datapoints than what we should have for val_ratio = 0.2)
        # then just run it but do a print
        # else trim it down

        # counting valid individuals

        if debug:
            print(f"(split_ratio| Debug) Train Setting 3: New question, new individual processing active")
        sets = ['train', 'val', 'test']
        splits: List[Dict] = [0] * 3
        for i in range(len(sets)):
            spl = sets[i]
            valid_indivs = [] # assemble all the individual ids
            for indiv_map in package[spl]:
                indiv_map: Dict
                if indiv_valid_response(indiv_map=indiv_map, splt=spl, check=valid_indiv_setting, debug=debug):
                    all_valid_indivs.append(indiv_map)
            
            splits[i] = valid_indivs

        if debug:
            print(f"(split_ratio | Debug) num datapoints per split:\nTrain: '{len(splits[0])}'\nVal: '{len(splits[1])}'\nTest: '{len(splits[2])}'")
        unpacked_distribution = [target_ratios['train'], target_ratios['val'], target_ratios['test']]
        min_bucket_idx = np.argmin([len(splits[i]) for i in range(len(splits))])
        sampler_output = exhaustive_hierarchal_sample(value_buckets=splits, targ_bucket_idx=min_bucket_idx, bucket_distrib=unpacked_distribution, debug=debug)

        outPack = DataPackage(package.dataset_name, package.train_plan, package.time_period)
        outPack['train'] = sampler_output[0]
        outPack['val'] = sampler_output[1]
        outPack['test'] = sampler_output[2]

    if debug:
        print(f"(split_ratio | Debug) FINAL num datapoints per split:\nTrain: '{len(outPack['train'])}'\nVal: '{len(outPack['val'])}'\nTest: '{len(outPack['test'])}'")
    if save:
        assert out_path is not None, f"(split_ratio) Cannot save if out_path not specified"
        file_name = f"{package.train_plan}_{package.dataset_name}_fullsplit.json"
        full_path = Path(out_path) / file_name
        file_saver(out_path=full_path, data=outPack, data_type='DataPackage', indnt=2, verbose=verbose)

    return outPack

def split_ratio_validator(pack: DataPackage, valid_indiv_setting: str = None, verbose: bool = False, debug: bool = False):
    """
    Ensures every unique individual exists in exactly one split with no overlaps.
    Catch both duplicates within a set and leakage across sets.
    """
    id_registry = defaultdict(list) # maps individuals : splits they're in
    
    splits = ['train', 'val', 'test']
    
    # append all the individuals into a registry first
    for splt in splits:
        if splt not in pack:
            if verbose: print(f"(split_ratio_validator | INFO) split '{splt}' missing from package keys")
            continue
            
        # handles for individuals being in train multiple times
        for indiv_map in pack[splt]:
            indiv_id = get_unique_id(indiv_map)
            id_registry[indiv_id].append(splt)
            if not indiv_valid_response(indiv_map=indiv_map, splt=splt, check=valid_indiv_setting, debug=debug):
                id_registry[indiv_id].append(f"{splt}-contains NaN")

    leaks = {
        idx: appearances 
        for idx, appearances in id_registry.items() 
        if len(appearances) > 1
    }

    if leaks:
        error_report = "\n".join([f"ID {k}: found in {v}" for k, v in leaks.items()])
        msg = f"(split_ratio_validator | ERROR) Data leakage/duplicates detected!\n{error_report}"
        
        raise ValueError(msg)
    
    if verbose:
        total_unique = len(id_registry)
        print(f"(split_ratio_validator | SUCCESS) No leakage detected across {total_unique} individuals.")
