import pandas as pd
import numpy as np
from typing import Union, List, Dict, Iterable
from collections import defaultdict

from calyapo.data_preprocessing.cleaning_objects import DataPackage, Individual
from calyapo.configurations.config import UNIVERSAL_NA_FILLER
from calyapo.utils.sampling import *
from calyapo.utils.persistence import *

def get_unique_id(indiv_map: Dict):
    if 'uniqueid' in indiv_map:
        id = indiv_map.get('uniqueid')
    else:
        id = indiv_map.get('id')

    return id

def _indiv_helper(option_map: Dict, check: str) -> bool:
    """
    Takes in only one option map. Combine outputs from stacking these functions together then use union/intersection operations 
    to find relevant combinations (eg all individuals who are in the train and validation set).
    """

    results = [opt.get('option_letter') != UNIVERSAL_NA_FILLER 
            for opt in option_map.values()]
    if check == 'all':
        # need to iterate thru all the label : option_dict values
        # option_map = {
        #   'ideology' : {'option_letter' : 'A', 'option_text' : 'XYZ'}, 'trump' : {'option_letter' : 'B', 'option_text' : 'shdsah'}
        # }
        has_data = all(results) 
        # training plans with multiple train/val/test questions we want ALL questions to be not NA to include
    elif check == 'any':
        has_data = any(results)

    return has_data
    
def indiv_valid_response(indiv_map: Dict, splt: Union[str|Iterable[str]], check: str = None, debug: str = False) -> bool:
    "Checks if a given indiv_map corresponding to an individual is valid or not under different rules"
    if check is None:
        check = 'all'
    
    def get_option_map(inputted_splt):
        return indiv_map[inputted_splt].get('var_label2qst_option', {})
    
    if isinstance(splt, str):
        option_map = get_option_map(splt)
        has_data = _indiv_helper(option_map=option_map, check=check)
    elif isinstance(splt, Iterable):
        # each index corresponds to a split
        # each elem corresponds to if the individual has a valid response to the questions corresponding with that split
        indiv_valid_on_splts = [_indiv_helper(get_option_map(curr_split), check) for curr_split in splt]
        has_data = all(indiv_valid_on_splts) # individual must be valid across all inputted split
    else:
        raise ValueError(f"(indiv_valid_response) Inputted splt '{splt}' has invalid datatype '{type(splt)}'")

    return has_data

def split_ratio(
        package: DataPackage, 
        target_ratios: Dict[str, float], 
        homogenous_plan: bool, 
        ques_split_varying: bool, 
        train_setting: int, 
        valid_indiv_setting: str = None, 
        out_path: str = None, 
        seed: int = 42, 
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

    rng = np.random.default_rng(seed)

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
        
        indices = rng.permutation(len(all_valid_indivs)) # shuffle using more up to date numpy rng generator
        
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
            if indiv_valid_response(indiv_map=indiv_map, splt=['train', 'val'], check=valid_indiv_setting, debug=debug):
                val_valid_indivs.append(indiv_map)
            if indiv_valid_response(indiv_map=indiv_map, splt=['train', 'test'], check=valid_indiv_setting, debug=debug):
                test_valid_indivs.append(indiv_map)

        if debug:
            print(f"(split_ratio| Debug) num train individuals: '{len(train_valid_indivs)}'\n(split_ratio| Debug) num val individuals: '{len(val_valid_indivs)}'\n(split_ratio| Debug) num val individuals: '{len(test_valid_indivs)}'")

        # reduce total num of val and test datapoints
        # in order to be consistent with standard ML literature make val and test size a fraction of the training set size NOT their own respective sets
        # otherwise it's possible for the validation and test sizes to be way larger than the train size even with small target_ratio values
        num_train = len(train_valid_indivs)
        val_size = int(num_train * target_ratios['val'])
        test_size = int(num_train * target_ratios['test'])
        val_idx = rng.choice(len(val_valid_indivs), size=min(len(val_valid_indivs), val_size), replace=False)
        test_idx = rng.choice(len(test_valid_indivs), size=min(len(test_valid_indivs), test_size), replace=False)

        outPack = DataPackage(package.dataset_name, package.train_plan, package.time_period)
        outPack['train'] = train_valid_indivs
        outPack['val'] = [val_valid_indivs[i] for i in val_idx]
        outPack['test'] = [test_valid_indivs[i] for i in test_idx]
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

    TODO: Build out to also validate training setting #2 to make sure individuals in test are also in train
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
