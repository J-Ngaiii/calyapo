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

def split_ratio(
        package: DataPackage, 
        target_ratios: Dict[str, float], 
        homogenous_plan: bool, 
        ques_split_varying: bool,
        out_path: str = None, 
        save: bool = False, 
        debug: bool = False, 
        verbose: bool = False 
    ):
    """
    Splits up data according to predefined ratios so the model sees new individuals and questions at validation.
    In the case of validating on multiple questions => don't want model to learn to predict "no response"
    => only select individuals who responded to ALL questions into the validation set. 
    """
    if homogenous_plan and not ques_split_varying:
        # if you are train, val, testing on the exact same question => exact same variable label
        all_data = package['full']
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        
        n = len(all_data)
        train_end = int(n * target_ratios['train'])
        val_end = train_end + int(n * target_ratios['val'])
        
        package['train'] = [all_data[i] for i in indices[:train_end]]
        package['val'] = [all_data[i] for i in indices[train_end:val_end]]
        package['test'] = [all_data[i] for i in indices[val_end:]]
    else:
        # see which split set has the least data points --> check if that is less than the respective ratios
        # if num data points in split set with the least data points is less than its ratio (eg we have less val datapoints than what we should have for val_ratio = 0.2)
        # then just run it but do a print
        # else trim it down

        if debug:
            test = (
                len(package['train']) == len(package['val'])
                and
                len(package['val']) == len(package['test'])
                and
                len(package['train']) == len(package['test'])
            )
            print(f"Length of all splits equal: '{test}'")

        # counting valid individuals
        sets = ['train', 'val', 'test']
        splits: List[Dict] = [0] * 3
        for i in range(len(sets)):
            spl = sets[i]
            valid_indivs = [] # assemble all the individual ids
            for indiv_map in package[spl]:
                indiv_map: Dict
                option_map = indiv_map[spl].get('var_label2qst_option', {})
                
                has_data = all(
                    opt.get('option_letter') != UNIVERSAL_NA_FILLER 
                    for opt in option_map.values()
                ) # training plans with multiple train/val/test questions we want ALL questions to be not NA to include
                
                if has_data:
                    valid_indivs.append(indiv_map)
            
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

        if save:
            assert out_path is not None, f"(split_ratio) Cannot save if out_path not specified"
            file_saver(out_path=out_path, data=outPack, data_type='DataPackage', indnt=2, verbose=verbose)

        return outPack

