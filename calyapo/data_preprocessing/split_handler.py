from typing import List, Dict, Any
from collections import defaultdict

from calyapo.data_preprocessing.funcs.raw_cleaners import * 
from calyapo.data_preprocessing.funcs.clean_datasets import *
from calyapo.data_preprocessing.funcs.data_combiner import *
from calyapo.data_preprocessing.funcs.ratioed import *
from calyapo.data_preprocessing.raw_handler import RawHandler
from calyapo.configurations.data_map_config import TRAIN_PLANS
from calyapo.configurations.config import UNIVERSAL_PENULTIMATE_FOLDER
from calyapo.utils.persistence import *

class SplitHandler:
    """
    Splits data into train, test, val based on idx as well as based on question. Combines data from different datasets.
    Handles balancing passing in memory vs pulling from paths so cleaning funcs don't have to.
    """
    NAME = 'Split Handler'

    def __init__(self, train_plan: str, train_ratio: float = None, val_ratio: float = None, test_ratio: float = None):
        
        
        self.train_plan = train_plan
        self.plan_config = TRAIN_PLANS[train_plan]
        self.variable_map = self.plan_config['variable_map']
        self.homogenous_plan = self.plan_config['homogenous_var_plan']
        self.ques_split_varying = self.plan_config['question_varies_by_split']
        self.datasets = self.plan_config['datasets']
        
        self.training_ratios = {
            'train' : float(train_ratio), 
            'val' : float(val_ratio), 
            'test' : float(test_ratio)
        }

    def split_on_questions(self, package: DataPackage = None, dataset_name: str = None, save: bool = False, debug: bool = False, verbose: bool = False):
        """
        Splits data based on specific questions. 
        Doesn't drop NaN values. 
        Handles for automatic file path checking if package is none.
        Connects to RawHandler's clean_dataset.
        """
        if package is None or package['data'] is None:
            assert dataset_name is not None, f"(Split Handler | Splitting) cannot file pull if no dataset_name given"
            
            interim_in_path = DATA_PATHS[dataset_name]['intermediate']
            all_in_paths = list(Path(interim_in_path).glob('*.csv'))
            if all_in_paths is None:
                # directly call clean funcs
                rawHand = RawHandler(special_cond = 'fallback handler created in Split Handler Splitting logic')
                tempPack = rawHand.clean_dataset(dataset_name=dataset_name)
                interim_data = tempPack['data']
            else:
                # rely on default path pull
                interim_data = [file_loader(in_path=path, data_type='csv', debug=debug, verbose=verbose) for path in all_in_paths]
            if verbose: print(f"(Split Handler | Splitting) Found {len(all_in_paths)} files for {dataset_name}.")
        else:
            # use data passed in-memory
            interim_data = package['data']
            dataset_name = package.dataset_name
            if verbose: print(f"(Split Handler | Splitting) Processing {len(interim_data)} DataFrames passed in-memory.")
        
        out_path = DATA_PATHS[dataset_name]['processed']
        out_pack = split_questions(data=interim_data, dataset_name=dataset_name, train_plan=self.train_plan, out_path=out_path, save=save, debug=debug, verbose=verbose)
        # train plan is written onto out_pack
        return out_pack
    
    def split_on_ratio(self, package: DataPackage = None, dataset_name: str = None, save: bool = False, debug: bool = False, verbose: bool = False):
        """
        Splits data based on ratio. 
        Connects to RawHandler's clean_dataset.
        """
        if package is None or package['data'] is None:
            # default path pull
            processed_dir = DATA_PATHS[dataset_name]['processed']    
            target_json = processed_dir / f"{self.train_plan}_{dataset_name}_fullpack_processed.json"
            
            if target_json.exists():
                if verbose: print(f"(Split Handler | Ratioing) Loading existing package: {target_json.name}")
                raw_json = file_loader(in_path=target_json, data_type='json', verbose=verbose)
                package = DataPackage.from_dict(raw_json) 
            else:
                # if we cannot pull from path generate from scratch
                if verbose: print(f"(Split Handler | Ratioing) No processed data found. Building steering dataset for {dataset_name}...")
                package = self._split_on_questions(dataset_name=dataset_name, train_plan=self.train_plan, save=save, debug=debug, verbose=verbose)
        
        out_path = UNIVERSAL_PENULTIMATE_FOLDER
        out_pack = split_ratio(
            package=package, 
            target_ratios=self.training_ratios, 
            homogenous_plan=self.homogenous_plan, 
            ques_split_varying=self.ques_split_varying, 
            out_path=out_path, 
            save=save, 
            debug=debug, 
            verbose=verbose
        )

        return out_pack
    
    def combine_datasets(self, package: DataPackage = None, dataset_name: str = None, save: bool = False, debug: bool = False, verbose: bool = False):
        """
        Combines splitted up data into one massive dataset.
        Does drop NaN values. 
        Handles for automatic file path checking if package is none.
        """
        def valid_combine_pack(package, verbose: bool = False):

            for field in set(['train', 'val', 'test']):
                if field not in package:
                    if verbose: f"(Split Handler | Combining) field {field} missing from inputted package {package}"
                    return False
            return True

        if package is None or valid_combine_pack(package, verbose):
            # default path pull
            processed_dir = DATA_PATHS[dataset_name]['processed']    
            target_json = processed_dir / f"{self.train_plan}_{dataset_name}_fullpack_processed.json"
            
            if target_json.exists():
                if verbose: print(f"(Split Handler | Combining) Loading existing package: {target_json.name}")
                raw_json = file_loader(in_path=target_json, data_type='json', verbose=verbose)
                package = DataPackage.from_dict(raw_json) 
            else:
                # if we cannot pull from path generate from scratch
                if verbose: print(f"(Split Handler | Combining) No processed data found. Building steering dataset for {dataset_name}...")
                package = self._split_on_questions(dataset_name=dataset_name, train_plan=self.train_plan, save=save, debug=debug, verbose=verbose)
        else:
            if verbose: print(f"(Split Handler | Combining) Processing {len(package['full'])} json-ized individuals passed in-memory.")

        out_path = UNIVERSAL_FINAL_FOLDER
        out_dict = split_combine(package=package, out_path=out_path, save=save, debug=debug, verbose=verbose)      
        return out_dict