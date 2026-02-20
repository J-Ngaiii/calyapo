
from typing import List, Dict, Any
import string
import json
from pathlib import Path
import pandas as pd

from calyapo.configurations.data_map_config import ALL_DATA_MAPS, TRAIN_PLANS
from calyapo.configurations.config import DATA_PATHS, UNIVERSAL_NA_FILLER
from calyapo.utils.persistence import *

class Orchestrator:
    def __init__(self, dataset_name: str, orchestration_plan: Dict):
        """
        Takes a orechestration_plan dict:
        {
            'num_stages' : 4, 
            'modules' : {
                mod1 : {
                    'functions' : [func_a, func_b], 
                    'params' : [param_dict_a, param_dict_b], 
                    'in_path' : "specify", 
                    'in_types' : [csv, sav], 
                    'out_path' : "specify", 
                    'out_type' : "json", 
                    'save_results' : False, 
                }, 
                mod2 : {
                    'desc' : 'Generating cleaned dataset', 
                    'functions' : [func_c, func_d], 
                    'params' : [param_dict_c, param_dict_d], 
                    'in_path' : "specify", 
                    'in_types' : [csv, sav], 
                    'out_path' : "specify", 
                    'out_type' : "json", 
                    'save_results' : False, 
                }
            }, 
            'orchestrator_metadata' : {
                plan_desc : "Calyapo version 0.2 standard orchestration plan", 
            }
        }
        """
        self.dataset_name = dataset_name
        self.num_stages = orchestration_plan['num_stages']
        self.modules = orchestration_plan['modules']
        self.metadata = orchestration_plan['plan_metadata']
        self.cleaned_data = self._generate_metadata(self.modules)

    def _generate_metadata(self, modules_dict):
        outline = []
        for mod, cfg in modules_dict.items():
            funcs_str = ", ".join([f"{f.__name__}({p})" for f, p in zip(cfg['functions'], cfg['params'])])
            outline.append(f"Module {mod} calls: {funcs_str}")
        
        self.metadata['plan_outline'] = " | ".join(outline)

    def print_metadata(self):
        """Prints a clean, formatted summary of the orchestration plan."""
        print(f"\n------ CALYAPO ORCHESTRATOR: {self.dataset_name} ------ ")
        
        print(f"{'Description:':<15} {self.metadata.get('plan_desc', 'N/A')}")
        print(f"{'Total Stages:':<15} {self.num_stages}")
        
        print("\nEXECUTION PIPELINE:")
        print("-" * 20)
        
        for step in self.metadata.get('plan_outline', []):
            print(f" • {step}")

    def _data_extracter(self, in_path: Path, data_type: str, verbose: bool = False):
        return file_loader(in_path=in_path, data_type=data_type, verbose=verbose)
        
    def _data_saver(self, data: Any, out_path: Path, data_type: str, verbose: bool = False):
        """
        Saves data to files based on type. 
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
                
        if isinstance(data, pd.DataFrame):
            data.to_csv(out_path, index=False)
        elif isinstance(data, (list, dict)):
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=4)
        elif hasattr(data, 'to_dict'):
            with open(out_path, 'w') as f:
                json.dump(data.to_dict(), f, indent=4)
        
        if verbose: print(f"  [SUCCESS] Results saved to: {out_path}")
        
    def execute(self, verbose: bool = False):
        if verbose: print(f"--- Starting Orchestration: {self.metadata.get('plan_desc', 'Unknown Orchestration Plan')} ---")
        
        curr_data = None
        for mod_name, config in self.modules.items():
            if verbose: print(f"\n[Stage: {mod_name}]")
            if config.get('in_path'):
                curr_data = self._data_extracter(Path(config['in_path']), config.get('in_type', 'csv'), verbose)
                
            for func, params in zip(config['functions'], config['params']):
                if verbose: print(f"  > Executing: {func.__name__} (In-Memory: {config.get('in_memory', True)})")
                
                if config.get('in_memory', True):
                    # standard flow: pass the object directly
                    curr_data, configs = func(curr_data, **params)
                else:
                    in_path = Path(config.get('in_path')) 
                    curr_data = self._data_extracter(in_path, config.get('in_type', 'csv'), verbose)
                    curr_data = func(curr_data, **params)

            if config.get('save_results'):
                out_path = Path(config['out_path'])
                self._data_saver(curr_data, out_path, config.get('out_type', 'csv'), verbose)
            else:
                if verbose: print("  > Stage complete. Results passed in-memory.")
                
        self.cleaned_data = curr_data
        print("\n--- Orchestration Complete ---")
            

class DataPackage:
    def __init__(self, dataset_name: str, train_plan: str, time_period: str):
        """
        DataPackage is a hashmap/dictionary wrapper that allows us to track metadata for debugging.
        
        :param self: Description
        :param dataset_name: Description
        :type dataset_name: str
        :param train_plan: Description
        :type train_plan: str
        :param time_period: Description
        :type time_period: str
        """
        self.dataset_name = dataset_name # tracked for metadata only
        self.train_plan = train_plan # tracked for metadata only
        self.time_period = time_period # tracked for metadata only
        self.data_store = {}

    # ----------------------------
    # Dictionary Emulation Methods
    # ----------------------------

    def __getitem__(self, key: str) -> Any:
        """Allows access via package['key']"""
        return self.data_store[key]

    def __setitem__(self, key: str, value: Any):
        """Allows assignment via package['key'] = value"""
        self.data_store[key] = value

    def __contains__(self, key: str) -> bool:
        """Allows checks via 'key' in package"""
        return key in self.data_store

    def __repr__(self):
        """Nice string representation for debugging"""
        return f"<DataPackage: {self.dataset_name} ({self.time_period}) keys={list(self.data_store.keys())}>"
    
    def to_dict(self) -> dict:
        """
        Flattens metadata and data_store into a single dictionary.
        
        NOTE: If a key in data_store matches a metadata field name, 
        the data_store value will overwrite the metadata.
        """
        return {
            "dataset_name": self.dataset_name,
            "train_plan": self.train_plan,
            "time_period": self.time_period,
            **self.data_store  # unpacks all key-value pairs 
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataPackage':
        """
        Creates a DataPackage instance from a flattened dictionary.
        """
        dataset_name = data.get("dataset_name", "unknown")
        train_plan = data.get("train_plan", "unknown")
        time_period = data.get("time_period", "unknown")

        instance = cls(dataset_name, train_plan, time_period) # cls represents the class itself, this is an init statement

        metadata_keys = {"dataset_name", "train_plan", "time_period"}
        for key, value in data.items():
            if key not in metadata_keys:
                instance[key] = value

        return instance

    def get(self, key: str, default=None) -> Any:
        """Safe retrieval: package.get('missing', None)"""
        return self.data_store.get(key, default)

    def keys(self):
        return self.data_store.keys()

    def values(self):
        return self.data_store.values()

    def items(self):
        return self.data_store.items()

    # ---------------
    # Legacy Wrappers
    # ---------------
    
    def add_data(self, keyword: str, data: Any):
        self[keyword] = data

    def get_data(self, keyword: str):
        return self.get(keyword)
    
class Individual:
    def __init__(self, idx, time_period, train_plan, dataset_name, na_filler = UNIVERSAL_NA_FILLER):
        if train_plan not in TRAIN_PLANS:
            raise ValueError(f"Unknown plan: {train_plan}. Check TRAIN_PLANS.")
        if dataset_name not in DATA_PATHS:
            raise ValueError(f"No path defined for {dataset_name} in DATA_PATHS")
        if 'raw' not in DATA_PATHS[dataset_name]:
            raise ValueError(f"No path for raw folder for {dataset_name} in DATA_PATHS")
        if 'processed' not in DATA_PATHS[dataset_name]:
            raise ValueError(f"No path for processed folder for {dataset_name} in DATA_PATHS")
        if dataset_name not in ALL_DATA_MAPS:
            raise ValueError(f"Unknown dataset {dataset_name} in ALL_DATA_MAPS")
        
        self.id = idx
        self.time_period = time_period
        self.dataset_name = dataset_name
        self.train_plan = train_plan # tracked for metadata only
        self.demog = {}
        self.question_map = {
                            "train" : {
                                "var_label2qst_text": {}, # we keep track of the variable label (`ideology`), question text and question response
                                "var_label2qst_choices": {}, # choices are what you can choose
                                "var_label2qst_option": {} # option is what's selected by the respondent
                            }, 
                            "val" : {
                                "var_label2qst_text": {},
                                "var_label2qst_choices": {}, 
                                "var_label2qst_option": {}
                            }, 
                            "test" : {
                                "var_label2qst_text": {}, 
                                "var_label2qst_choices": {}, 
                                "var_label2qst_option": {}
                            }, 
                        }
        
        self.na_filler = na_filler
        
        # setting up question text and variable mappers NOT invariant across time
        self.dataset_maps = ALL_DATA_MAPS[dataset_name].get(time_period)
        self.label2qes = self.dataset_maps.get('label2qes', {})
        self.label2opt = self.dataset_maps.get('label2opt', {})
        self.var2label = self.dataset_maps.get('var2label', {})
        self.label2var = {v: k for k, v in self.var2label.items()} # label2var = {'age': 'Q21', 'ideology': 'Q27'}
        self.opt_decoders = self._get_dataset_decoders() # opt_decoders = { 'partyid': {1: 'Democrat', ...}, 'age': {1: '18-29', ...} }
    
    # ---------------
    # PRIVATE METHODS
    # ---------------

    def _invert_mapping(self, opt_map: Dict[str, Any]) -> Dict[int, str]:
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
    
    def _get_dataset_decoders(self) -> Dict[str, Dict[int, str]]:
        """
        Generates a dictionary of decoders for a specific dataset.
        Returns: {<variable_label> : <inverted mapping>}
        - eg. { 'partyid': {1: 'Democrat', ...}, 'age': {1: '18-29', ...} }
        """
        decoders = {}
        for variable_label, mapping in self.label2opt.items():
            if mapping: # only invert if mapping exists
                decoders[variable_label] = self._invert_mapping(mapping)
            
        return decoders
    
    def _process_response(self, variable_label: str, raw_val: Any, debug_var_label = None):
        """Helper to decode a raw value into text."""
        # debug this
        raw_str = str(raw_val).strip()

        if variable_label == debug_var_label:
            print(f"raw val: ", raw_str)
            print(f"var label: ", variable_label)
            print(f"decoder: ", self.opt_decoders)

            print("label existence: ", variable_label in self.opt_decoders)
            print("raw value existence: ", raw_str in self.opt_decoders[variable_label])

        if variable_label in self.opt_decoders and raw_str in self.opt_decoders[variable_label]:
            return self.opt_decoders[variable_label][raw_str]
        
        try:
            raw_int = int(float(raw_str)) # Handle "1.0" -> 1
            if variable_label in self.opt_decoders and raw_int in self.opt_decoders[variable_label]:
                return self.opt_decoders[variable_label][raw_int]
        except ValueError:
            pass

        return self.na_filler
    
    def _get_choices_string(self, variable_label: str, raw_val: str) -> str:
        """
        Generates the 'A. Option1\nB. Option2' string for the prompt.
        """
        result = {
            "formatted": "", 
            "option_text": self.na_filler, 
            "option_letter": self.na_filler
        }
        
        if variable_label not in self.label2opt:
            return result

        # check option text first
        user_response_text = self._process_response(variable_label, raw_val)
        result['option_text'] = user_response_text
        if user_response_text == self.na_filler:
            return result

        # then build out full options map
        options_map = self.label2opt[variable_label]
        choices_list = list(options_map.keys())
        letters = string.ascii_uppercase
        
        formatted_lines = []
        
        for i, choice_text in enumerate(choices_list):
            if i >= len(letters): break
            
            letter = letters[i]
            formatted_lines.append(f"{letter}. {choice_text}")
            
            # save the letter if the option matches 
            if str(user_response_text).strip() == str(choice_text).strip():
                result['option_letter'] = letter

        result['formatted'] = "\n".join(formatted_lines)
        return result
    
    def _add_question_data(self, split: str, variable_label: str, raw_val: Any):
        """Generic method for train/val/test."""
        question_text = self.label2qes.get(variable_label, "Missing Question Text")
        answer_dict: Dict = self._get_choices_string(variable_label, raw_val)

        self.question_map[split]['var_label2qst_text'][variable_label] = str(question_text)
        self.question_map[split]['var_label2qst_choices'][variable_label] = str(answer_dict.get('formatted'))
        self.question_map[split]['var_label2qst_option'][variable_label] = {
            'option_letter' : str(answer_dict.get('option_letter')), 
            'option_text' : str(answer_dict.get('option_text'))
        }

    # ---------------
    # PUBLIC METHODS
    # ---------------

    def add_demog(self, variable_label, raw_val, debug=False):
        decoded = self._process_response(variable_label, raw_val)
        if debug:
            if variable_label == 'partyid':
                print(f"decoded val: ", decoded)
        self.demog[variable_label] = str(decoded)

    def add_train(self, variable_label, raw_val):
        self._add_question_data('train', variable_label, raw_val)

    def add_val(self, variable_label, raw_val):
        self._add_question_data('val', variable_label, raw_val)
        
    def add_test(self, variable_label, raw_val):
        self._add_question_data('test', variable_label, raw_val)

    def return_split_indiv_map(self, split):
        if split == 'full':
            return self.return_full_indiv_map()
        elif split == 'train':
            return self.return_train_indiv_map()
        elif split == 'val':
            return self.return_val_indiv_map()
        elif split == 'test':
            return self.return_test_indiv_map()

    def return_full_indiv_map(self):
        entry = {
                "id" : self.id, # default to row index, can use igs id or can just not
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "train" : self.question_map['train'],
                "val" : self.question_map['val'],
                "test" : self.question_map['test'],
            }
        return entry
    
    def return_train_indiv_map(self):
        entry = {
                "id" : self.id, 
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "train" : self.question_map['train']
            }
        return entry
    
    def return_val_indiv_map(self):
        entry = {
                "id" : self.id, 
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "val" : self.question_map['val'],
            }
        return entry
    
    def return_test_indiv_map(self):
        entry = {
                "id" : self.id, 
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "test" : self.question_map['test'],
            }
        return entry
    
class TrainPlanWrapper:
    def __init__(self, dataset_name, train_plan):
        if train_plan not in TRAIN_PLANS:
            raise ValueError(f"Unknown plan: {train_plan}. Check TRAIN_PLANS.")
        self.plan_config = TRAIN_PLANS[train_plan]
        self.dataset_name = dataset_name # tracked for metadata only
        self.train_plan = train_plan # tracked for metadata only

        self.variable_map = self.plan_config.get('variable_map') # variable_map = {'demo' : ['age', 'partyid'], 'train_resp' : ['ideology'], 'val_resp' : ['trump_opinion']}

    def get_var_lst(self, split: str):
        """
        Docstring for get_var_lst
        
        :param self: Description
        :param split: Description
        :type split: str

        Handles for:
            demo_vars = self.variable_map['demo'] # ['age', 'partyid']
            train_resp_vars = self.variable_map['train_resp'] # ['ideology']
            val_resp_vars = self.variable_map['val_resp'] # ['trump_opinion']
           test_resp_vars = self.variable_map['test_resp'] # ['abortion_senate']
        """
        if split not in self.variable_map:
            raise ValueError(f"Unknown split arg: {split}. Choose from: {self.dataset_plan.keys()}.")
        
        return self.variable_map[split]