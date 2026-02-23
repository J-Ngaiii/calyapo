from typing import List, Dict, Any
from pathlib import Path

from calyapo.utils.persistence import *

class Orchestrator:
    def __init__(self, dataset_name: str, orchestration_plan: Dict):
        """
        Takes a orechestration_plan dict:
        { 
            'modules' : {
                mod1 : {
                    'functions' : [func_a, func_b], 
                    'params' : [param_dict_a, param_dict_b], 
                    'mod_meta' : {
                        'is_data_validator_mod' : False, 
                        'is_cleaning_mod' : True, 
                        'is_splitting_mod' : False
                    }, 
                    'persistence' : {
                        'in_path' : "specify", 
                        'in_types' : [csv, sav], # can specify here but will check params first
                        'out_path' : "specify", 
                        'out_type' : "json", 
                        'in_memory' : True
                        'save_results' : False, 
                    }   
                }, 
                ...
                <remaining modules here>
                ...
            }, 
            'meta' : {
                'num_stages' : 4,
                'plan_desc' : "Calyapo version 0.2 standard orchestration plan", 
            }
        }
        """
        # Metadata fields
        self.dataset_name = dataset_name
        self.num_stages = orchestration_plan['meta']['num_stages']
        self.meta_desc = orchestration_plan['meta']['plan_desc']
        self.orchestrator_desc = self._generate_metadata(self.modules)

        # Module fields
        self.modules = orchestration_plan['modules']

    # --------------
    # Metadata funcs
    # --------------
    def _generate_metadata(self, modules_dict):
        outline = []
        for mod, cfg in modules_dict.items():
            funcs_str = ", ".join([f"{f.__name__}({p})" for f, p in zip(cfg['functions'], cfg['params'])])
            outline.append(f"Module {mod} calls: {funcs_str}")
        
        msg = " | ".join(outline)
        return msg

    def print_metadata(self):
        """Prints a clean, formatted summary of the orchestration plan."""
        print(f"\n------ CALYAPO ORCHESTRATOR: {self.dataset_name} ------ ")
        
        print(f"{'Description:':<15} {self.meta_desc}")
        print(f"{'Total Stages:':<15} {self.num_stages}")
        
        print("\nEXECUTION PIPELINE:")
        print("-" * 20)
        
        for step in self.orchestrator_desc:
            print(f" • {step}")

    # --------------
    # Persistence funcs
    # --------------
    def _data_extracter(self, in_path: Path, data_type: str, verbose: bool = False):
        return file_loader(in_path=in_path, data_type=data_type, verbose=verbose)
        
    def _data_saver(self, data: Any, out_path: Path, data_type: str, verbose: bool = False):
        file_saver(out_path=out_path, data=data, data_type=data_type, verbose=verbose)
        
    # --------------
    # Main Loop
    # --------------
    def execute(self, verbose: bool = False):
        if verbose: print(f"--- Starting Orchestration: {self.metadata.get('plan_desc', 'Unknown Orchestration Plan')} ---")
        
        
        curr_data = None
        for mod_name, config in self.modules.items():
            if verbose: print(f"\n[Stage: {mod_name}]")

            persistence_conf = config['persistence']

            # deprecate file pulling, consolidate it inside the handlers
            if persistence_conf['in_path'] is not None and persistence_conf['in_memory'] == False:
                curr_data = self._data_extracter(Path(persistence_conf['in_path']), persistence_conf.get('in_type', 'csv'), verbose)
                
            counter = 1
            for func, params in zip(config['functions'], config['params']):
                if verbose: print(f"|>>({counter})>>| Executing: {func.__name__} (In-Memory: {persistence_conf.get('in_memory', True)})")
                
                if persistence_conf.get('in_memory', True) == True:
                    # standard flow: pass the object directly
                    # future implmentation: just make the funcs pass data packages
                    # that way we can wrap output meta that later funcs might need and data together
                    # all in one output
                    curr_data = func(curr_data, **params)
                else:
                    in_path = Path(persistence_conf['in_path']) 
                    curr_data = self._data_extracter(in_path, persistence_conf.get('in_type', 'csv'), verbose)
                    curr_data = func(curr_data, **params)

                counter += 1

            if persistence_conf['save_results'] == True: # write out for clearer code
                out_path = Path(persistence_conf['out_path'])
                self._data_saver(curr_data, out_path, persistence_conf.get('out_type', 'csv'), verbose)
            else:
                if verbose: print(f"|:D| Stage complete. Results passed in-memory.")
                
        self.cleaned_data = curr_data
        print("\n--- Orchestration Complete ---")