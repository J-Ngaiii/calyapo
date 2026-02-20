import argparse
from calyapo.data_preprocessing.cleaning_objects import Orchestrator
from calyapo.data_preprocessing.raw_cleaners import * 
from calyapo.data_preprocessing.clean_datasets import *
from calyapo.data_preprocessing.data_combiner import *
from calyapo.configurations.config import DATA_PATHS

IGS_orchestration_plan = {
        'num_stages': 3,
        'modules': {
            'mod1': {
                'functions': [IGS_raw_clean], # NOT VECTORIZED
                'params': [{
                    'in_path': DATA_PATHS['IGS']['raw'], 
                    'out_path': DATA_PATHS['IGS']['intermediate']
                    }],
                
                'in_path': 'calyapo/data/raw/igs/', # Vectorized directory scan
                'in_type': 'csv',
                'out_path': 'calyapo/data/raw/intermediate/', # Vectorized directory scan
                'out_type': 'csv',

                'vectorize' : False, 
                'in_memory' : True, 
                'save_results': True
            },
            'mod2': {
                'functions': [build_steering_dataset],
                'params': [{
                    
                    }],
                
                'in_path': 'calyapo/data/raw/intermediate/', # Vectorized directory scan
                'type': 'csv',
                'out_path': 'calyapo/data/raw/intermediate/', # Vectorized directory scan
                'out_type': 'csv',

                'vectorize' : False, 
                'in_memory' : True, 
                'save_results': True
            },
            'mod3': {
                'functions': [split_combine],
                'params': [{
                    
                    }],
                 
                'in_path': 'calyapo/data/raw/igs/', # Vectorized directory scan
                'type': 'csv',
                
                'vectorize' : False, 
                'in_memory' : True, 
                'save_results': True
            },
        },
        'metadata': {
            'plan_desc': "IGS 2024",
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Fully runs preprocessing pipeline using the Orchestrator.") 
    parser.add_argument("dataset", type=str, nargs='?', default='IGS', help="Dataset to process")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True) # Set True to see the metadata
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()
    plan = get_igs_plan(save_results=args.save)

    # 2. Initialize Orchestrator
    orch = PreProcessOrchestrator(dataset_name=args.dataset, orchestration_plan=plan)
    
    # 3. Print Metadata summary before running
    if args.verbose or args.debug:
        orch.print_metadata()

    # 4. Execute
    try:
        orch.execute(verbose=args.verbose)
    except Exception as e:
        if args.debug:
            raise e
        print(f"[ERROR] Orchestration failed: {e}")

if __name__ == "__main__":
    main()