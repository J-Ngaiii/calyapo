import argparse
from calyapo.data_preprocessing.split_handler import SplitHandler
from calyapo.configurations.config import DATA_PATHS
from calyapo.configurations.data_map_config import TRAIN_PLANS


def main():
    parser = argparse.ArgumentParser(description="Fully runs preprocessing pipeline using the Orchestrator.") 
    parser.add_argument("train_plan", type=str, nargs='?', default='ideology_to_ideology', help="Dataset to process")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True) # Set True to see the metadata
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()
    
    plan_cfg = TRAIN_PLANS[args.train_plan]
    datasets = plan_cfg['datasets']
    
    split_handler = SplitHanlder(
        train_plan=train_plan_name, 
        train_ratio=ratios['train'], 
        val_ratio=ratios['val'], 
        test_ratio=ratios['test']
    )
    
    # This will hold the "Ratioed" packages for every dataset
    processed_packages = []

    # --- PHASE 1: Process Each Dataset ---
    for ds_name in datasets:
        print(f"\n>>> PROCESSING DATASET: {ds_name} <<<")
        
        # 1. Raw Cleaning (via RawHandler)
        raw_handler = RawHandler()
        clean_package = raw_handler.clean_dataset(dataset_name=ds_name, verbose=verbose)
        
        # 2. Split on Questions (Assigning roles to individuals)
        question_split_pack = handler.split_on_questions(package=clean_package, verbose=verbose)
        
        # 3. Apply Ratios (Reservoir/Hierarchical sampling)
        ratioed_pack = handler.split_on_ratio(package=question_split_pack, verbose=verbose)
        
        processed_packages.append(ratioed_pack)

    # --- PHASE 2: Global Combination ---
    print("\n>>> PHASE 2: GLOBAL COMBINATION <<<")
    
    # We create a Master Package to hold everything
    master_package = DataPackage(
        dataset_name=f"COMBINED_{train_plan_name}",
        train_plan=train_plan_name,
        time_period="various"
    )
    
    # Initialize master lists
    master_package['train'] = []
    master_package['val'] = []
    master_package['test'] = []

    for pkg in processed_packages:
        master_package['train'].extend(pkg['train'])
        master_package['val'].extend(pkg['val'])
        master_package['test'].extend(pkg['test'])

    # Final stringification and Llama-formatting
    final_steering_data = handler.combine_datasets(package=master_package, verbose=verbose)
    
    return final_steering_data
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