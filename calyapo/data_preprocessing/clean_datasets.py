import json
import pandas as pd
from pathlib import Path

from calyapo.configurations.data_map_config import ALL_DATA_MAPS, TRAIN_PLANS
from calyapo.configurations.config import DATA_PATHS, UNIVERSAL_NA_FILLER
from calyapo.data_preprocessing.cleaning_objects import Individual, TrainPlanWrapper, DataPackage

def process_csv(csv_path: Path, dataset_name: str, train_plan: str) -> DataPackage:
    """
    Process a single CSV file and return a DataPackage.
    """
    # Identify time period
    time_period = csv_path.stem.split('_')[-1] # e.g. '2024'
    
    # Validation
    if time_period not in ALL_DATA_MAPS.get(dataset_name, {}):
        print(f"  Warning: No config map for {time_period}. Skipping.")
        return None

    df = pd.read_csv(csv_path)
    
    # Initialize Wrappers
    tp_wrap = TrainPlanWrapper(dataset_name, train_plan)
    dataset_maps = ALL_DATA_MAPS[dataset_name][time_period]
    var2label = dataset_maps.get('var2label', {})
    label2var = {v: k for k, v in var2label.items()}
    
    # Containers
    cleaned_data = []
    train_data = []
    val_data = []
    test_data = []

    # Iterate Rows
    for idx, row in df.iterrows():
        # ID Logic
        id_col = label2var.get('dataset_id')
        indiv_id = row[id_col] if (id_col and id_col in row) else idx
        
        # Create Individual
        entry = Individual(indiv_id, time_period, train_plan, dataset_name)

        # 1. Demographics
        for var_label in tp_wrap.get_var_lst('demo'):
            csv_col = label2var.get(var_label)
            if csv_col and csv_col in row:
                entry.add_demog(var_label, row[csv_col])
            else:
                entry.add_demog(var_label, UNIVERSAL_NA_FILLER)

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

        # Store
        cleaned_data.append(entry.return_full_individual())
        train_data.append(entry.return_split_individual('train'))
        val_data.append(entry.return_split_individual('val'))
        test_data.append(entry.return_split_individual('test'))

    # Package results
    pack = DataPackage(dataset_name, train_plan, time_period)
    pack.add_data('full', cleaned_data)
    pack.add_data('train', train_data)
    pack.add_data('val', val_data)
    pack.add_data('test', test_data)
    
    return pack


def build_steering_dataset(dataset_name: str, train_plan: str = "ideology_to_trump", save: bool = True):
    """
    Main Driver: Iterates through all CSVs for a dataset, cleans them, saves intermediates,
    and returns a combined DataPackage.
    """
    # Validation
    if train_plan not in TRAIN_PLANS:
        raise ValueError(f"Unknown plan: {train_plan}")
    if dataset_name not in DATA_PATHS:
        raise ValueError(f"No path for {dataset_name}")
    
    raw_dir = DATA_PATHS[dataset_name]['raw']
    csv_files = list(raw_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}")
    print(f"Found {len(csv_files)} files for {dataset_name}.")

    # Master Containers
    master_full = []
    master_train = []
    master_val = []
    master_test = []

    for csv_path in csv_files:
        # Process individual file
        pack = process_csv(csv_path, dataset_name, train_plan)
        
        if not pack: continue # Skip if config missing
        
        # Aggregate
        full_data = pack.get_data('full')
        master_full.extend(full_data)
        master_train.extend(pack.get_data('train'))
        master_val.extend(pack.get_data('val'))
        master_test.extend(pack.get_data('test'))

        # Save Intermediate
        if save:
            output_dir = DATA_PATHS[dataset_name]['processed']
            output_dir.mkdir(parents=True, exist_ok=True)
            # e.g. ideology_to_trump_IGS_2024_processed.json
            out_name = f"{train_plan}_{dataset_name}_{pack.time_period}_processed.json"
            
            with open(output_dir / out_name, 'w') as f:
                json.dump(full_data, f, indent=2)
            print(f"  Saved {len(full_data)} rows to {out_name}")

    # Return Combined Package
    master_pack = DataPackage(dataset_name, train_plan, "all_combined")
    master_pack.add_data('full', master_full)
    master_pack.add_data('train', master_train)
    master_pack.add_data('val', master_val)
    master_pack.add_data('test', master_test)
    
    return master_pack


if __name__ == "__main__":
    build_steering_dataset('IGS', 'ideology_to_trump')