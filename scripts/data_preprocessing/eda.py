import pandas as pd
import argparse
from pathlib import Path
from calyapo.configurations.data_map_config import ALL_DATA_MAPS
from calyapo.configurations.config import DATA_PATHS

def check_data_coverage(dataset_name, year):
    """
    Finds supported variable_labels for a dataset/year and calculates NA percentages.
    """
    try:
        dataset_map = ALL_DATA_MAPS[dataset_name][year]
        var2label = dataset_map['var2label']
    except KeyError:
        print(f"No maps found for {dataset_name}-{year}.")
        return
    raw_path = DATA_PATHS[dataset_name]['raw']
    
    files = list(raw_path.glob("*.csv"))

    if not files:
        print(f"No CSV files found in path: {raw_path}")
        return
    
    df = pd.read_csv(files[0])
    
    print(f"\n------ Coverage Report: {dataset_name}-{year} ------")
    print(f"{'Variable':<15} | {'Label':<20} | {'% NA':<10}")

    supported_vars = var2label.keys()
    
    for var in supported_vars:
        label = var2label[var] # abortion_senate
        if var in df.columns: 
            na_percent = df[var].isnull().mean() * 100
            print(f"{var:<15} | {label:<20} | {na_percent:>8.2f}%")
        else:
            print(f"{var:<15} | {label:<20} | {'NA':>10}")

def main():
    parser = argparse.ArgumentParser(description="Check NA percentages for raw datasets.")
    parser.add_argument("dataset_name", type=str, nargs='?', default='IGS', 
                        help="Name of the dataset.")
    parser.add_argument("time_period", type=str, nargs='?', default='2024', 
                        help="Time period of the dataset.")
    args = parser.parse_args()

    check_data_coverage(args.dataset_name, args.time_period)

if __name__ == "__main__":
    main()