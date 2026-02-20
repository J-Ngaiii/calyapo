import numpy as np
import pandas as pd
import json
from pathlib import Path
from calyapo.data_preprocessing.cleaning_objects import DataPackage

def file_loader(in_path: Path, data_type: str, verbose: bool = False):
        """
        Handles extracting data. Returns a list of data objects if given a 
        directory, or a single data object if given a file.
        """
        if not in_path.exists():
            raise FileNotFoundError(f"Path not found: {in_path}")

        if in_path.is_dir():
            if verbose: print(f"Scanning directory for *.{data_type} files...")
            target_files = list(in_path.glob(f"*.{data_type}"))
        else:
            target_files = [in_path]

        results = []
        for file_path in target_files:
            if verbose: print(f"Loading file: {file_path.name}")
            
            if data_type == 'csv':
                results.append(pd.read_csv(file_path))
            elif data_type == 'json':
                with open(file_path, 'r') as f:
                    results.append(json.load(f))
            elif data_type == 'DataPackage':
                 with open(file_path, 'r') as f:
                    results.append(DataPackage.from_dict(json.load(f)))

        return results[0] if len(results) == 1 else results