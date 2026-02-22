import numpy as np
import pandas as pd
import re
import json
from pathlib import Path
from typing import Any, Dict
from calyapo.data_preprocessing.cleaning_objects import DataPackage

def file_loader(in_path: Path, data_type: str, path_extract: str = None, verbose: bool = False):
        """
        Handles extracting data. Returns a list of data objects if given a 
        directory, or a single data object if given a file.
        """
        if not in_path.exists():
            raise FileNotFoundError(f"(File Loader) Inputted path '{in_path}' not found")

        if isinstance(in_path, str):
            in_path = Path(in_path)

        if in_path.is_dir():
            if verbose: print(f"(File Loader) Scanning directory for *.{data_type} files...")
            target_files = list(in_path.glob(f"*.{data_type}"))
        else:
            target_files = [in_path]

        data_files = []
        path_extractions = []
        for file_path in target_files:
            if verbose: print(f"(File Loader) Loading {file_path.name}...")
            
            if data_type == 'csv':
                data_files.append(pd.read_csv(file_path))
            elif data_type == 'json':
                with open(file_path, 'r') as f:
                    data_files.append(json.load(f))
            elif data_type == 'DataPackage':
                 with open(file_path, 'r') as f:
                    data_files.append(DataPackage.from_dict(json.load(f)))

            if path_extract is not None:
                match = re.search(path_extract, file_path)
                if match:
                    extracted_text = match.group(1)
                    if verbose: print(f"(File Loader) extracted text from path: {extracted_text} via pattern {path_extract}")
                else:
                    if verbose: print(f"(File Loader) could not find regex match via pattern {path_extract}")

        if len(data_files) == 1:
            output = data_files[0]
            outextract = path_extractions[0]
        else:
            output = data_files
            outextract = path_extractions
        return output, outextract

def file_saver(out_path: Path, data: Any, data_type: str, indnt: int = 4, verbose: bool = False):
    """
    Saves data to files based on type. 
    """
    if isinstance(out_path, str):
        out_path = Path(out_path)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
            raise FileNotFoundError(f"(File Saver) Inputted path '{out_path}' not found")
            
    if isinstance(data, pd.DataFrame):
        data.to_csv(out_path, index=False)
    elif isinstance(data, (list, dict)):
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=indnt)
    elif hasattr(data, 'to_dict'):
        with open(out_path, 'w') as f:
            json.dump(data.to_dict(), f, indent=indnt)
    
    if verbose: print(f"(File Saver) Results saved to: {out_path}")