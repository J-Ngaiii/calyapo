import numpy as np
import pandas as pd
import re
import json
from pathlib import Path
from typing import Any, List, Tuple
from calyapo.data_preprocessing.cleaning_objects import DataPackage

def file_loader(
            in_path: Path, 
            data_type: str, 
            path_extract: str = None, 
            always_return_lst: bool = False, 
            debug: bool = False, 
            verbose: bool = False
        ) -> Tuple[List[pd.DataFrame], List[str]]:
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

        data = []
        path_extractions = []
        for file_path in target_files:
            if verbose: print(f"(File Loader) Loading {file_path.name}...")
            
            if data_type == 'csv':
                data.append(pd.read_csv(file_path))
            elif data_type == 'json':
                with open(file_path, 'r') as f:
                    data.append(json.load(f))
            elif data_type == 'DataPackage':
                 with open(file_path, 'r') as f:
                    data.append(DataPackage.from_dict(json.load(f)))

            if path_extract is not None:
                if debug:
                    print(f"(File Loader | Debug) scanning file path: {file_path}")
                match = re.search(str(path_extract), str(file_path))
                if match:
                    extracted_text = match.group(1)
                    path_extractions.append(extracted_text)
                    if verbose: print(f"(File Loader) extracted text '{extracted_text}' from path: {extracted_text} via pattern {path_extract}")
                else:
                    if verbose: print(f"(File Loader) could not find regex match via pattern {path_extract}")

        if data == []:
            raise ValueError(f"(File Loader) inputted path {in_path} legitimate but no files loaded from path")
        if path_extractions == [] and path_extract is not None:
            raise ValueError(f"(File Loader) inputted path {in_path} legitimate but no match via pattern {path_extract} found")
        
        if path_extract is not None:
            if len(data) > 1 or always_return_lst:
                return data, path_extractions
            elif len(data) == 1 and not always_return_lst:
                return data[0], path_extractions[0]
        else:
            if len(data) > 1 or always_return_lst:
                return data
            elif len(data) == 1 and not always_return_lst:
                return data[0]
            

def file_saver(out_path: Path, data: Any, data_type: str, indnt: int = 4, verbose: bool = False):
    """
    Saves data to files based on type. 
    """
    if isinstance(out_path, str):
        out_path = Path(out_path)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
            
    if isinstance(data, pd.DataFrame) or data_type == "csv":
        data.to_csv(out_path, index=False)
    elif isinstance(data, (list, dict)) or data_type == "json":
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=indnt)
    elif isinstance(data, pd.DataFrame) or data_type == "DataPackage":
        with open(out_path, 'w') as f:
            json.dump(data.to_dict(debug=True), f, indent=indnt)
    elif hasattr(data, 'to_dict'):
        with open(out_path, 'w') as f:
            json.dump(data.to_dict(), f, indent=indnt)
    
    if verbose: print(f"(File Saver) Results saved to: {out_path}")