import pandas as pd
import numpy as np
from typing import List, Union
from pathlib import Path

from calyapo.configurations.config import DATA_PATHS
from calyapo.utils.persistence import *

def IGS_raw_clean(data: Union[pd.DataFrame, List[pd.DataFrame]] = None, 
                  in_path: str = None, 
                  out_path: str = None, 
                  save: bool = False, 
                  debug: bool = False, 
                  verbose: bool = False):
    """
    Main entry point for IGS cleaning. Handles both single DFs and 
    vectorized lists of DFs (e.g., from a directory load).
    """
    # try to pull from path
    if data is None:
        if in_path is None:
            raise ValueError("Must provide either 'data' or 'in_path'")
        data = file_loader(in_path=in_path, data_type='csv', verbose=verbose)
    # if not list skip and go to singular case below
    if isinstance(data, list):
        return [IGS_raw_clean(df, out_path=out_path, save=save) for df in data]
    cleaned_df = _process_single_df(data)

    if save:
        if out_path is None:
            out_path = DATA_PATHS['IGS']['intermediate'] 
        
        data_name = out_path / f"IGS_cleaned_{cleaned_df['time_period'].iloc[0]}.csv"
        
        file_saver(Path(out_path), cleaned_df, 'csv', verbose)
    
    return cleaned_df

def _process_single_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hidden helper: Performs the actual vectorized race collapse and metadata tagging.
    """
    df = df.copy()
    
    if 'time_period' not in df.columns:
        df['time_period'] = "2024" # Default or extracted logic

    RACE_MAP = {
        'Q24_1': 'White',
        'Q24_2': 'Black/African American',
        'Q24_3': 'Hispanic/Latino',
        'Q24_4': 'Asian/Asian American',
        'Q24_5': 'Native American/Alaska Native',
        'Q24_6': 'Native Hawaiian/Pacific Islander',
        'Q24_7': 'Other'
    }

    present_race_cols = [col for col in RACE_MAP.keys() if col in df.columns]
    if present_race_cols:
        labels = np.array([f"{RACE_MAP[col]}, " for col in present_race_cols])
        matrix = df[present_race_cols].fillna(0).astype(int).values
        collapsed = np.dot(matrix, labels)
        df['race'] = pd.Series(collapsed, index=df.index).str.rstrip(', ')
        df['race'] = df['race'].replace('', 'Other/Not Disclosed')
        df = df.drop(columns=present_race_cols)
        
    return df