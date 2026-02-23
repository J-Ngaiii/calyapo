import pandas as pd
import numpy as np
from typing import List, Union
from pathlib import Path

from calyapo.configurations.config import DATA_PATHS, UNIVERSAL_NA_FILLER
from calyapo.data_preprocessing.cleaning_objects import DataPackage
from calyapo.utils.persistence import *

def IGS_raw_clean(
        dataPackage: DataPackage, 
        out_path: str = None, 
        save: bool = False, 
        debug: bool = False, 
        verbose: bool = False
    ) -> List[pd.DataFrame]:
    """
    Main entry point for IGS cleaning. Handles both single DFs and 
    vectorized lists of DFs (e.g., from a directory load).
    """
    
    data = dataPackage['data']
    time_periods = dataPackage['time_periods']

    if debug:
        print(f"(IGS_raw | Debug) Data <type '{type(data)}'>:{data}\n(IGS_raw | Debug) Time Periods <type '{type(time_periods)}'>:{time_periods}")

    output = []
    for df, period in zip(data, time_periods):
        cleaned_df = _process_single_df(df, period, debug)
        output.append(cleaned_df)
        if save:
            assert out_path is not None, f"(IGS_raw) Cannot save file without valid out path"
                
            data_name = out_path / f"IGS_cleaned_{cleaned_df['time_period'].iloc[0]}.csv"
            
            file_saver(Path(data_name), cleaned_df, 'csv', verbose=verbose)
    
    return output

def _process_single_df(df: pd.DataFrame, period: str, debug: bool = False) -> pd.DataFrame:
    """
    Hidden helper: Performs the actual vectorized race collapse and metadata tagging.
    """
    df = df.copy()
    
    if 'time_period' not in df.columns:
        df['time_period'] = period

    RACE_MAP = {
        'Q24_1': '1',
        'Q24_2': '2',
        'Q24_3': '3',
        'Q24_4': '4',
        'Q24_5': '5',
        'Q24_6': '6',
        'Q24_7': '7'
    }

    present_race_cols = [col for col in RACE_MAP.keys() if col in df.columns]
    if present_race_cols:
        encodings = np.array([RACE_MAP[col] for col in present_race_cols])
        matrix = (
            df[present_race_cols]
            .replace(r'^\s*$', np.nan, regex=True) # replaces empty or whitespace strings
            .fillna(0)
            .astype(int)
            .values
        )
        if debug:
            print(f"(process_single | Debug) matrix datatype: {matrix.dtype}\n(process_single | Debug) labels datatype: {encodings.dtype}")
        temp_df = pd.DataFrame(matrix, columns=encodings)
        collapsed = temp_df.idxmax(axis=1)
        row_sums = matrix.sum(axis=1)
        collapsed[row_sums == 0] = UNIVERSAL_NA_FILLER
        if debug:
            print(f"(process_single | Debug) collapsed vals:\n{collapsed}")
        df['racial_id'] = collapsed.values
        if debug:
            print(f"(process_single | Debug) final df race col and time col:\n{df[['racial_id', 'time_period']]}")
    return df