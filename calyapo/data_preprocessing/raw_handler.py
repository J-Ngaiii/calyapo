import pandas as pd

from calyapo.configurations.data_map_config import TRAIN_PLANS
from calyapo.data_preprocessing.funcs.raw_cleaners import *
from calyapo.data_preprocessing.cleaning_objects import DataPackage
from calyapo.utils.persistence import *

class RawHandler: 
    """
    Balances and coordinates pulling and cleaning raw csvs. 
    Handles balancing passing in memory vs pulling from paths
    """
    NAME = 'Raw Handler'

    def __init__(self, special_cond: str = None):
        self.special_cond = special_cond

    def clean_dataset(
            self, 
            dataset_name: str, 
            in_path: str = None, 
            out_path: str = None, 
            save: bool = False, 
            debug: bool = False, 
            verbose: bool = False
        ):

        if in_path is None:
            # default path pull if no in_path
            in_path = DATA_PATHS[dataset_name]['raw']
        if out_path is None:
            out_path = DATA_PATHS[dataset_name]['intermediate']

        end_of_str_time_pat = '_([^_]+).csv'
        data, time_periods = file_loader(in_path=in_path, data_type='csv', path_extract=end_of_str_time_pat, always_return_lst=True, debug=debug, verbose=verbose)
        inpack = DataPackage(
            dataset_name=dataset_name, 
            train_plan='N/A, this is a raw cleaning inpack', 
            time_period='multiple, this is a raw cleaning inpack', 
        )
        inpack['data'] = data
        inpack['time_periods'] = time_periods

        raw_cleaned_dfs: List[pd.DataFrame] = IGS_raw_clean(dataPackage=inpack, out_path=out_path, save=save, debug=debug, verbose=verbose)
        outpack = DataPackage(
            dataset_name=dataset_name, 
            train_plan='N/A, this is a raw cleaning outpack', 
            time_period='multiple, this is a raw cleaning outpack', 
        )
        outpack['data'] = raw_cleaned_dfs
        return outpack