import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER
from calyapo.utils import file_saver

class Tabularizer:
    def __init__(self, train_plan: str, keyword: str, root_path: str = ".", debug: bool = False, verbose = False):
        self.train_plan = train_plan
        self.keyword = keyword
        self.root = Path(root_path)
        
        self.debug = debug
        self.verbose = verbose

        self.base_report_path = self.root / "inference_outputs" / train_plan / f"reports_{keyword}"
        self.tabular_folder_path = self.base_report_path / "evaluation_datasets"
        self.results_output_path = self.base_report_path / "results"
        
        self.meta_regex = re.compile(
            r"from the (?P<date>.*?)\s+Berkeley.*?profile:\s*(?P<demogs>.*?)\.\nAnswer.*?about\s+(?P<topic>.*?)\s+according", 
            re.DOTALL | re.IGNORECASE
        )
        self.file_pattern = re.compile(r"(results|config)_(training|train|validation|test)_.*?(lora|base)_(\d{8}_\d{6})\.(jsonl|json)")

    def setup_directories(self):
        """
        Creates the report folder structure.
        """
        self.tabular_folder_path.mkdir(parents=True, exist_ok=True)
        self.results_output_path.mkdir(parents=True, exist_ok=True)

    def parse_base_calyapo_data(self, data_path: Path, meta_path: Path) -> pd.DataFrame:
        """
        Parses the original Calyapo data from the 'final' stage of the pipeline.
        Data should be in '<prompt> : <completion>' format.
        Function extracts demographics and topics.

        Assumes order of data and meta are the same (which it is under current implementation of split_combine).
        Integrates information from meta files outputted by calyapo pipeline. 
        """
        rows = []
        if not data_path.exists():
            raise ValueError(f"Inputted file path '{data_path}' invalid.")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                data = json.loads(line)
                prompt = data.get("prompt", "")
                completion = data.get("completion", "").strip()
                
                match = self.meta_regex.search(prompt)
                if match:
                    entry = {
                        "dataset_date": match.group("date"),
                        "topic": match.group("topic"),
                        "true_answer": completion
                    }
                    
                    demog_str = match.group("demogs")
                    parts = re.split(r",\s*(?=[A-Za-z\s]+:)", demog_str)
                    for p in parts:
                        if ":" in p:
                            k, v = p.split(":", 1)
                            entry[k.strip()] = v.strip()
                    rows.append(entry)
        
        if self.verbose:
            print(f"Processed {len(rows)} lines from final calyapo dataset: '{data_path}'")

        with open(meta_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip() or idx >= len(rows): 
                    continue
                meta_data = json.loads(line)
                rows[idx].update(meta_data)
                
        return pd.DataFrame(rows)

    def get_inference_files(self, model_subfolder: Path,) -> Dict:
        """
        Locates results and config file paths for a specific model.
        """
        base_path = self.root / "inference_outputs" / self.train_plan / model_subfolder
        found = {}
        
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {base_path}")

        for file_path in base_path.iterdir():
            match = self.file_pattern.match(file_path.name)
            if match:
                file_type, split, model_type, timestamp, file_format = match.groups()
                if 'train' in split:
                    split_key = 'train' 
                elif 'val' in split:
                    split_key = 'val' 
                elif 'test' in split:
                    split_key = 'test'
                else:
                    if self.verbose: 
                        print(f"Split keyword '{split}' not recognized, skipping file.")
                    continue
                
                key = f"{split_key}_{model_type}"
                if key not in found: 
                    found[key] = {}
                else:
                    found[key][f"{file_type}_path"] = file_path
        return found

    def run_pipeline(self, model_map: Dict[str, str]):
        self.setup_directories()

        partitions = ['train', 'val', 'test']
        report_meta_config = {
            'calaypo_data_paths' : {}, 
            'inference_data_paths' : {}
        }
        combined_dataframes = {}
        for split in partitions:
            d_path = UNIVERSAL_FINAL_FOLDER / f"{self.train_plan}_{split}.jsonl"
            m_path = UNIVERSAL_FINAL_FOLDER / f"{self.train_plan}_{split}_meta.jsonl"
            combined_dataframes[split] = self.parse_base_calyapo_data(d_path, m_path)
            
            report_meta_calyapo_paths_subdict: Dict = report_meta_config['calaypo_data_paths']
            report_meta_calyapo_paths_subdict[split] = {
                'data_path' : str(d_path), # cannot serialize path objects into json 
                'meta_path' : str(m_path)
            }


        for model_nickname, sub_path in model_map.items():
            inf_files = self.get_inference_files(Path(sub_path))
            
            report_meta_inf_paths_subdict: Dict = report_meta_config['inference_data_paths']
            report_meta_inf_paths_subdict[model_nickname] = str(sub_path)

            for key, inf_paths in inf_files.items():
                if 'results_path' not in inf_paths: 
                    continue
                split, model_type = key.split('_')
                
                df_calyapo = combined_dataframes[split]
                if df_calyapo.empty: 
                    if self.debug: print(f"No dataframe found for split '{split}' from key '{key}'.")
                    continue
                
                df_inf = pd.read_json(inf_paths['results_path'], lines=True)
                if df_inf.empty: 
                    raise ValueError(f"No dataframe found for path 'inf_paths['results_path']'.")
                
                col_pred = f"{model_nickname}_{model_type}_pred"
                col_corr = f"{model_nickname}_{model_type}_correct"
                
                df_calyapo[col_pred] = df_inf['prediction'].values
                df_calyapo[col_corr] = df_inf['is_correct'].values

        for split, df in combined_dataframes.items():
            if not df.empty:
                out_file = self.tabular_folder_path / f"{self.train_plan}_{split}_tabular.csv"
                df.to_csv(out_file, index=False)
                if self.verbose: 
                    print(f"Created Tabular Data: {out_file}")

        report_meta_config.update({
            "train_plan": self.train_plan,
            "run_keyword": self.keyword,
            "models_included": list(model_map.keys()),
        })
        if self.verbose: 
            print(f"Final Meta Report:\n{report_meta_config}")
        with open(self.base_report_path / "report_meta_config.json", "w") as f:
            json.dump(report_meta_config, f, indent=4)