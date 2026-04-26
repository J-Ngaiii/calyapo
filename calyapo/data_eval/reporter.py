import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import entropy, wasserstein_distance
from typing import Union, Dict, List, Tuple
from calyapo.utils.persistence import file_saver

class Reporter:
    def __init__(self, train_plan: str, run_keyword: str, root_path = ".", debug: bool = False, verbose = False):
        self.train_plan = train_plan
        self.run_keyword = run_keyword

        self.debug = debug
        self.verbose = verbose

        self.root = Path(root_path)
        self.base_report_path = self.root / "inference_outputs" / train_plan / f"reports_{run_keyword}"
        self.tabular_folder_path = self.base_report_path / "evaluation_datasets"
        self.results_folder = self.base_report_path / "results"
        
        config_path = self.base_report_path / 'report_meta_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Meta config not found at {config_path}. Run tabularizer first.")
        with open(config_path, 'r') as f:
            self.report_meta_config = json.load(f)
        self.model_names = self.report_meta_config['models_included']
        
        self.debug = debug
        self.verbose = verbose

    def load_tabulars(self, splits: List[str] = None, file_end_tag: str = 'tabular') -> Dict[str, pd.DataFrame]:
        """
        Loads tabularized CSVs into a dictionary keyed by split.
        """
        if splits is None: 
            splits = ['train', 'val', 'test']
            
        output = {}
        for spl in splits:
            file_name = f"{self.train_plan}_{spl}_{file_end_tag}.csv"
            file_path = self.tabular_folder_path / file_name
            if not file_path.exists():
                if self.verbose: print(f"Warning: {spl} split not found at {file_path}")
                continue
            output[spl] = pd.read_csv(file_path)
        return output

    # ----------------------------
    # Model Accuracy Reporting
    # ----------------------------
    def _helper_acc_df(self, tabulars_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        report_list = []
        for split, df in tabulars_dict.items():
            for model in self.model_names:
                for m_type in ['base', 'lora']:
                    col = f"{model}_{m_type}_correct"
                    if col in df.columns:
                        report_list.append({
                            'Model_Name': model,
                            'Split': split.capitalize(),
                            'Type': m_type.upper(),
                            'Accuracy': df[col].mean()
                        })
                    else:
                        if verbose: 
                            print(f"Warning could not find column {col}")
        # creates up to 24 entries (4 llama models base and lora versions each getting an entry for each of the three splits)
        return pd.DataFrame(report_list)

    def _acc_plot(self, df: pd.DataFrame, save_filename: str = None, show: bool = False):
            sns.set_style("whitegrid")
            palette = {"LORA": "orange", "BASE": "dodgerblue"}
            
            models = df['Model_Name'].unique()
            # handle cases where you might have fewer than 4 models
            n_models = len(models)
            nrows = (n_models + 1) // 2
            fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows))
            axes = axes.flatten()

            for i, model in enumerate(models):
                ax = axes[i]
                model_df = df[df['Model_Name'] == model]
                sns.barplot(data=model_df, x="Split", y="Accuracy", hue="Type", 
                            palette=palette, ax=ax, alpha=0.8)
                
                ax.set_title(f"Performance: {model}")
                ax.set_ylim(0, 1.0) # accuracy is 0-1
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', padding=3)
                if i != 0:
                    ax.get_legend().remove()

            plt.tight_layout()
            
            if save_filename:
                self.results_folder.mkdir(parents=True, exist_ok=True)
                save_path = self.results_folder / save_filename
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.verbose: print(f"Plot saved to: {save_path}")
            
            if show: plt.show()

    def accuracy(self, show_plots = False):
        """
        Main entry point to auto-run the accuracy analysis.
        """
        if self.verbose: 
            print(f"Generating Accuracy Report for {self.run_keyword}...")
        
        tabulars = self.load_tabulars()
        if not tabulars:
            print("No data loaded. Check paths.")
            return

        report_df = self._helper_acc_df(tabulars)
        
        if not report_df.empty:
            save_name = f"{self.train_plan}_accuracy_comparison.png"
            self._acc_plot(report_df, save_filename=save_name, show=show_plots)
            # also save the raw numbers
            report_df.to_csv(self.results_folder / "accuracy_metrics.csv", index=False)
        else:
            print("No matching accuracy columns found.")

    # ----------------------------
    # Crosstab Generation
    # ----------------------------
    def generate_crosstabs(self):
        """
        Creates crosstab responses based on demographics.
        """
        if self.verbose: print(f"Generating Crosstabs for {self.run_keyword}...")
        
        tabs = self.load_tabulars()
        crosstab_out = self.results_folder / "crosstabs"

        for split, df in tabs.items():
            # Identify demographics dynamically
            exclude = ['dataset_date', 'topic', 'true_answer', 'Question', 'index', 'uniqueid', 'id'] + \
                      [c for c in df.columns if c.endswith('_correct') or c.endswith('_pred')]
            demog_cols = [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]
            
            topics = df['topic'].unique()
            model_pred_cols = [c for c in df.columns if c.endswith('_pred')]

            for topic_var in topics:
                topic_label = "".join([c if c.isalnum() else "_" for c in topic_var])
                topic_df = df[df['topic'] == topic_var]

                for demog in demog_cols:
                    # True Distribution
                    base_ct = pd.crosstab(topic_df[demog], topic_df['true_answer'], normalize='index') * 100
                    base_ct.columns = [f"true_{c}" for c in base_ct.columns]
                    
                    all_cts = [base_ct]
                    for mcol in model_pred_cols:
                        m_name = mcol.replace('_pred', '')
                        m_ct = pd.crosstab(topic_df[demog], topic_df[mcol], normalize='index') * 100
                        m_ct.columns = [f"{m_name}_{c}" for c in m_ct.columns]
                        all_cts.append(m_ct)

                    master_ct = pd.concat(all_cts, axis=1).round(2).reset_index()
                    save_path = crosstab_out / split / topic_label / f"by_{demog}_comparison.csv"
                    file_saver(out_path=save_path, data=master_ct, data_type='csv', verbose=False)

    # ----------------------------
    # Distributional Accuracy (KL/WD)
    # ----------------------------
    def _calculate_dist_metrics(self, p, q):
        p = p.astype(float) / (p.sum() + 1e-12)
        q = q.astype(float) / (q.sum() + 1e-12)
        eps = 1e-6
        p = (p + eps) / (p + eps).sum()
        q = (q + eps) / (q + eps).sum()
        return entropy(p, q), wasserstein_distance(p, q)

    def _pull_weights(self):
        """Use unique id generator to then match """

    def distributional_accuracy(self, demog_col_indices: List[int] = [0]):
        """
        Calculates KL and WD from generated crosstabs or directly from tabular.csvs to incorperate weights"""
        if self.verbose: print(f"Calculating Distributional Accuracy (KL/WD)...")
        
        crosstab_root = self.results_folder / "crosstabs"
        csv_files = list(crosstab_root.glob("**/*_comparison.csv"))
        
        if not csv_files:
            print("No crosstabs found. Run generate_crosstabs() first.")
            return

        all_results = []
        for file_path in csv_files:
            split = file_path.parts[-3]
            question = file_path.parts[-2]
            df = pd.read_csv(file_path)
            
            demog_labels = df.columns[demog_col_indices].tolist()
            true_cols = [c for c in df.columns if c.startswith('true_')]
            choices = [c.replace('true_', '') for c in true_cols]
            
            # Identify models from columns
            potential_models = [c for c in df.columns if c not in true_cols and c not in demog_labels]
            models = sorted(list(set([m.rsplit('_', 1)[0] for m in potential_models if '_' in m])))

            for demog_col in demog_labels:
                for _, row in df.iterrows():
                    subgroup = row[demog_col]
                    if pd.isna(subgroup): continue
                    
                    p_dist = row[true_cols].values.astype(float)
                    
                    for model_id in models:
                        q_vals = [row[f"{model_id}_{c}"] if f"{model_id}_{c}" in df.columns else 0.0 for c in choices]
                        kl, wd = self._calculate_dist_metrics(p_dist, np.array(q_vals))
                        
                        all_results.append({
                            "Split": split, "Question": question, "Demographic": demog_col,
                            "Subgroup": subgroup, "Model": model_id,
                            "KL_Divergence": kl, "Wasserstein_Distance": wd
                        })

        final_df = pd.DataFrame(all_results)
        out_path = self.results_folder / "distributional_accuracy"
        out_path.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(out_path / "aggregated_kl_metrics.csv", index=False)
        summary = final_df.groupby(['Split', 'Question', 'Model'])[['KL_Divergence', 'Wasserstein_Distance']].mean()
        summary.to_csv(out_path / "summary_metrics.csv")
        if self.verbose: print(f"KL/WD Metrics saved to {out_path}")