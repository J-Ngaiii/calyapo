import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
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

    def _pull_weights(self) -> pd.Series:
        """
        Gathers weights from all intermediate IGS datasets and 
        returns a Series indexed by calyapo_uniqueid.
        """
        if self.verbose: 
            print(f"Searching for intermediate IGS weights in calyapo/data/intermediate/igs...")

        igs_path = self.root / "calyapo" / "data" / "intermediate" / "igs"
        
        if not igs_path.exists():
            if self.verbose: 
                print(f"( _pull_weights | Reporter) Warning: Path '{igs_path}' does not exist.")
            return pd.Series(dtype=float)

        weight_col_base_name = 'w1'
        all_weight_dfs = []
        weight_files = list(igs_path.glob("*.csv"))
        if self.verbose:
            print(f"( _pull_weights | Reporter) Found '{len(weight_files)}' CSVs from calyapo intermediate path: '{igs_path}'.")
        for file_path in weight_files:
            try:
                if self.verbose:
                    print(f"( _pull_weights | Reporter) Reading intermediate calyapo csv from path: '{file_path}'.")
                df = pd.read_csv(
                    file_path, 
                    usecols=['calyapo_uniqueid', weight_col_base_name],
                    dtype={'calyapo_uniqueid': str, weight_col_base_name: float}
                )
                all_weight_dfs.append(df)
            except ValueError as e:
                if self.verbose: 
                    print(f"( _pull_weights | Reporter) Skipping {file_path.name}: Required columns not found. ({e})")
                continue

        if not all_weight_dfs:
            if self.verbose: 
                print("( _pull_weights | Reporter) No valid weight data found in intermediate folder.")
            return pd.Series(dtype=float)

        full_weight_df = pd.concat(all_weight_dfs, ignore_index=True)
        
        # remove duplicates (if a respondent appears in multiple waves, 
        # keep the last one or handle as needed)
        full_weight_df = full_weight_df.drop_duplicates(subset=['calyapo_uniqueid'])
        
        # set index for easy mapping later
        full_weight_df = full_weight_df.set_index('calyapo_uniqueid')[weight_col_base_name]
        return full_weight_df

    def load_tabulars(self, splits: List[str] = None, file_end_tag: str = 'tabular') -> Dict[str, pd.DataFrame]:
        """
        Loads tabularized CSVs into a dictionary keyed by split.
        Handles matching with calyapo intermediate CSVs to populate weights
        """
        if splits is None: 
            splits = ['train', 'val', 'test']
            
        weight_lookup = self._pull_weights()

        output = {}
        for spl in splits:
            file_name = f"{self.train_plan}_{spl}_{file_end_tag}.csv"
            file_path = self.tabular_folder_path / file_name
            if not file_path.exists():
                if self.verbose: print(f"( load_tabulars | Reporter) Warning: {spl} split not found at {file_path}")
                continue
            
            df = pd.read_csv(file_path)
            if not weight_lookup.empty:
                df['weight'] = df['uniqueid'].astype(str).map(weight_lookup).fillna(1.0)
            else:
                if self.verbose:
                    print(f"(load_tabulars | Reporter) weight lookup was empty, filling in with 1 values.")
                df['weight'] = 1.0
            
            if self.debug:
                print(f"(load_tabulars | Reporter) average weight col values: {np.average(df['weight'])}")

            output[spl] = df
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
                        if self.verbose: 
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
    def _get_weighted_crosstab(self, df, group_col, target_col, weight_col='weight'):
        """
        Helper to calculate weighted proportions manually.

        group_col corresponds to demographic col we're grouping by with multiple discrete categories (eg. Age: 18-29, 30-39, ..., etc).
        target_col corresponds to survey question col we're analyzing (eg. Strongly Favor, Somewhat Favor, ..., etc).
        Each row in the df must represent an individual with a weight col. 
        """
        if weight_col not in df.columns:
            df[weight_col] = 1.0
            
        if self.debug:
            # print(f"( _get_weighted_crosstab | Reporter) group_col unique values: {df[group_col].unique()}")
            # print(f"( _get_weighted_crosstab | Reporter) target_col unique values: {df[target_col].unique()}")
            # print(f"( _get_weighted_crosstab | Reporter) weight_col class: {df[weight_col].dtype}")
            pass
        # group by demog and answer, sum the weights
        weighted_counts = df.pivot_table(
            index=group_col, 
            columns=target_col, 
            values=weight_col, 
            aggfunc='sum', 
            fill_value=0
        )
        
        # normalize rows to create percentages
        total_weight_per_subgroup = weighted_counts.sum(axis=1) # sum up weights across all choices for a given subgroup (eg Age 18-29)
        weighted_probs = weighted_counts.div(total_weight_per_subgroup, axis=0) * 100 # df.div(input_arr) broadcasts division -> making it so that values across all cols in row 1 are divided by input_arr[1]
        return weighted_probs.round(2)
    
    def generate_crosstabs(self):
        """
        Creates crosstab responses based on demographics.
        """
        if self.verbose: print(f"Generating Crosstabs for {self.run_keyword}...")
        
        tabs = self.load_tabulars()
        crosstab_out = self.results_folder / "crosstabs"

        for split, df in tqdm(tabs.items(), desc='Generating crosstabs on train, val and test data.'):
            # Identify demographics dynamically
            exclude = ['dataset_date', 'time_period', 'dataset',  'weight', 'topic', 'true_answer', 'Question', 'index', 'uniqueid', 'id'] + \
                      [c for c in df.columns if c.endswith('_correct') or c.endswith('_pred')]
            demog_cols = [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]

            if self.debug:
                # print(f"( generate_crosstab | Reporter) demog_cols extracted {demog_cols}") 
                pass
            
            topics = df['topic'].unique()
            model_pred_cols = [c for c in df.columns if c.endswith('_pred')]

            for topic_var in topics:
                topic_label = "".join([c if c.isalnum() else "_" for c in topic_var])
                topic_df = df[df['topic'] == topic_var]

                for demog in demog_cols:
                    # True Distribution
                    base_ct = pd.crosstab(topic_df[demog], topic_df['true_answer'], normalize='index') * 100
                    base_ct.columns = [f"true_{c}" for c in base_ct.columns]

                    # Weighted True Distribution
                    weighted_base_ct = self._get_weighted_crosstab(df=topic_df, group_col=demog, target_col='true_answer', weight_col='weight')
                    weighted_base_ct.columns = [f"weighted_true_{c}" for c in weighted_base_ct.columns]
                    
                    # Distribution of model predictions, iterate thru all models and crosstab
                    all_cts = [base_ct, weighted_base_ct]
                    for mcol in model_pred_cols:
                        m_name = mcol.replace('_pred', '')
                        m_ct = pd.crosstab(topic_df[demog], topic_df[mcol], normalize='index') * 100
                        m_ct.columns = [f"{m_name}_{c}" for c in m_ct.columns]
                        all_cts.append(m_ct)

                        m_weighted_ct = self._get_weighted_crosstab(df=topic_df, group_col=demog, target_col=mcol, weight_col='weight')
                        m_weighted_ct.columns = [f"weighted_model_{m_name}_{c}" for c in m_weighted_ct.columns]
                        all_cts.append(m_weighted_ct)

                    master_ct = pd.concat(all_cts, axis=1).round(2).reset_index()
                    save_path = crosstab_out / split / topic_label / f"by_{demog}_comparison.csv"
                    file_saver(out_path=save_path, data=master_ct, data_type='csv', verbose=self.verbose)

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

    def distributional_accuracy(self, demog_col_indices: List[int] = [0]):
        """
        Calculates KL and WD for both Weighted and Unweighted distributions
        by comparing True survey distributions against Model prediction distributions.
        """
        if self.verbose: print(f"Calculating Distributional Accuracy (KL/WD)...")
        
        crosstab_root = self.results_folder / "crosstabs"
        csv_files = list(crosstab_root.glob("**/*_comparison.csv"))
        
        if not csv_files:
            print("( distributional_accuracy | Reporter) No crosstabs found. Run generate_crosstabs() first.")
            return

        all_results = []
        for file_path in tqdm(csv_files, desc="Processing Crosstabs for Metrics"):
            split = file_path.parts[-3]
            question = file_path.parts[-2]
            df = pd.read_csv(file_path)
            
            demog_labels = df.columns[demog_col_indices].tolist()
            
            true_cols = [c for c in df.columns if c.startswith('true_')]
            w_true_cols = [c for c in df.columns if c.startswith('weighted_true_')]
            choices = [c.replace('true_', '') for c in true_cols]
            
            # identify models (look for columns ending in _A, _B, etc., but not 'true')
            # convention: {model_name}_{type}_{choice} 
            potential_models = [c for c in df.columns if c not in true_cols 
                               and c not in w_true_cols 
                               and c not in demog_labels 
                               and not c.startswith('weighted_model_')]
            
            # extract unique model nicknames
            models = sorted(list(set([m.rsplit('_', 1)[0] for m in potential_models if '_' in m])))

            for demog_col in demog_labels:
                for _, row in df.iterrows():
                    subgroup = row[demog_col]
                    if pd.isna(subgroup): continue
                    
                    # calculate metrics for each model
                    for model_id in models:
                        # --- UNWEIGHTED ANALYSIS ---
                        p_unweighted = row[true_cols].values.astype(float)
                        # build q vector handling cases where model never predicted a certain choice
                        q_unweighted = np.array([
                            row[f"{model_id}_{c}"] if f"{model_id}_{c}" in df.columns else 0.0 
                            for c in choices
                        ], dtype=float)
                        
                        kl_u, wd_u = self._calculate_dist_metrics(p_unweighted, q_unweighted)

                        # --- WEIGHTED ANALYSIS ---
                        # can extract results of weighted models but just appending the "weighted_model_" col name prior
                        p_weighted = row[w_true_cols].values.astype(float)
                        q_weighted = np.array([
                            row[f"weighted_model_{model_id}_{c}"] if f"weighted_model_{model_id}_{c}" in df.columns else 0.0 
                            for c in choices
                        ], dtype=float)

                        if self.debug:
                            # print(f"row: {row}")
                            # print(f"p_weighted: {p_weighted}")
                            # print(f"q_weighted: {q_weighted}")
                            pass
                        
                        kl_w, wd_w = self._calculate_dist_metrics(p_weighted, q_weighted)

                        # 4. Store Results
                        all_results.append({
                            "Split": split, 
                            "Question": question, 
                            "Demographic": demog_col,
                            "Subgroup": subgroup, 
                            "Model": model_id,
                            "KL_Unweighted": kl_u, 
                            "WD_Unweighted": wd_u,
                            "KL_Weighted": kl_w, 
                            "WD_Weighted": wd_w
                        })

        final_df = pd.DataFrame(all_results)
        out_path = self.results_folder / "distributional_accuracy"
        out_path.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(out_path / "aggregated_kl_metrics.csv", index=False)
        
        # summary grouping updated to include weighted metrics
        metrics = ['KL_Unweighted', 'WD_Unweighted', 'KL_Weighted', 'WD_Weighted']
        summary = final_df.groupby(['Split', 'Question', 'Model'])[metrics].mean()
        summary.to_csv(out_path / "summary_metrics.csv")
        
        if self.verbose: 
            print(f"( distributional_accuracy | Reporter) Success: Weighted and Unweighted metrics saved to {out_path}")