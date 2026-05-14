import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import geopandas as gpd
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
        self.graphs_folder = self.results_folder / "graphs"
        
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
            
            if self.verbose:
                print(f"Loading tabulars from path: '{file_path}'")
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
    def _calculate_weighted_accuracy(self, df: pd.DataFrame, correct_col: str, weight_col: str = 'weight') -> float:
        """
        Calculates the weighted mean of the 'correct' column.
        """
        if df.empty:
            return 0.0
        
        weights = df[weight_col].fillna(1.0)
        correct = df[correct_col].fillna(0).astype(int)
        
        weighted_sum = (correct * weights).sum()
        total_weight = weights.sum()
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _helper_acc_df(self, tabulars_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        report_list = []
        for split, df in tabulars_dict.items():
            for model in self.model_names:
                for m_type in ['base', 'lora']:
                    col = f"{model}_{m_type}_correct"
                    if col in df.columns:
                        w_acc = self._calculate_weighted_accuracy(df, col, 'weight')
                        report_list.append({
                            'Model_Name': model,
                            'Split': split.capitalize(),
                            'Type': m_type.upper(),
                            'Accuracy': df[col].mean(), 
                            'Weighted Accuracy': w_acc
                        })
                    else:
                        if self.verbose: 
                            print(f"Warning could not find column {col}")
        # creates up to 24 entries (4 llama models base and lora versions each getting an entry for each of the three splits)
        return pd.DataFrame(report_list)
    
    def _acc_plot(self, df: pd.DataFrame, acc_col_name: str = 'Accuracy', save_filename: str = None, show: bool = False):
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
                
                ax.set_title(f"{acc_col_name} Performance: {model}")
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
            
            if show: 
                plt.show()

            plt.close()

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
            acc_name = f"{self.train_plan}_accuracy_comparison.png"
            weighted_acc_name = f"{self.train_plan}_weighted_accuracy_comparison.png"
            self._acc_plot(report_df, acc_col_name='Accuracy', save_filename=acc_name, show=show_plots)
            self._acc_plot(report_df, acc_col_name='Weighted Accuracy', save_filename=weighted_acc_name, show=show_plots)
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
    def _calculate_dist_metrics(self, p, q, config: Dict = None):
        p = p.astype(float) / (p.sum() + 1e-12)
        q = q.astype(float) / (q.sum() + 1e-12)
        eps = 1e-6
        p = (p + eps) / (p + eps).sum()
        q = (q + eps) / (q + eps).sum()
        tv = 0.5 * np.sum(np.abs(p - q))
        return entropy(p, q), wasserstein_distance(p, q), tv

    def distributional_accuracy(self, demog_col_indices: List[int] = [0]):
        """
        Calculates KL and WD for both Weighted and Unweighted distributions
        by comparing True survey distributions against Model prediction distributions.
        """
        if self.verbose: print(f"Calculating Distributional Accuracy (KL/WD/TV)...")
        
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
                        q_unweighted = []
                        for c in choices:
                            val = 0.0
                            if f"{model_id}_{c}" in df.columns:
                                val += row[f"{model_id}_{c}"]
                            if f"{model_id}_{c}." in df.columns:
                                val += row[f"{model_id}_{c}."]
                            q_unweighted.append(val)
                        q_unweighted = np.array(q_unweighted, dtype=float)

                        # if self.debug:
                        #     if model_id in {'Llama-3.1-8B-Instruct_base', 'Llama-3.2-3B-Instruct_base', 'Qwen2.5-14B-Instruct_base'}:
                        #         print(f"unweighted distribution for {model_id}: {q_unweighted}")
                        
                        u_config_for_prints = {
                            'model_id': model_id, 
                            'demog_col': demog_col, 
                            'weighting': 'non-weighted'
                        }
                        kl_u, wd_u, tv_u = self._calculate_dist_metrics(p_unweighted, q_unweighted, config=u_config_for_prints)

                        # --- WEIGHTED ANALYSIS ---
                        # can extract results of weighted models but just appending the "weighted_model_" col name prior
                        p_weighted = row[w_true_cols].values.astype(float)
                        q_weighted = []
                        for c in choices:
                            val = 0.0
                            if f"weighted_model_{model_id}_{c}" in df.columns:
                                val += row[f"weighted_model_{model_id}_{c}"]
                            if f"weighted_model_{model_id}_{c}." in df.columns:
                                val += row[f"weighted_model_{model_id}_{c}."]
                            q_weighted.append(val)
                        q_weighted = np.array(q_weighted, dtype=float)

                        if self.debug:
                            # print(f"row: {row}")
                            # print(f"p_weighted: {p_weighted}")
                            # print(f"q_weighted: {q_weighted}")
                            pass
                        
                        w_config_for_prints = {
                            'model_id': model_id, 
                            'demog_col': demog_col, 
                            'weighting': 'weighted'
                        }
                        kl_w, wd_w, tv_w = self._calculate_dist_metrics(p_weighted, q_weighted, config=w_config_for_prints)

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
                            "WD_Weighted": wd_w,
                            "TV_Unweighted": tv_u,
                            "TV_Weighted": tv_w
                        })

        # complete df
        final_df = pd.DataFrame(all_results)
        out_path = self.results_folder / "distributional_accuracy"
        out_path.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(out_path / "aggregated_kl_metrics.csv", index=False)

        # demographic aggregation csv
        metrics = metrics = ['KL_Unweighted', 'WD_Unweighted', 'KL_Weighted', 'WD_Weighted', 
           'TV_Unweighted', 'TV_Weighted']
        demog_summary = final_df.groupby(['Split', 'Question', 'Demographic', 'Model'])[metrics].mean().reset_index()
        for m_name in metrics:
            min_val = demog_summary.groupby(['Split', 'Question'])[m_name].transform('min')
            demog_summary[f'is_best_{m_name}'] = demog_summary[m_name] == min_val
        demog_summary.to_csv(out_path / "summary_demog_metrics.csv", index=False)
        
        # model aggregation csv
        summary = final_df.groupby(['Split', 'Question', 'Model'])[metrics].mean()
        for m_name in metrics:
            min_val = summary.groupby(['Split', 'Question'])[m_name].transform('min')
            summary[f'is_best_{m_name}'] = summary[m_name] == min_val
        summary.to_csv(out_path / "summary_metrics.csv")
        
        if self.verbose: 
            print(f"( distributional_accuracy | Reporter) Success: Weighted and Unweighted metrics saved to {out_path}")

    # -------------
    # Geo Analysis
    # ------------- 
    def generate_geographic_reports(self, split='test', geo_level='zip', train_setting: int = 1):
        """
        Wrapper to run both Ground Truth and Synthetic geographic analyses.
        """
        for mode in ['ground_truth', 'synthetic']:
            if self.verbose:
                print(f"Running {mode} geographic analysis...")
            self._plot_geo_core(split=split, geo_level=geo_level, visual_type=mode, setting=train_setting)

    def _geo_level_helper(self, geo_level):
        counties_map = set(['county', 'counties', 'c', 'cnty'])
        zips_map = set(['zip', 'zipcode', 'rzip'])

        if geo_level.lower().strip() in counties_map:
            return 'CNTY'
        elif geo_level.lower().strip() in zips_map:
            return 'RZIP'
        else:
            raise ValueError(f"Unkown geo_level {geo_level}")
    
    def _plot_geo_core(self, split: str, geo_level: str, visual_type: str, setting: int = 1):
        tabs = self.load_tabulars(splits=[split]) # loads evaluation_dataset tabulars with weights
        split_tabular = tabs[split]
        
        igs_path = self.root / "calyapo" / "data" / "intermediate" / "igs"
        geo_level_col_name = self._geo_level_helper(geo_level=geo_level)
        all_geo_fragments = []
        for interim_filepath in igs_path.glob("*.csv"):
            interim_df_cols = pd.read_csv(interim_filepath, nrows=0).columns.tolist()
            if 'calyapo_uniqueid' in interim_df_cols and geo_level_col_name in interim_df_cols:
                interim_df_full = pd.read_csv(interim_filepath, usecols=['calyapo_uniqueid', geo_level_col_name], dtype=str)
                all_geo_fragments.append(interim_df_full)

        if not all_geo_fragments:
            print("Error: No RZIP columns found in intermediate IGS files.")
            return

        geo_df = pd.concat(all_geo_fragments, ignore_index=True).drop_duplicates('calyapo_uniqueid')
        geo_df[geo_level_col_name] = geo_df[geo_level_col_name].astype(str).str.split('.').str[0].str.zfill(5) # make sure zip code is 5 digits
        joined_tabular = split_tabular.merge(geo_df, left_on='uniqueid', right_on='calyapo_uniqueid', how='left') # merge evaluation_dataset tabulars (uses uniqueid) with intermediate IGS data (uses calyapo_uniqueid)

        shape_path = self.root / "calyapo" / "data" / "eval" / "zip_poly.shp" 
        gdf = gpd.read_file(shape_path)
        if self.debug:
            print(f"gdf columns:\n{gdf.columns}")
            print(f"gdf first few rows:\n{gdf.head(5)}")

        # handling different ways of encoding zip code in shapefiles
        if 'ZIP_CODE' in gdf.columns:
            shape_join_col = 'ZIP_CODE'
        elif 'ZCTA5CE20' in gdf.columns:
            shape_join_col = 'ZCTA5CE20'
        elif 'ZCTA5' in gdf.columns:
            shape_join_col = 'ZCTA5'
        else:
            raise ValueError(f"Could not find a valid ZIP column in shapefile. Columns: {gdf.columns}") 
        
        gdf[shape_join_col] = gdf[shape_join_col].astype(str).str.zfill(5)

        if setting == 1: # hardcode for now
            survey_questions = ["Donald Trump", "Joe Biden", "Kamala Harris"]
            setting_label = "Favorability Towards Presidential Candidates"
            cmap = LinearSegmentedColormap.from_list("politics", ["#c91616", "#ebe701", "#00bc29"]) # red, purple and blue for politics
        else:
            survey_questions = ["Defending Abortion Rights"]
            setting_label = "Abortion_Access"
            cmap = LinearSegmentedColormap.from_list("opinion", ["#C21414", "#2483d1"]) # it just needs to be a dark color
        
        if visual_type == 'ground_truth':
            tabular_ground_truth_cols = ['true_answer']
            target_cols = tabular_ground_truth_cols
        elif visual_type == 'synthetic':
            tabular_LLM_model_cols = [c for c in joined_tabular.columns if c.endswith('_pred')]
            target_cols = tabular_LLM_model_cols
        for col in tqdm(target_cols, desc=f"Mapping {visual_type} via {geo_level_col_name}"):
            for targ_question in survey_questions:
                if 'topic' not in joined_tabular.columns:
                    raise ValueError(f"No 'topic' column detected in tabular df for inputted split '{split}'. Inputted tabular only has columns: {split_tabular.columns}")
                sub_df = joined_tabular[joined_tabular['topic'].str.contains(targ_question, case=False, na=False)].copy()
                # if self.debug:
                #     print(f"Sub df cols: {sub_df.columns}")
                if sub_df.empty: 
                    if self.verbose:
                        print(f"Did not find survey question '{targ_question}' under 'topic' col in tabular for split: '{split}'. Only had the following topics: {split_tabular['topic'].unique()}")
                    continue
                # col is either a 'true_answer' col or "_pred" cols that come from tabulars
                sentiment_map = {'A': 1.0, 'A.': 1.0,  
                                 'B': 0.66, 'B.': 0.66,
                                 'C': 0.33, 'C.': 0.33, 
                                 'D': 0.0, 'D.': 0.0}
                sub_df['harmonized_score'] = sub_df[col].map(sentiment_map)
                # geo_level_col_name comes from IGS intermediate and joining with IGS intermediate
                stats = sub_df.groupby(geo_level_col_name).apply( 
                    lambda x: (x['harmonized_score'] * x['weight']).sum() / x['weight'].sum() # sum up weights then divide, multiplying by indicator to toggle
                ).reset_index(name='prop')
                
                stats[geo_level_col_name] = stats[geo_level_col_name].astype(str).str.zfill(5)
                merged = gdf.merge(stats, left_on=shape_join_col, right_on=geo_level_col_name, how='left') # keep gdf col's zip col --> so every single zip code has a row
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                gdf.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc', linewidth=0.1) # make sure areas with no data stay greyed, make sure to plot onto the same axes
                valid_zips_groupby = merged.dropna(subset=['prop']) # drop rows corresponding to zip codes for which there was no value found from the join
                
                if self.debug:
                    num_zips_with_data = len(valid_zips_groupby) / len(merged)
                    print(f"Spatial coverage for {targ_question}: {num_zips_with_data:.2%}")

                valid_zips_groupby.plot( # plot onto the same axes
                    column='prop', 
                    cmap=cmap, 
                    legend=True, 
                    ax=ax, 
                    edgecolor='none', 
                    vmin=0, # fix axis
                    vmax=1
                )
                
                ax.set_title(f"{visual_type.upper()} {setting_label}: {targ_question}\n{geo_level_col_name} Weighted Opinion Towards {targ_question}", fontsize=14)
                ax.axis('off')
                out_dir = self.results_folder / "maps" / split / f"setting_{setting}" / visual_type
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path_png = out_dir / f"{targ_question.replace(' ', '_')}_{col}_zip_map.png"
                # out_path_csv = out_dir / f"{targ_question.replace(' ', '_')}_{col}_zip_df.csv"
                plt.savefig(out_path_png, dpi=900, bbox_inches='tight')
                # merged.to_csv(out_path_csv)
                if self.debug:
                    print(f"Saved png to: {out_path_png}")
                    # print(f"Saved csv to: {out_path_csv}")
                plt.close(fig)
    # -------------
    # Conf Analysis
    # -------------
   
    def _collect_confidence_df(self, tabulars_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Creates a copy of the tabulars but cleaned out
        """
        rows = []


        for split, df in tabulars_dict.items():


            for model in self.model_names:


                for m_type in ["base", "lora"]:


                    prefix = f"{model}_{m_type}"


                    top_lp = f"{prefix}_top_logprob"
                    top2 = f"{prefix}_top2_diff"
                    top5 = f"{prefix}_top5_sd"
                    correct = f"{prefix}_correct"


                    if top_lp not in df.columns:
                        continue


                    tmp = pd.DataFrame({
                        "Split": split,
                        "Model": model,
                        "Type": m_type.upper(),
                        "TopLogProb": df[top_lp],
                        "Top2Diff": df[top2] if top2 in df.columns else None,
                        "Top5SD": df[top5] if top5 in df.columns else None,
                        "Correct": df[correct] if correct in df.columns else None
                    })


                    rows.append(tmp)


        return pd.concat(rows, ignore_index=True)
   
    def _confidence_plot(
        self,
        df: pd.DataFrame,
        split: str,
        save_filename: str = None,
        show: bool = False,
        plot_top_logprob: bool = True,
        plot_top2_diff: bool = False,
        plot_top5_sd: bool = False,
    ):
        df = df[df['Split'] == split].copy()
        sns.set_style("whitegrid")
        plot_configs = []

        if plot_top_logprob:
            plot_configs.append(("TopLogProb", "Top Token Log-Probability (Confidence)"))
        if plot_top2_diff:
            plot_configs.append(("Top2Diff", "Top-2 Logprob Gap (Decision Margin)"))
        if plot_top5_sd:
            plot_configs.append(("Top5SD", "Top-5 Logprob Std Dev (Uncertainty Spread)"))

        y_labels = {
            "TopLogProb": "Log(P(Top Token))",
            "Top2Diff": "Top-2 Logprob Gap (Decision Margin)",
            "Top5SD": "Top-5 Logprob Std Dev (Uncertainty Spread)"
        }

        n = len(plot_configs)
        if n == 0:
            raise ValueError("No plots selected for confidence visualization.")
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (col, title) in zip(axes, plot_configs):

            sns.boxplot(
                data=df,
                x="Model",
                y=col,
                hue="Type",
                ax=ax,
                showfliers=False  # remove outliers
            )
            ax.set_title(title)
            ax.set_ylabel(y_labels.get(col, col))
            # rotate + shrink x-labels
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=30,
                ha="right",
                fontsize=9
            )
            # shrink y tick labels slightly too
            ax.tick_params(axis='y', labelsize=9)


        plt.tight_layout()


        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)
            save_path = folder_path / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")


        if show:
            plt.show()


    def _confidence_vs_accuracy(self, df: pd.DataFrame, split: str, save_filename: str = None, show: bool = False):
        """
        Works by first binning confidence tabulars by Confidence values (which is just the probability the model had for the top token it selected),
        then calculating average confidence per bin and model accuracy per bin.
        """
       
        df = df[df['Split'] == split].dropna(subset=["TopLogProb", "Correct"]).copy()
        df["Confidence"] = np.exp(df["TopLogProb"])
        df["FullPrefix"] = df["Model"] + "_" + df["Type"]
        prefixes = df["FullPrefix"].unique()


        plt.figure(figsize=(7, 6))
        for prefix in prefixes:
            sub = df[df["FullPrefix"] == prefix].copy()
            if len(sub) == 0:
                if self.debug:
                    print(f"(_confidence_vs_accuracy) No sub df detected in")
                continue
            sub["ConfBin"] = pd.cut(sub["Confidence"], bins=10)
            grouped = sub.groupby("ConfBin").agg(
                accuracy=("Correct", "mean"),
                confidence=("Confidence", "mean")
            ).reset_index()


            plt.plot(
                grouped["confidence"],
                grouped["accuracy"],
                marker="o",
                label=prefix
            )


        # ideal calibration
        plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")


        plt.xlabel("Prediction Confidence (P(Top Token))")
        plt.ylabel("Model Accuracy")
        plt.title(f"Calibration Curve ({split} Set)")
        plt.legend()


        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)
            save_path = folder_path / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")


        if show:
            plt.show()


    def _calibration_plots_by_family(self, df: pd.DataFrame, split: str, show: bool = False):
        df = df[df['Split'] == split].dropna(subset=["TopLogProb", "Correct"]).copy()
        df["Confidence"] = np.exp(df["TopLogProb"])
        df["FullPrefix"] = df["Model"] + "_" + df["Type"]


        families = {
            "llama3.1": df[df["Model"].str.contains("llama-3.1", case=False)],
            "llama3.2": df[df["Model"].str.contains("llama-3.2", case=False)],
            "qwen14b": df[df["Model"].str.contains("qwen", case=False)],
        }


        for fam_name, subdf in families.items():


            if subdf.empty:
                if self.debug:
                    print(f"Subdf for family '{fam_name}' is empty")
                continue


            plt.figure(figsize=(7, 6))
            prefixes = sorted(subdf["FullPrefix"].unique())
            base_prefixes = [p for p in prefixes if "_base" in p.strip().lower()]
            lora_prefixes = [p for p in prefixes if "_lora" in p.strip().lower()]


            # lora is orange, base is blue
            blue_palette = cm.Blues(np.linspace(0.4, 0.85, len(base_prefixes)))
            orange_palette = cm.YlOrBr(np.linspace(0.4, 0.9, len(lora_prefixes)))


            color_map = {}


            for p, c in zip(base_prefixes, blue_palette):
                color_map[p] = c


            for p, c in zip(lora_prefixes, orange_palette):
                color_map[p] = c


            for prefix in prefixes:


                mdf = subdf[subdf["FullPrefix"] == prefix].copy()


                if len(mdf) == 0:
                    continue


                mdf["ConfBin"] = pd.cut(mdf["Confidence"], bins=10)


                grouped = mdf.groupby("ConfBin").agg(
                    accuracy=("Correct", "mean"),
                    confidence=("Confidence", "mean")
                ).reset_index()


                plt.plot(
                    grouped["confidence"],
                    grouped["accuracy"],
                    marker="o",
                    label=prefix,
                    color=color_map.get(prefix, None)
                )


            # ideal calibration line
            plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")


            plt.xlabel("Prediction Confidence (P(Top Token))")
            plt.ylabel("Model Accuracy")
            plt.title(f"Calibration Curve: {fam_name}")
            plt.legend()


            folder_path = Path(f"{self.graphs_folder}/{split}")
            save_path = folder_path / f"calibration_{fam_name}.png"
            folder_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")


            if show:
                plt.show()


            plt.close()
   
    def confidence_analysis(self, split: str, show_plots=False):


        tabulars = self.load_tabulars()
        if not tabulars:
            print("No data loaded.")
            return


        conf_df = self._collect_confidence_df(tabulars)


        if conf_df.empty:
            print("No confidence columns found.")
            return


        self._confidence_plot(
            conf_df,
            split=split, 
            save_filename=f"{self.train_plan}_confidence_summary.png",
            show=show_plots
        )


        self._confidence_vs_accuracy(
            conf_df,
            split=split, 
            save_filename=f"{self.train_plan}_confidence_calibration.png",
            show=show_plots
        )


        self._calibration_plots_by_family(
            conf_df,
            split=split, 
        )


        conf_df.to_csv(
            self.results_folder / f"{split}_confidence_metrics.csv",
            index=False
        )
    # ----------------------------
    # Prediction Distribution / Collapse Analysis
    # ----------------------------
    def _prediction_distribution_plot(
        self,
        tabulars_dict: Dict[str, pd.DataFrame],
        split: str = "test",
        save_filename: str = None,
        show: bool = False,
        granular: bool = True,
    ):
        """
        Creates a normalized stacked bar chart showing
        prediction distributions for every model
        (base + lora).

        Parameters
        ----------
        granular : bool
            If True:
                - Each response choice gets its own shade
                (multiple greens/reds).
            If False:
                - Only two categories are shown:
                Valid (green) vs Invalid (red).
        """

        if split not in tabulars_dict:
            raise ValueError(f"Split '{split}' not found in tabulars.")

        df = tabulars_dict[split]
        pred_cols = [c for c in df.columns if c.endswith("_pred")]

        if len(pred_cols) == 0:
            raise ValueError("No *_pred columns found.")

        all_answers = sorted(
            pd.unique(
                pd.concat(
                    [df[col].dropna() for col in pred_cols],
                    ignore_index=True
                )
            )
        )

        # Optional: enforce Likert ordering
        likert_order = ["A", "B", "C", "D", "E"]

        if set(likert_order).issuperset(set(all_answers)):
            all_answers = [x for x in likert_order if x in all_answers]

        dist_rows = []

        for col in pred_cols:
            counts = (
                df[col]
                .value_counts(normalize=True)
                .reindex(all_answers, fill_value=0)
            )

            row = {"Model": col.replace("_pred", "")}

            for ans in all_answers:
                row[ans] = counts[ans]

            dist_rows.append(row)

        dist_df = pd.DataFrame(dist_rows)

        # Sort base models first
        dist_df["TypeOrder"] = dist_df["Model"].apply(
            lambda x: 0 if "base" in x.lower() else 1
        )

        dist_df = (
            dist_df
            .sort_values(["TypeOrder", "Model"])
            .drop(columns="TypeOrder")
        )

        # Define valid responses
        valid_answers = {
            "A", "A.",
            "B", "B.",
            "C", "C.",
            "D", "D.",
            "E", "E."
        }

        valid_cols = [a for a in all_answers if a in valid_answers]
        invalid_cols = [a for a in all_answers if a not in valid_answers]

        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(18, 8))

        bottom = np.zeros(len(dist_df))

        if granular:

            green_palette = sns.color_palette(
                "Greens",
                n_colors=max(len(valid_cols) + 2, 3)
            )[2:]

            red_palette = sns.color_palette(
                "Reds",
                n_colors=max(len(invalid_cols) + 2, 3)
            )[2:]

            color_map = {}

            for ans, color in zip(valid_cols, green_palette):
                color_map[ans] = color

            for ans, color in zip(invalid_cols, red_palette):
                color_map[ans] = color

            ordered_answers = valid_cols + invalid_cols

            for ans in ordered_answers:

                vals = dist_df[ans].values

                ax.bar(
                    dist_df["Model"],
                    vals,
                    bottom=bottom,
                    color=color_map[ans],
                    edgecolor="white",
                    linewidth=0.5
                )

                bottom += vals

            legend_handles = []

            if len(valid_cols) > 0:
                legend_handles.append(
                    Patch(
                        facecolor=green_palette[-1],
                        label="Valid"
                    )
                )

            if len(invalid_cols) > 0:
                legend_handles.append(
                    Patch(
                        facecolor=red_palette[-1],
                        label="Invalid"
                    )
                )

        else:

            valid_vals = dist_df[valid_cols].sum(axis=1).values
            invalid_vals = dist_df[invalid_cols].sum(axis=1).values

            valid_color = "forestgreen"
            invalid_color = "firebrick"

            ax.bar(
                dist_df["Model"],
                valid_vals,
                bottom=bottom,
                color=valid_color,
                edgecolor="white",
                linewidth=0.5,
                label="Valid"
            )

            bottom += valid_vals

            ax.bar(
                dist_df["Model"],
                invalid_vals,
                bottom=bottom,
                color=invalid_color,
                edgecolor="white",
                linewidth=0.5,
                label="Invalid"
            )

            legend_handles = [
                Patch(facecolor=valid_color, label="Valid"),
                Patch(facecolor=invalid_color, label="Invalid")
            ]

        ax.set_ylim(0, 1)

        ax.set_ylabel("Prediction Proportion")
        ax.set_xlabel("Model")

        title_suffix = (
            "Granular"
            if granular
            else "General"
        )

        ax.set_title(
            f"{title_suffix} Prediction Distributions "
            f"({split.capitalize()} Set)"
        )

        plt.xticks(rotation=45, ha="right")

        ax.legend(
            handles=legend_handles,
            title="Prediction Type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.tight_layout()

        if save_filename:

            self.results_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            save_path = (
                self.results_folder / save_filename
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            if self.verbose:
                print(
                    f"Saved prediction distribution plot to: "
                    f"{save_path}"
                )

        if show:
            plt.show()

        plt.close()


    def prediction_distribution_analysis(
        self,
        split: str = "test",
        show_plots: bool = False,
        granular: bool = True,
    ):
        """
        Main entry point for prediction
        collapse visualization.
        """

        if self.verbose:
            print("Generating predictio distribution analysis...")

        tabulars = self.load_tabulars(
            splits=[split]
        )

        if not tabulars:
            print("No tabulars loaded.")
            return

        suffix = (
            "granular"
            if granular
            else "general"
        )

        self._prediction_distribution_plot(
            tabulars_dict=tabulars,
            split=split,
            save_filename=(
                f"{self.train_plan}_"
                f"prediction_distribution_"
                f"{suffix}.png"
            ),
            show=show_plots,
            granular=granular
        )

    # -----------------------------
    # Output Distribution Alignment
    # -----------------------------
    def _distribution_heatmap(
        self,
        df: pd.DataFrame,
        split: str,
        score: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        Heatmap of model performance across demographics.
        Lower KL/WD = better.
        """

        sns.set_style("white")

        subdf = df[df["Split"] == split].copy()

        avg_scores = (
            subdf
            .groupby(["Model", "Demographic"])[score]
            .mean()
            .reset_index()
        )

        pivot = avg_scores.pivot(
            index="Model",
            columns="Demographic",
            values=score
        )

        plt.figure(figsize=(12, 7))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            linewidths=0.5,
            cbar_kws={"label": score}
        )

        plt.title(
            f"{score} Across Models and Demographics "
            f"({split.capitalize()} Set)"
        )

        plt.xlabel("Demographic")
        plt.ylabel("Model")

        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)

            save_path = folder_path / save_filename

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            if self.verbose:
                print(f"Saved heatmap to: {save_path}")

        if show:
            plt.show()

        plt.close()

    def _distribution_heatmap_best(
        self,
        df: pd.DataFrame,
        split: str,
        score: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        Heatmap of model performance across demographics,
        with the best (lowest) score per demographic circled
        and the corresponding model label bolded.
        """
        sns.set_style("white")

        subdf = df[df["Split"] == split].copy()

        avg_scores = (
            subdf
            .groupby(["Model", "Demographic"])[score]
            .mean()
            .reset_index()
        )

        pivot = avg_scores.pivot(
            index="Model",
            columns="Demographic",
            values=score
        )

        # find best model (row) for each demographic (col)
        best_row_per_col = pivot.idxmin(axis=0)  # Series: demographic -> best model

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            linewidths=0.5,
            cbar_kws={"label": score},
            ax=ax
        )

        # draw rectangle around best cell per demographic
        for col_idx, demographic in enumerate(pivot.columns):
            best_model = best_row_per_col[demographic]
            row_idx = pivot.index.get_loc(best_model)
            ax.add_patch(plt.Rectangle(
                (col_idx, row_idx),       # (x, y) = (col, row) in heatmap coords
                1, 1,                      # width, height
                fill=False,
                edgecolor='red',
                linewidth=2.5,
                clip_on=False
            ))

        # bold y-tick labels for models that win at least one demographic
        winning_models = set(best_row_per_col.values)
        yticklabels = ax.get_yticklabels()
        for label in yticklabels:
            if label.get_text() in winning_models:
                label.set_fontweight('bold')
        ax.set_yticklabels(yticklabels)

        ax.set_title(
            f"{score} Across Models and Demographics "
            f"({split.capitalize()} Set) — Best per Demographic Highlighted"
        )
        ax.set_xlabel("Demographic")
        ax.set_ylabel("Model")

        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)
            save_path = folder_path / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbose:
                print(f"Saved best heatmap to: {save_path}")

        if show:
            plt.show()

        plt.close()

    def _distribution_violinplot(
        self,
        df: pd.DataFrame,
        split: str,
        score: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        Violin plot showing distributional
        performance spread across demographics.
        """

        sns.set_style("whitegrid")

        subdf = df[df["Split"] == split].copy()

        plt.figure(figsize=(14, 7))

        sns.violinplot(
            data=subdf,
            x="Model",
            y=score,
            inner="box",
            cut=0
        )

        plt.xticks(rotation=45, ha="right")

        plt.title(
            f"{score} Distribution Across Demographics "
            f"({split.capitalize()} Set)"
        )

        plt.ylabel(score)
        plt.xlabel("Model")

        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)

            save_path = folder_path / save_filename

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            if self.verbose:
                print(f"Saved violin plot to: {save_path}")

        if show:
            plt.show()

        plt.close()

    def _distribution_pareto_plot(
        self,
        df: pd.DataFrame,
        split: str,
        score: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        Pareto-style plot:
        x-axis = mean metric
        y-axis = std deviation

        Lower-left = best overall + most stable.
        """

        sns.set_style("whitegrid")

        subdf = df[df["Split"] == split].copy()

        grouped = (
            subdf
            .groupby("Model")[score]
            .agg(["mean", "std"])
            .reset_index()
        )

        plt.figure(figsize=(9, 7))

        sns.scatterplot(
            data=grouped,
            x="mean",
            y="std",
            s=150
        )

        for _, row in grouped.iterrows():
            plt.text(
                row["mean"],
                row["std"],
                row["Model"],
                fontsize=8,
                ha="left",
                va="bottom"
            )

        plt.xlabel(f"Mean {score}")
        plt.ylabel(f"Std Dev {score}")

        plt.title(
            f"Pareto Frontier: Fidelity vs Stability "
            f"({split.capitalize()} Set)"
        )

        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)

            save_path = folder_path / save_filename

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            if self.verbose:
                print(f"Saved pareto plot to: {save_path}")

        if show:
            plt.show()

        plt.close()

    def _best_model_plot(
        self,
        df: pd.DataFrame,
        split: str,
        score: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        For each demographic group, selects the model with the lowest
        (best) score and plots the results as a bar chart, color-coded
        by model name.
        """
        subdf = df[df['Split'] == split].copy()
        avg_scores = subdf.groupby(['Demographic', 'Model'])[score].mean().reset_index()
        best_models = avg_scores.loc[
            avg_scores.groupby('Demographic')[score].idxmin()
        ]
        best_models = best_models.sort_values(by=score, ascending=False)
        best_scores = best_models[score]

        unique_models = best_models['Model'].unique()
        colors_palette = plt.cm.get_cmap('tab10', len(unique_models))
        model_color_map = {m: colors_palette(i) for i, m in enumerate(unique_models)}
        bar_colors = [model_color_map[m] for m in best_models['Model']]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(
            best_models['Demographic'],
            best_models[score],
            color=bar_colors,
            edgecolor='black'
        )

        for bar, model_score in zip(bars, best_scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{model_score:.3f}',
                va='bottom', ha='center',
                fontsize=10
            )

        handles = [
            Patch(facecolor=model_color_map[m], label=m)
            for m in unique_models
        ]
        plt.legend(handles=handles, title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 12}, title_fontsize=12)

        plt.title(f'Best Model per Demographic Group ({self.train_plan})', fontsize=14)
        plt.xlabel('Demographic Group', fontsize=12)
        plt.ylabel(f'Average {score}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)
            save_path = folder_path / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved best-model plot to: {save_path}")

        if show:
            plt.show()

        plt.close()

    def distributional_alignment_analysis(
        self,
        split: str = "test",
        score: str = "WD_Weighted",
        show_plots: bool = False
    ):
        """
        Generates:
        - Heatmap
        - Violin plot
        - Pareto frontier plot

        using distributional accuracy metrics.
        """

        metrics_path = self.results_folder / Path("distributional_accuracy/summary_demog_metrics.csv")
        granular_metrics_path = self.results_folder / Path("distributional_accuracy/aggregated_kl_metrics.csv") # it says kl but it includes wd that's my bad

        if not metrics_path.exists():
            raise FileNotFoundError(
                "summary_demog_metrics.csv not found. "
                "Run distributional_accuracy() first."
            )
        if not granular_metrics_path.exists():
            raise FileNotFoundError(
                "aggregated_kl_metrics.csv not found. "
                "Run distributional_accuracy() first."
            )

        df = pd.read_csv(metrics_path)
        gran_df = pd.read_csv(granular_metrics_path)

        self._best_model_plot(
            df=df,
            split=split,
            score=score,
            save_filename=f"{self.train_plan}_{score}_best_model.png",
            show=show_plots
        )

        self._distribution_heatmap(
            df=df,
            split=split,
            score=score,
            save_filename=f"{self.train_plan}_{score}_heatmap.png",
            show=show_plots
        )

        self._distribution_heatmap_best(
            df=df,
            split=split,
            score=score,
            save_filename=f"{self.train_plan}_{score}_heatmap_best.png",
            show=show_plots
        )

        self._distribution_violinplot(
            df=gran_df,
            split=split,
            score=score,
            save_filename=f"{self.train_plan}_{score}_violin.png",
            show=show_plots
        )

        self._distribution_pareto_plot(
            df=gran_df,
            split=split,
            score=score,
            save_filename=f"{self.train_plan}_{score}_pareto.png",
            show=show_plots
        )

    # ----------------------------
    # Metric Agreement Analysis
    # ----------------------------
    def _rank_scatter_plot(
        self,
        df: pd.DataFrame,
        split: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        Scatter plot of average KL rank vs average WD rank per model.
        Points on the diagonal = metrics agree on model standing.
        Points off diagonal = metric-sensitive ranking.
        Size of point = number of demographics the model wins under either metric.
        """
        subdf = df[df['Split'] == split].copy()

        # compute per-demographic ranks for each model
        for metric in ['KL_Weighted', 'WD_Weighted']:
            subdf[f'{metric}_rank'] = subdf.groupby('Demographic')[metric].rank(
                ascending=True, method='min'
            )

        avg_ranks = (
            subdf
            .groupby('Model')[['KL_Weighted_rank', 'WD_Weighted_rank']]
            .mean()
            .reset_index()
        )

        # count demographics won under either metric
        kl_wins = subdf.loc[subdf.groupby('Demographic')['KL_Weighted'].idxmin(), 'Model'].value_counts()
        wd_wins = subdf.loc[subdf.groupby('Demographic')['WD_Weighted'].idxmin(), 'Model'].value_counts()
        total_wins = kl_wins.add(wd_wins, fill_value=0).reset_index()
        total_wins.columns = ['Model', 'wins']
        avg_ranks = avg_ranks.merge(total_wins, on='Model', how='left').fillna({'wins': 0})

        unique_models = avg_ranks['Model'].unique()
        colors_palette = plt.cm.get_cmap('tab10', len(unique_models))
        model_color_map = {m: colors_palette(i) for i, m in enumerate(unique_models)}

        sns.set_style("whitegrid")
        plt.figure(figsize=(9, 7))

        for _, row in avg_ranks.iterrows():
            plt.scatter(
                row['KL_Weighted_rank'],
                row['WD_Weighted_rank'],
                color=model_color_map[row['Model']],
                s=100 + row['wins'] * 40,
                edgecolors='black',
                linewidths=0.5,
                zorder=3
            )
            plt.text(
                row['KL_Weighted_rank'] + 0.05,
                row['WD_Weighted_rank'] + 0.05,
                row['Model'],
                fontsize=7,
                ha='left',
                va='bottom'
            )

        # diagonal = perfect agreement
        lims = [1, len(unique_models)]
        plt.plot(lims, lims, linestyle='--', color='gray', linewidth=1, label='Perfect agreement')

        plt.xlabel('Average KL Divergence Rank (lower = better)', fontsize=11)
        plt.ylabel('Average WD Rank (lower = better)', fontsize=11)
        plt.title(
            f'KL vs WD Model Rankings ({split.capitalize()} Set)\n'
            f'Point size = total demographic wins across both metrics',
            fontsize=12
        )
        plt.legend(fontsize=9)
        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)
            save_path = folder_path / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved rank scatter to: {save_path}")

        if show:
            plt.show()

        plt.close()


    def _alluvial_plot(
        self,
        df: pd.DataFrame,
        split: str,
        save_filename: str = None,
        show: bool = False
    ):
        """
        Alluvial/flow diagram comparing best model per demographic
        under KL Divergence vs Wasserstein Distance.
        Each demographic is a row. Left column = KL best model,
        right column = WD best model. Crossing bands = metric disagreement.
        """
        subdf = df[df['Split'] == split].copy()

        kl_best = (
            subdf
            .loc[subdf.groupby('Demographic')['KL_Weighted'].idxmin()]
            [['Demographic', 'Model']]
            .rename(columns={'Model': 'KL_Best'})
        )
        wd_best = (
            subdf
            .loc[subdf.groupby('Demographic')['WD_Weighted'].idxmin()]
            [['Demographic', 'Model']]
            .rename(columns={'Model': 'WD_Best'})
        )
        flow_df = kl_best.merge(wd_best, on='Demographic').reset_index(drop=True)

        demographics = flow_df['Demographic'].tolist()
        n = len(demographics)

        unique_models = sorted(
            set(flow_df['KL_Best'].tolist() + flow_df['WD_Best'].tolist())
        )
        colors_palette = plt.cm.get_cmap('tab10', len(unique_models))
        model_color_map = {m: colors_palette(i) for i, m in enumerate(unique_models)}

        fig, ax = plt.subplots(figsize=(11, max(6, n * 0.9)))

        y_positions = {d: (n - 1 - i) for i, d in enumerate(demographics)}

        bar_width = 0.08
        left_x = 0.2
        right_x = 0.8

        for demog, y in y_positions.items():
            row = flow_df[flow_df['Demographic'] == demog].iloc[0]
            kl_model = row['KL_Best']
            wd_model = row['WD_Best']
            agrees = kl_model == wd_model

            band_color = model_color_map[kl_model]
            alpha = 0.5 if agrees else 0.35

            # control points for smooth cubic bezier band
            band_height = 0.3
            xs = np.linspace(left_x + bar_width, right_x, 100)
            t = (xs - (left_x + bar_width)) / (right_x - (left_x + bar_width))
            y_left = y
            y_right = y_positions[demog]  # same row, but visually connects models

            y_top = (1 - t)**3 * (y_left + band_height/2) + 3*(1-t)**2*t * (y_left + band_height/2) + \
                    3*(1-t)*t**2 * (y_right + band_height/2) + t**3 * (y_right + band_height/2)
            y_bot = (1 - t)**3 * (y_left - band_height/2) + 3*(1-t)**2*t * (y_left - band_height/2) + \
                    3*(1-t)*t**2 * (y_right - band_height/2) + t**3 * (y_right - band_height/2)

            ax.fill_between(xs, y_bot, y_top, color=band_color, alpha=alpha)

            # left node (KL best)
            ax.barh(y, bar_width, left=left_x, height=0.5,
                    color=model_color_map[kl_model], edgecolor='black', linewidth=0.5)

            # right node (WD best)
            ax.barh(y, bar_width, left=right_x, height=0.5,
                    color=model_color_map[wd_model], edgecolor='black', linewidth=0.5)

            # disagreement marker
            if not agrees:
                ax.text(
                    0.5, y + 0.28,
                    '✗',
                    ha='center', va='bottom',
                    fontsize=10, color='firebrick', fontweight='bold'
                )

            # demographic label center
            ax.text(0.5, y, demog, ha='center', va='center', fontsize=9, fontweight='500')

            # model name labels
            ax.text(left_x - 0.01, y, kl_model, ha='right', va='center', fontsize=7)
            ax.text(right_x + bar_width + 0.01, y, wd_model, ha='left', va='center', fontsize=7)

        # column headers
        ax.text(left_x + bar_width/2, n, 'KL Divergence\nBest Model',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(right_x + bar_width/2, n, 'Wasserstein Distance\nBest Model',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # legend
        handles = [
            Patch(facecolor=model_color_map[m], label=m, edgecolor='black', linewidth=0.5)
            for m in unique_models
        ]
        ax.legend(
            handles=handles, title='Model', loc='lower center',
            bbox_to_anchor=(0.5, -0.12), ncol=2,
            prop={'size': 8}, title_fontsize=9
        )

        ax.set_xlim(0, 1.1)
        ax.set_ylim(-0.8, n + 0.3)
        ax.axis('off')
        ax.set_title(
            f'Best Model per Demographic: KL vs WD Agreement ({split.capitalize()} Set)\n'
            f'✗ = metric disagreement',
            fontsize=12, pad=12
        )

        plt.tight_layout()

        if save_filename:
            folder_path = Path(f"{self.graphs_folder}/{split}")
            folder_path.mkdir(parents=True, exist_ok=True)
            save_path = folder_path / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved alluvial plot to: {save_path}")

        if show:
            plt.show()

        plt.close()


    def metric_agreement_analysis(
        self,
        split: str = 'test',
        show_plots: bool = False
    ):
        """
        Generates:
        - Rank scatter plot (KL rank vs WD rank per model)
        - Alluvial flow diagram (best model per demographic under KL vs WD)

        Requires distributional_accuracy() to have been run first.
        """
        metrics_path = (
            self.results_folder
            / 'distributional_accuracy'
            / 'summary_demog_metrics.csv'
        )

        if not metrics_path.exists():
            raise FileNotFoundError(
                "summary_demog_metrics.csv not found. "
                "Run distributional_accuracy() first."
            )

        df = pd.read_csv(metrics_path)

        self._rank_scatter_plot(
            df=df,
            split=split,
            save_filename=f"{self.train_plan}_kl_vs_wd_rank_scatter.png",
            show=show_plots
        )

        self._alluvial_plot(
            df=df,
            split=split,
            save_filename=f"{self.train_plan}_kl_vs_wd_alluvial.png",
            show=show_plots
        )
