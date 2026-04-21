import pandas as pd
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from pathlib import Path
import argparse
from typing import Dict, Iterable

def calculate_metrics(p, q):
    # Normalize to ensure sum = 1
    p = p.astype(float) / (p.sum() + 1e-12)
    q = q.astype(float) / (q.sum() + 1e-12)

    # Laplace smoothing
    epsilon = 1e-6
    p = (p + epsilon) / (p + epsilon).sum()
    q = (q + epsilon) / (q + epsilon).sum()

    kl = entropy(p, q)
    wd = wasserstein_distance(p, q)
    
    return kl, wd

def run_analysis(df: pd.DataFrame, df_config: Dict, demographics: Iterable, models_info: Iterable, verbose: bool = False, debug: bool = False):
    results = []
    
    # Logic for crosstabs: comparing distribution vectors
    true_cols = [c for c in df.columns if c.startswith('true_')]
    option_choices = [c.replace('true_', '') for c in true_cols]
    demog_col_labels = df.columns[df_config['demog_col_idx']]

    if debug:
        print(f"(run_analysis | debug) demog_col_labels: {demog_col_labels}")
    for demog_col in demog_col_labels:
        for _, row in df.iterrows():
            subgroup = row[demog_col]

            if debug: 
                print(f"(run_analysis | debug) subgroup: {demog_col_labels}")

            if pd.isna(subgroup): continue 
            
            p_dist = row[true_cols].values.astype(float)

            if debug: 
                print(f"(run_analysis | debug) p_dist: {p_dist}")

            for model_id in models_info:
                m_cols = [f"{model_id}_{l}" for l in option_choices]

                if debug: 
                    print(f"(run_analysis | debug) mcol: {m_cols}, demog_col: {demog_col}")

                q_values = []
                for col in m_cols:
                    if col in df.columns:
                        q_values.append(row[col])
                    else:
                        # col missing from crosstab => the model failed to generate this choice entirely
                        q_values.append(0.0)
                
                q_dist = np.array(q_values, dtype=float)
                kl, wd = calculate_metrics(p_dist, q_dist)

                if debug: 
                    print(f"(run_analysis | debug) outputted kl: {kl}, outputted wd: {wd}")

                results.append({
                    "Demographic": demog_col,
                    "Subgroup": subgroup,
                    "Model": model_id,
                    "KL_Divergence": kl,
                    "Wasserstein_Distance": wd
                })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_plan", type=str, default="opinion_school")
    parser.add_argument("--demog_col_idx", type=int, nargs='+', default=[0])
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root_path = "inference_outputs" / Path(args.train_plan) / "reports" / "crosstabs"
    all_results = []

    # Iterate through split (train/test/val) -> question folder -> csv files
    # Glob pattern matches: split/question_folder/comparison_csv
    csv_files = list(root_path.glob("*/*/*_comparison.csv"))

    if not csv_files:
        print(f"No CSV files found in {root_path}")
        return

    for file_path in csv_files:
        split = file_path.parts[-3]    # e.g., 'train'
        question = file_path.parts[-2] # e.g., 'Joe_Biden_Favorability'
        
        df = pd.read_csv(file_path)
        assert df is not None, f"Invalid file path: '{file_path}'"
        demog_cols = df.columns[args.demog_col_idx].tolist()
        true_cols = [c for c in df.columns if c.startswith('true_')]
        potential_models = [c for c in df.columns if c not in true_cols and c not in demog_cols]
        models = sorted(list(set([m.rsplit('_', 1)[0] for m in potential_models if '_' in m])))

        df_config = {'demog_col_idx': args.demog_col_idx}
        
        if args.debug:
            print(f"(main | debug) Inputted df: {df}")

        file_results = run_analysis(df, df_config, demog_cols, models, verbose=args.verbose, debug=args.debug)

        if args.debug:
            print(f"(main | debug) Results of run_analysis: {file_results}")
        
        for res in file_results:
            res.update({
                "Split": split,
                "Question": question,
                "File_Source": file_path.name
            })
            all_results.append(res)

    if args.debug:
        print(all_results)
    final_df = pd.DataFrame(all_results)

    if args.output_folder is not None:
        out_path = Path(args.output_folder)
    else:
        out_path = "inference_outputs" / Path(args.train_plan) / "reports" / "distributional_accuracy"
    out_path.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(out_path / "aggregated_kl_metrics.csv", index=False)

    summary = final_df.groupby(['Split', 'Question', 'Model'])[['KL_Divergence']].mean()
    summary.to_csv(out_path / "summary_metrics.csv")

    print(f"Analysis complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()