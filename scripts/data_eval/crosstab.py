import pandas as pd
from pathlib import Path
import argparse
import os

# Importing your specific persistence utilities
from calyapo.utils.persistence import file_loader, file_saver

def run_crosstab_analysis(train_plan: str, data_folder: str, output_folder: str, verbose: bool = False):
    base_in_path = Path(data_folder)
    base_out_path = Path(output_folder) / "crosstabs"
    
    # Define splits to process
    splits = ['train', 'val', 'test']
    
    for split in splits:
        file_name = f"{train_plan}_{split}_eval_data.csv"
        csv_path = base_in_path / file_name
        
        if not csv_path.exists():
            if verbose: print(f"Skipping {split}: {csv_path} not found.")
            continue

        # Use your file_loader utility
        df = file_loader(in_path=csv_path, data_type='csv', verbose=verbose)

        # 1. Identify Model Columns and Demographic Columns
        # Models are identified by the '_correct' suffix from your previous script
        model_cols = [c.replace('_correct', '') for c in df.columns if c.endswith('_correct')]
        
        # Demographics: exclude meta-data and the 'correct' flag columns
        exclude = ['dataset_date', 'topic', 'answer', 'Question', 'index', 'model_name'] + \
                  [c for c in df.columns if c.endswith('_correct')]
        demog_cols = [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]

        unique_topics = df['topic'].unique()

        for topic in unique_topics:
            # Create a clean string for the filename from the topic name
            safe_topic = "".join([c if c.isalnum() else "_" for c in topic])
            topic_df = df[df['topic'] == topic]

            for demog in demog_cols:
                # We create a crosstab showing respondent answers normalized by demographic group
                # This shows: "Of [Demog X], what % chose A, B, C, etc."
                ct_df = pd.crosstab(topic_df[demog], topic_df['answer'], normalize='index')
                
                # Convert to percentage for readability before saving
                ct_df = (ct_df * 100).round(2)

                # Construct output path: /output/plan/crosstabs/split/topic_by_demog.csv
                save_path = base_out_path / split / f"{safe_topic}" / f"by_{demog}.csv"
                
                # Use your file_saver utility
                file_saver(
                    out_path=save_path, 
                    data=ct_df.reset_index(), # reset_index ensures the demog label is a column, not just an index
                    data_type='csv', 
                    verbose=verbose
                )

def main():
    parser = argparse.ArgumentParser(description="Generate and save demographic crosstabs.")
    parser.add_argument("--train_plan", type=str, default='opinion_school')
    parser.add_argument("--verbose", action='store_true', default=True)

    args = parser.parse_args()

    DATA_FOLDER = Path("inference_outputs") / Path(args.train_plan) / "evaluation_datasets"
    OUTPUT_FOLDER = Path("inference_outputs") / Path(args.train_plan) / "reports"

    run_crosstab_analysis(
        train_plan=args.train_plan,
        data_folder=DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()