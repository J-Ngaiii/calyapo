import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from calyapo.utils.persistence import file_loader, file_saver

def load_evaluation_datasets(folder_path, train_plan):
    """
    Loads the train, val, and test combined CSVs into a dictionary.

    Combined eval datasets have the following schema: 
        dataset_date (int): date encoding from calyapo data transformation pipeline	
        topic (str): variable_label from calyapo data transformation pipeline
        answer (char): correct answer choice
        Age	(strP): age range of respondent as encoded in calyapo transformation pipeline
        Party Identity (str): party id of respondent as encoded in calyapo transformation pipeline	
        Political Ideology (str): political ideology of respondent as encoded in calyapo transformation pipeline
        Race (str): race of respondent as encoded in calyapo transformation pipeline
        Gender Identity	(str): gender of respondent as encoded in calyapo transformation pipeline
        Biological Sex	(str): sex of respondent as encoded in calyapo transformation pipeline
        Residence Urbanicity (str): enviornment of respondent, varies between urban, suburb and rural.
        Marital Status (str): Marital status of respondent as encoded in calyapo transformation pipeline

        Llama-3.1-8B_lora_pred (char): Prediction of model for that respondent
        Llama-3.1-8B_lora_correct (bool): Correctness of model prediction for that respondent 
        Llama-3.1-8B_base_pred (char): Prediction of model for that respondent	
        Llama-3.1-8B_base_correct (bool): Correctness of model prediction for that respondent 	
        Llama-3.1-8B-Instruct_lora_pred (char): Prediction of model for that respondent	
        Llama-3.1-8B-Instruct_lora_correct (bool): Correctness of model prediction for that respondent 	
        Llama-3.1-8B-Instruct_base_pred (char): Prediction of model for that respondent	
        Llama-3.1-8B-Instruct_base_correct (bool): Correctness of model prediction for that respondent 

        Llama-3.2-3B_lora_pred (char): Prediction of model for that respondent	
        Llama-3.2-3B_lora_correct (bool): Correctness of model prediction for that respondent 	
        Llama-3.2-3B_base_pred (char): Prediction of model for that respondent	
        Llama-3.2-3B_base_correct (bool): Correctness of model prediction for that respondent 	
        Llama-3.2-3B-Instruct_lora_pred (char): Prediction of model for that respondent	
        Llama-3.2-3B-Instruct_lora_correct (bool): Correctness of model prediction for that respondent 	
        Llama-3.2-3B-Instruct_base_pred (char): Prediction of model for that respondent	
        Llama-3.2-3B-Instruct_base_correct (bool): Correctness of model prediction for that respondent 
    """
    splits = ['train', 'val', 'test']
    data = {}
    for split in splits:
        file_name = f"{train_plan}_{split}_combined_eval.csv"
        file_path = Path(folder_path) / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Missing expected CSV: {file_path}")
            
        data[split] = file_loader(in_path=file_name, data_type='csv')
    return data

def get_accuracy_report(all_results, model_names):
    """
    Aggregates accuracy metrics for all models across traun, val and test splits.
    """
    report_list = []
    splits = ['train', 'val', 'test']
    
    for model in model_names:
        for split in splits:
            df = all_results[split]
            base_col = f"{model}_base_correct"
            lora_col = f"{model}_lora_correct"
            
            if base_col in df.columns:
                report_list.append({
                    'Model_Name': model,
                    'Split': split.capitalize(),
                    'Type': 'BASE',
                    'Accuracy': df[base_col].mean()
                })
            
            if lora_col in df.columns:
                report_list.append({
                    'Model_Name': model,
                    'Split': split.capitalize(),
                    'Type': 'LORA',
                    'Accuracy': df[lora_col].mean()
                })
    
    # creates up to 24 entries (4 llama models base and lora versions each getting an entry for each of the three splits)
    return pd.DataFrame(report_list)

def plot_performance(df, save_path=None):
    """
    Renders 2x2 grid using the Orange/DodgerBlue palette.
    """
    sns.set_style("whitegrid")
    palette = {"LORA": "orange", "BASE": "dodgerblue"}
    
    models = df['Model_Name'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        model_df = df[df['Model_Name'] == model]
        
        bar = sns.barplot(
            data=model_df, 
            x="Split", 
            y="Accuracy", 
            hue="Type", 
            palette=palette, 
            ax=ax, 
            alpha=0.8
        )
        
        ax.set_title(f"Performance: {model}")
        ax.set_ylabel("Accuracy")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
            
        if i != 0:
            ax.get_legend().remove()

    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CSV-based model results.")
    parser.add_argument("--train_plan", type=str, default='opinion_school', help="Training plan prefix.")
    args = parser.parse_args()

    LLAMA_MODELS = [
        'Llama-3.1-8B', 
        'Llama-3.1-8B-Instruct', 
        'Llama-3.2-3B', 
        'Llama-3.2-3B-Instruct'
    ]

    INPUT_FOLDER = Path("inference_outputs") / args.train_plan / "evaluation_datasets"
    OUTPUT_FOLDER = Path("inference_outputs") / args.train_plan / "reports"
    SAVE_FILE = OUTPUT_FOLDER / f"{args.train_plan}_model_comparison_bars.png"

    try:
        datasets = load_evaluation_datasets(INPUT_FOLDER, args.train_plan)
        report_df = get_accuracy_report(datasets, LLAMA_MODELS)
        
        if not report_df.empty:
            plot_performance(report_df, save_path=SAVE_FILE)
        else:
            print("No matching accuracy columns found in CSVs.")
            
    except Exception as e:
        print(f"Error: {e}")