import pandas as pd
from pathlib import Path
import argparse
import os
from calyapo.utils.persistence import file_loader, file_saver

def run_crosstab_analysis(train_plan: str, data_folder: str, output_folder: str, verbose: bool = False):
    """
    Loads combined eval datasets with following schema: 
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

    Assembles crosstab responses to all unique topics found in the dataset based on demographic cols. 
    """
    base_in_path = Path(data_folder)
    base_out_path = Path(output_folder) / "crosstabs"

    splits = ['train', 'val', 'test']
    
    for split in splits:
        file_name = f"{train_plan}_{split}_combined_eval.csv"
        csv_path = base_in_path / file_name
        
        if not csv_path.exists():
            if verbose: print(f"Skipping {split}: {csv_path} not found.")
            continue

        df = file_loader(in_path=csv_path, data_type='csv', verbose=verbose)

        # to identify demographics exclude meta-data and the model cols
        exclude = ['dataset_date', 'topic', 'answer', 'Question', 'index', 'model_name'] + \
                  [c for c in df.columns if c.endswith('_correct') or c.endswith('_pred')]
        demog_cols = [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]

        unique_var_labels = df['topic'].unique() # collects all survey questions asked under the current split in this training plan
        model_pred_cols = sorted([c for c in df.columns if c.endswith('_pred')]) # identify model and demographic cols --> so we can crosstab on them later

        if verbose:
            print(f"Demographic columns identified: {demog_cols}")
            print(f"Topic Variable Labels identified: {list(unique_var_labels)}")
            print(f"Model Prediction cols identified: {model_pred_cols}")
        for topic_var_label in unique_var_labels:
            topic_label = "".join([c if c.isalnum() else "_" for c in topic_var_label]) # retain characters that are alphanumeric, otherwise replace with underscore so we can always construct a file name
            topic_df = df[df['topic'] == topic_var_label] # filter out to rows pertinent to just that variable label

            for demog in demog_cols:
                # calculate the true proportions first
                base_ct = pd.crosstab(topic_df[demog], topic_df['answer'], normalize='index')
                base_ct = (base_ct * 100).round(2)
                base_ct.columns = [f"true_{c}" for c in base_ct.columns] # mark the crosstab col values corresponding to true proportions accordingly
                
                # list to hold all DataFrames for this specific demographic
                all_cts = [base_ct]

                for mcol in model_pred_cols:
                    model_name = mcol.replace('_pred', '')
                    
                    # calculate model crosstab proportions
                    m_ct = pd.crosstab(topic_df[demog], topic_df[mcol], normalize='index')
                    m_ct = (m_ct * 100).round(2)
                    m_ct.columns = [f"{model_name}_{c}" for c in m_ct.columns] # mark accordingly
                    
                    all_cts.append(m_ct)

                # concatenate everything horizontally along rows 
                master_ct = pd.concat(all_cts, axis=1)
                combined_save_path = base_out_path / split / f"{topic_label}" / f"by_{demog}_comparison.csv"
                file_saver(
                    out_path=combined_save_path, 
                    data=master_ct.reset_index(),
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