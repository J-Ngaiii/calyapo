import pandas as pd
import json

def create_steering_prompt(row):
    # Convert your one-hot/categorical columns into a string representation
    # Example: row['race'] might be 1 for Hispanic, 2 for White, etc.
    
    demographics = []
    if row['party'] == 'Democrat': demographics.append("Democrat")
    elif row['party'] == 'Republican': demographics.append("Republican")
    
    if row['race'] == 'Hispanic': demographics.append("Hispanic")
    # ... add other features ...
    
    demo_str = ", ".join(demographics)
    return f"Context: A survey respondent with the following traits: {demo_str}.\nQuestion: {row['question_text']}\nAnswer:"

def preprocess_california_data():
    # 1. Load your datasets (PPIC, IGS, CES)
    df_ppic = pd.read_csv("path/to/ppic_2024.csv")
    # ... load others ...
    
    # 2. Filter for California (for national datasets like CES)
    # df_ces = df_ces[df_ces['state'] == 'CA']
    
    # 3. Combine and iterate
    data_to_save = []
    
    # Example iteration
    for index, row in df_ppic.iterrows():
        prompt = create_steering_prompt(row)
        
        # For Scheme 1, 'completion' is the SINGLE answer this specific person gave
        completion = row['response_text']  # e.g., "Strongly Agree"
        
        data_to_save.append({
            "prompt": prompt,
            "completion": completion
        })
        
    # 4. Save as JSONL
    with open("data/california_scheme1_train.jsonl", "w") as f:
        for entry in data_to_save:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    preprocess_california_data()