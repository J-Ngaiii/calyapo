import pandas as pd
from pathlib import Path
import argparse

def run_analysis(csv_path, verbose=False):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    split_name = csv_path.stem.split('_')[1] # Extracts 'train', 'val', or 'test'

    correct_cols = [c for c in df.columns if c.endswith('_correct')]
    
    print("\n--- General Model Accuracies ---")
    for col in correct_cols:
        acc = df[col].mean()
        print(f"{col.replace('_correct', ''):<30}: {acc:.2%}")

    meta_cols = ['dataset_date', 'topic', 'answer', 'Question'] + correct_cols
    demog_cols = [c for c in df.columns if c not in meta_cols and not c.startswith('Unnamed')]

    unique_topics = df['topic'].unique()

    for topic in unique_topics:
        print(f"\n\n[ Topic: {topic} ]")
        topic_df = df[df['topic'] == topic]
        
        for demog in demog_cols:
            # Crosstab of Demographic vs Answer Choice
            ct = pd.crosstab(topic_df[demog], topic_df['answer'], normalize='index')
            
            print(f"\nBreakdown by {demog}:")
            print((ct * 100).round(1).astype(str) + '%')
            print("-" * 30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_plan", type=str, default='opinion_school')
    parser.add_argument("--data_folder", type=str, default='evaluation_datasets')
    args = parser.parse_args()

    base_path = Path(args.data_folder)
    
    # Iterate through the three standard files your previous script generated
    for split in ['train', 'val', 'test']:
        file_name = f"{args.train_plan}_{split}_eval_data.csv"
        run_analysis(base_path / file_name)

if __name__ == "__main__":
    main()