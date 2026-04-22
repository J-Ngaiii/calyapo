from pathlib import Path
import json
import argparse
from collections import defaultdict

from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER

import json
from collections import defaultdict

def validate_jsonl_by_label(jsonl_path: str):
    """
    Counts how many entries exist for each specific question label in the JSONL.
    """
    stats = defaultdict(int)
    total_lines = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            total_lines += 1
            data = json.loads(line)
            prompt = data['prompt']
            
            # Logic: Extracts "Donald Trump Favorability" from 
            # "...Question (Donald Trump Favorability): Regardless..."
            if "Question (" in prompt and "):" in prompt:
                label = prompt.split("Question (")[1].split("):")[0]
                stats[label] += 1
            else:
                stats["Unknown/Other"] += 1

    print(f"--- JSONL Breakdown: {jsonl_path} ---")
    for label, count in sorted(stats.items()):
        print(f"  {label}: {count} / {100*count/total_lines}%")
    print(f"  TOTAL: {total_lines}")
    
    return stats

def primary_checker(train_plan):
    train_path = UNIVERSAL_FINAL_FOLDER / f"{train_plan}_train.jsonl"
    val_path = UNIVERSAL_FINAL_FOLDER / f"{train_plan}_val.jsonl"
    test_path = UNIVERSAL_FINAL_FOLDER / f"{train_plan}_test.jsonl"

    results = {}

    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f)
            results[name] = count
            print(f"{name.capitalize()} set found: {count} datapoints.")
            validate_jsonl_by_label(path)
        else:
            results[name] = 0
            print(f"{name.capitalize()} file not found at: {path}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Runs checks on final JSONL datafiles") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='test_plan', help="Name of training plan to check jsonl files for.")

    args = parser.parse_args()

    primary_checker(args.train_plan)
if __name__ == "__main__":
    main()