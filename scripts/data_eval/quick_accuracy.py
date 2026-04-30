import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def calculate_accuracy(results_path: str, split: str, model_name: str):
    data_labels = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                pred_correct = int(json_obj['is_correct'])
                data_labels.append(pred_correct)
    
    labels_array = np.array(data_labels)
    num_datapoints = len(data_labels)
    acc = np.mean(labels_array)
    print(f"{split} accuracy for model '{model_name}' (LoRA={'lora' in results_path.lower()}): {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan.")
    parser.add_argument("--split", type=str, nargs='?', default='train', help="Dataset partitin to analyze.")
    parser.add_argument("--run_keyword", type=str, nargs='?', default='archon', help="Keyword corresponding with inference run.")
    parser.add_argument("--model_idx", type=str, nargs='?', default='5', help="Integer assigned to a model.")
    parser.add_argument("--jsonl", type=str, nargs='?', default='results_test_os_base_20260428_024715.jsonl', help="Exact json file with results.")
    
    args = parser.parse_args()
    
    LOOKUP = {
        '1' : f"meta-llama/Llama-3.1-8B",
        '2' : f"meta-llama/Llama-3.1-8B-Instruct", 
        '3' : f"meta-llama/Llama-3.2-3B",
        '4' : f"meta-llama/Llama-3.2-3B-Instruct", 
        '5': f"Qwen/Qwen2.5-14B", 
        '6': f"Qwen/Qwen2.5-14B-Instruct", 
    }

    IN_PATH = f"inference_outputs/{args.train_plan}/outputs_{args.run_keyword}/{LOOKUP[args.model_idx]}/{args.jsonl}"
    calculate_accuracy(results_path=IN_PATH, split=args.split, model_name=LOOKUP[args.model_idx].split('/')[1])
