import argparse
import json
import os
import matplotlib.pyplot as plt

def chart_instruct_vs_base(lookup, train_plan, run_keyword):
    results = {}

    for model_alias, file_id in lookup.items():
        file_name = file_id if file_id.endswith(".jsonl") else f"{file_id}.jsonl"
        path = f"inference_outputs/{train_plan}/outputs_{run_keyword}/{model_alias}/{file_name}"
        
        try:
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
            correct = sum(1 for entry in data if entry.get('is_correct') is True)
            acc = correct / len(data) if len(data) > 0 else 0
            results[model_alias] = acc
        except Exception as e:
            print(f"Skipping {model_alias}: {e}")

    families = {
        "Llama 3.2 (3B)": (results.get("llama_base"), results.get("llama_instruct")),
        "Qwen 2.5 (14B)": (results.get("qwen_base"), results.get("qwen_instruct"))
    }

    labels = list(families.keys())
    base_scores = [v[0] for v in families.values() if v[0] is not None]
    inst_scores = [v[1] for v in families.values() if v[1] is not None]
    
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], base_scores, width, label='Base', color='#95a5a6')
    rects2 = ax.bar([i + width/2 for i in x], inst_scores, width, label='Instruct', color='#3498db')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison: Base vs. Instruct Alignment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i in x:
        delta = inst_scores[i] - base_scores[i]
        color = 'green' if delta > 0 else 'red'
        ax.text(i, max(inst_scores[i], base_scores[i]) + 0.02, f"Δ: {delta:+.2%}", 
                ha='center', fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig("instruct_vs_base_comparison.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_plan", type=str, default='opinion_school')
    parser.add_argument("--run_keyword", type=str, default='archon')
    args = parser.parse_args()

    MODEL_LOOKUP = {
        "meta-llama/Llama-3.2-3B": "results_test_os_lora_20260503_192154",      
        "meta-llama/Llama-3.2-3B-Instruct": "results_test_os_lora_20260503_192254",  
        "Qwen/Qwen2.5-14B": "results_test_os_lora_20260503_201244",       
        "Qwen/Qwen2.5-14B-Instruct": "results_test_os_lora_20260503_201754",   
    }

    chart_instruct_vs_base(MODEL_LOOKUP, args.train_plan, args.run_keyword)