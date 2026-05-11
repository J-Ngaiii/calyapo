import argparse
import json
import os
import matplotlib.pyplot as plt

def chart_accuracy(lookup, train_plan, run_keyword, full_report=True):
    accuracies = {}
    
    # 1. Data Aggregation
    for model_name, file_id in lookup.items():
        # Clean up path logic (handling the .jsonl extension inconsistency)
        file_name = file_id if file_id.endswith(".jsonl") else f"{file_id}.jsonl"
        path = f"inference_outputs/{train_plan}/outputs_{run_keyword}/{model_name}/{file_name}"
        
        try:
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
            
            # Simple accuracy calculation: (sum of correct / total)
            # Adjust the key 'is_correct' based on your specific JSONL schema
            correct = sum(1 for entry in data if entry.get('is_correct') is True)
            total = len(data)
            acc = correct / total if total > 0 else 0
            accuracies[model_name] = acc
        except FileNotFoundError:
            print(f"Warning: File not found for {model_name} at {path}")

    if full_report:
        print("\n--- Full Accuracy Report ---")
        for model, score in accuracies.items():
            print(f"{model}: {score:.4f}")
        print("----------------------------\n")

    # 2. Plotting All Models
    names = list(accuracies.keys())
    scores = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color='skyblue')

    plt.title(f"Model Inference Test Set Accuracy")
    plt.ylabel("Model Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig("all_models_accuracy.png")
    plt.show()

    # 3. Reasoning (Qwen) vs Non-Reasoning (Llama) Comparison
    # Filter best models
    qwen_models = {k: v for k, v in accuracies.items() if "Qwen" in k}
    llama_models = {k: v for k, v in accuracies.items() if "Llama" in k}

    if qwen_models and llama_models:
        best_qwen = max(qwen_models, key=qwen_models.get)
        best_llama = max(llama_models, key=llama_models.get)
        
        comp_names = [f"Best Qwen\n({best_qwen.split('/')[-1]})", 
                      f"Best Llama\n({best_llama.split('/')[-1]})"]
        comp_scores = [qwen_models[best_qwen], llama_models[best_llama]]

        plt.figure(figsize=(6, 6))
        bars = plt.bar(comp_names, comp_scores, color=['orange', 'lightgreen'])
        for bar, score in zip(bars, comp_scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score:.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )
        plt.title("Reasoning vs. Non-Reasoning Test Set Accuracy")
        plt.ylabel("Model Accuracy")
        plt.ylim(0, 1.0) # Standardize scale for comparison
        plt.tight_layout()
        plt.savefig("reasoning_vs_non_reasoning.png")
        plt.show()

    # 4. Instruct vs Base Comparison

    def is_instruct(model_name: str) -> bool:
        return "Instruct" in model_name

    base_models = {k: v for k, v in accuracies.items() if not is_instruct(k)}
    instruct_models = {k: v for k, v in accuracies.items() if is_instruct(k)}

    if base_models and instruct_models:
        best_base = max(base_models, key=base_models.get)
        best_instruct = max(instruct_models, key=instruct_models.get)

        labels = [
            f"Best Base\n({best_base.split('/')[-1]})",
            f"Best Instruct\n({best_instruct.split('/')[-1]})"
        ]
        scores = [base_models[best_base], instruct_models[best_instruct]]

        plt.figure(figsize=(6, 6))
        bars = plt.bar(labels, scores, color=["steelblue", "darkorange"])

        plt.title("Best Base vs Best Instruct Model")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)

        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score:.3f}",
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.tight_layout()
        plt.savefig("best_base_vs_instruct.png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan.")
    parser.add_argument("--run_keyword", type=str, nargs='?', default='archon', help="Keyword corresponding with inference run.")
    parser.add_argument("--full_report", action="store_true", default=True, help="Print all accuracies.")
    
    args = parser.parse_args()
    
    # All these are test set inference outputs
    LOOKUP = {
        "opinion_school" : {
            "meta-llama/Llama-3.1-8B": "results_test_os_lora_20260508_005922", 
            "meta-llama/Llama-3.1-8B-Instruct": "results_test_os_lora_20260503_181527", 
            "meta-llama/Llama-3.2-3B": "results_test_os_lora_20260503_192154",
            "meta-llama/Llama-3.2-3B-Instruct": "results_test_os_lora_20260503_192254", 
            "Qwen/Qwen2.5-14B": "results_test_os_lora_20260503_201244", 
            "Qwen/Qwen2.5-14B-Instruct": "results_test_os_lora_20260503_201754", 
            }, 
        "presidents_to_abortion" : {
            "meta-llama/Llama-3.1-8B": "results_test_p2a_lora_20260507_233430", 
            "meta-llama/Llama-3.1-8B-Instruct": "results_test_p2a_lora_20260507_234607", 
            "meta-llama/Llama-3.2-3B": "results_test_p2a_lora_20260507_031806",
            "meta-llama/Llama-3.2-3B-Instruct": "results_test_p2a_lora_20260507_031630", 
            "Qwen/Qwen2.5-14B": "results_test_p2a_lora_20260507_234812", 
            "Qwen/Qwen2.5-14B-Instruct": "results_test_p2a_lora_20260507_234952", 
            }
    }

    chart_accuracy(
        lookup=LOOKUP[args.train_plan], 
        train_plan=args.train_plan, 
        run_keyword=args.run_keyword, 
        full_report=args.full_report
    )