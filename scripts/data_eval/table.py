import argparse
import sys
import json
from pathlib import Path
from calyapo.data_eval.tabularizer import Tabularizer

def main():
    parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='presidents_to_abortion', help="Name of training plan.")
    parser.add_argument("--run_keyword", type=str, nargs='?', default='archon', help="Keyword for the report folder.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    LLAMA_SUBFOLDER = "meta-llama"
    QWEN_SUBFOLDER = "Qwen"
    LLAMA_MODELS_FINETUNED = [
        'Llama-3.1-8B', 
        'Llama-3.1-8B-Instruct', 
        'Llama-3.2-3B', 
        'Llama-3.2-3B-Instruct', 
    ]
    QWEN_MODELS_FINETUNED = [
        'Qwen2.5-14B', 
        'Qwen2.5-14B-Instruct' 
    ]

    llama_map = {model: Path(f"outputs_{args.run_keyword}") / Path(LLAMA_SUBFOLDER) / model for model in LLAMA_MODELS_FINETUNED }
    qwen_map = {model: Path(f"outputs_{args.run_keyword}") / Path(QWEN_SUBFOLDER) / model for model in QWEN_MODELS_FINETUNED }
    llama_map.update(qwen_map)
    model_map = llama_map
    tab = Tabularizer(
        train_plan=args.train_plan,
        keyword=args.run_keyword,
        root_path=".",
        debug=args.debug,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"--- Starting Tabularize Pipeline ---")
        print(f"Train Plan: {args.train_plan}")
        print(f"Keyword:    {args.run_keyword}")
    tab.run_pipeline(model_map=model_map)

if __name__ == "__main__":
    main()