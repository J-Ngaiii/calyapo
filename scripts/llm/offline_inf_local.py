import json
import torch
from vllm import LLM, SamplingParams
from pathlib import Path
import os

# --- Configuration ---
DATASET_PATH = Path("calyapo/data/final/presidents_to_abortion_train.jsonl")
OUTPUT_FOLDER = Path("inference_outputs")
OUTPUT_PATH = OUTPUT_FOLDER / "results_train_offline.jsonl"

def load_data(file_path):
    data = []
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return data
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main(engine_params, sampling_params, input_path, output_path, verbose=False):
    if not os.path.exists(input_path):
        raise ValueError(f"Input path '{input_path}' does not exist")

    raw_data = load_data(input_path)
    if not raw_data:
        return
        
    prompts = [item["prompt"] for item in raw_data]
    print(f"Loaded {len(prompts)} prompts.")

    if verbose: 
        model_name = engine_params.get('model', 'Unknown')
        quant = engine_params.get('quantization', 'None')
        print(f"Initializing vLLM engine for model: '{model_name}', quantization: '{quant}'")
    llm = LLM(**engine_params)

    vllm_sampling_config = SamplingParams(**sampling_params)

    print("Starting batch inference...")
    outputs = llm.generate(prompts, vllm_sampling_config)

    with open(output_path, "w") as f:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            true_label = raw_data[i].get("completion", "").strip()
            
            logprobs_data = output.outputs[0].logprobs
            
            result = {
                "index": i,
                "prediction": generated_text,
                "true_label": true_label,
                "is_correct": generated_text.startswith(true_label),
                "logprobs": str(logprobs_data) 
            }
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    engine_config = {
        "model": "meta-llama/Llama-2-7b-hf",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "dtype": "float16",
        "max_model_len": 256, # prompts are not that long
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True, 
        "seed": 42
    }

    sampling_config = {
        "temperature": 0,
        "max_tokens": 2, # only need to generate one response (A, B, C or D) but give some flexibility
        "logprobs": 5 
    }

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    main(engine_config, sampling_config, DATASET_PATH, OUTPUT_PATH, verbose=True)