import json
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from pathlib import Path
import os
from datetime import datetime

# --- Configuration ---
TRAIN_PLAN = "presidents_to_abortion"
TRAIN_PATH = Path(f"calyapo/data/final/{TRAIN_PLAN}_train.jsonl")
VAL_PATH = Path(f"calyapo/data/final/{TRAIN_PLAN}_val.jsonl")
OUTPUT_FOLDER = Path(f"inference_outputs/{TRAIN_PLAN}")
LORA_ADAPTER_PATH = Path(f"calyapo/training/checkpoints/{TRAIN_PLAN}_dataset")

TP_ABBREVIATIONS = {
        "presidents_to_abortion" : "p2a"
    }

def get_timestamp():
    """Returns current time as a string: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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

def run_inference(engine_params, sampling_params, split, train_plan, input_path, output_folder, lora_path = None, verbose=False):
    if not os.path.exists(input_path):
        raise ValueError(f"Input path '{input_path}' does not exist")

    raw_data = load_data(input_path)
    if not raw_data:
        return
        
    prompts = [item["prompt"] for item in raw_data]
    print(f"Loaded {len(prompts)} prompts.")

    if verbose: 
        model_name = engine_params.get('model', 'Unknown')
        print(f"\n------------------------Dataset Stats------------------------")
        print(f"Dataset:                 {split}")
        print(f"Training Plan:           {train_plan}")
        print(f"Plan using Abbreviation: {TP_ABBREVIATIONS.get(train_plan, 'no abbreviations found')}")
        print(f"-------------------------------------------------------------")
        
        print(f"\n------------------------Engine Stats------------------------")
        print(f"Initializing vLLM engine for model: '{model_name}'")
        print(f"quantization:            {engine_params.get('quantization', None)}")
        print(f"max_model_len:           {engine_params.get('max_model_len', None)}")
        print(f"max_num_seqs:            {engine_params.get('max_num_seqs', None)}")
        print(f"gpu_memory_utilization:  {engine_params.get('gpu_memory_utilization', None)}")
        print(f"LoRA enabled:            {engine_params.get('enable_lora', None)}")
        print(f"seed:                    {engine_params.get('seed', None)}")
        print(f"-------------------------------------------------------------")

        print(f"\n------------------------Sampler Stats------------------------")
        print(f"temperature:             {sampling_params.get('temperature', None)}")
        print(f"max_tokens:              {sampling_params.get('max_tokens', None)}")
        print(f"logprobs:                {sampling_params.get('logprobs', None)}")
        print(f"-------------------------------------------------------------")
    llm = LLM(**engine_params)

    vllm_sampling_config = SamplingParams(**sampling_params)

    print("Starting batch inference...")
    if engine_params.get('enable_lora', False):
        if not os.path.exists(lora_path):
            raise f"LoRA is enabled but inputted LoRA path '{lora_path}' does not exist"
        print("LoRA model detected")
        lora_request = LoRARequest("my_finetuned_model", 1, lora_path)
        outputs = llm.generate(prompts, vllm_sampling_config, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, vllm_sampling_config)

    ts = get_timestamp()
    model_type = "lora" if engine_params.get('enable_lora', False) else "base"
    results_file = output_folder / f"results_{split}_{TP_ABBREVIATIONS[train_plan]}_{model_type}_{ts}.jsonl"
    config_file = output_folder / f"config_{split}_{TP_ABBREVIATIONS[train_plan]}_{model_type}_{ts}.json"
    full_config = {
        "timestamp": ts,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
        "input_dataset": str(input_path),
        "lora_path": lora_path
    }
    with open(config_file, "w") as cf:
        json.dump(full_config, cf, indent=4)
    print(f"Config saved to: {config_file}")

    with open(results_file, "w") as f:
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
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    USE_LORA = True
    USE_VAL = True

    basic_inf_engine_config = {
        "model": "meta-llama/Llama-2-7b-hf",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "dtype": "float16",
        "max_model_len": 212, # prompts are not that long
        "max_num_seqs": 96,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True, 
        "seed": 42
    }

    lora_inf_engine_config = {**basic_inf_engine_config, "enable_lora": True, "max_loras": 1}

    sampling_config = {
        "temperature": 0,
        "max_tokens": 2, # only need to generate one response (A, B, C or D) but give some flexibility
        "logprobs": 5 
    }

    if USE_LORA:
        engine_config = lora_inf_engine_config
        lora_path = str(LORA_ADAPTER_PATH)
    else:
        engine_config = basic_inf_engine_config
        lora_path = None

    if USE_VAL:
        inf_split = "validation"
        input_path = VAL_PATH
    else:
        inf_split = "training"
        input_path = TRAIN_PATH

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    run_inference(
        engine_params=engine_config, 
        sampling_params=sampling_config, 
        split=inf_split, 
        train_plan='presidents_to_abortion', 
        input_path=input_path,
        output_folder=OUTPUT_FOLDER, 
        lora_path=lora_path, 
        verbose=True
    )