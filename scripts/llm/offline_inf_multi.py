import json
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from pathlib import Path
import os
from datetime import datetime
import argparse
from typing import Any, Union

# --- Configuration ---
TP_ABBREVIATIONS = {
        "presidents_to_abortion" : "p2a", 
        "opinion_school" : "os", 
        "test_plan" : "test"
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

def run_inference(engine_params, sampling_params, train_plan, path_conf, verbose=False):
    
    model_name_path = engine_params.get('model', 'model')
    model_name = model_name_path.split('/')[-1]
    if verbose: 
        print(f"\n------------------------Engine Stats------------------------")
        print(f"Initializing vLLM engine for model: '{model_name}'")
        print(f"quantization:            {engine_params.get('quantization', None)}")
        print(f"max_model_len:           {engine_params.get('max_model_len', None)}")
        print(f"max_num_seqs:            {engine_params.get('max_num_seqs', None)}")
        print(f"gpu_memory_utilization:  {engine_params.get('gpu_memory_utilization', None)}")
        print(f"seed:                    {engine_params.get('seed', None)}")
        print(f"-------------------------------------------------------------")

        print(f"\n------------------------Sampler Stats------------------------")
        print(f"temperature:             {sampling_params.get('temperature', None)}")
        print(f"max_tokens:              {sampling_params.get('max_tokens', None)}")
        print(f"logprobs:                {sampling_params.get('logprobs', None)}")
        print(f"-------------------------------------------------------------")

    llm = LLM(**engine_params)
    vllm_sampling_config = SamplingParams(**sampling_params)

    model_types = ['lora', 'base']
    dataset_types = ['train', 'val', 'test']

    for modtype in model_types:

        if modtype == 'lora':
            lora_path = path_conf.get('lora_weights_folder', None)
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA is enabled but inputted LoRA path '{lora_path}' does not exist")
            lora_request = LoRARequest("my_finetuned_model", 1, lora_path)
            print("LoRA request successfully loaded")
        elif modtype == 'base':
            lora_request = None
        else:
            raise ValueError(f"Unkown model type: '{modtype}'")

        for splittype in dataset_types:
            print("Starting batch inference...")
            input_path = path_conf.get(splittype, None)
            raw_data = load_data(input_path)
            assert splittype in path_conf, f"Split type '{splittype}' not specified in path config: {path_conf}"
            assert input_path is not None, f"Split type '{splittype}' specified but path does not exist"
            assert raw_data is not None, f"No data found at input path '{input_path}'"

            if verbose: 
                print(f"\n------------------------Dataset Stats------------------------")
                print(f"Dataset:                 {splittype}")
                print(f"Number of Datapoints:    {len(raw_data)}")
                print(f"Training Plan:           {train_plan}")
                print(f"Plan using Abbreviation: {TP_ABBREVIATIONS.get(train_plan, 'no abbreviations found')}")
                print(f"-------------------------------------------------------------")

            prompts = [item["prompt"] for item in raw_data]
            print(f"Loaded {len(prompts)} prompts.")
            outputs = llm.generate(prompts, sampling_config, lora_request=lora_request)

            ts = get_timestamp()
            output_folder = path_conf.get('output_folder', None)
            assert 'output_folder' in path_conf, f"No 'output_folder' key in path config: {path_conf}"
            assert output_folder is not None, f"Inputted output folder path does not exist"
            
            save_dir = output_folder / Path(model_name)
            os.makedirs(save_dir, exist_ok=True)

            results_file = save_dir / f"results_{splittype}_{TP_ABBREVIATIONS[train_plan]}_{modtype}_{ts}.jsonl"
            config_file = save_dir / f"config_{splittype}_{TP_ABBREVIATIONS[train_plan]}_{modtype}_{ts}.json"
            full_config = {
                "timestamp": ts,
                "engine_params": engine_params,
                "sampling_params": sampling_params,
                "input_dataset": str(input_path),
                "lora_path": lora_path
            }

            # saving config
            with open(config_file, "w") as cf:
                json.dump(full_config, cf, indent=4)
            print(f"Config saved to: {config_file}")

            # saving results
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
    parser = argparse.ArgumentParser(description="Fully runs offline inference pipeline.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan to finetune on.")
    parser.add_argument("--adapter_folder", type=str, nargs='?', default=None, help="Folder with safetensor and json.")
    parser.add_argument("--model_name", type=str, nargs='?', default='meta-llama/Llama-3.1-8B', help="Full name for model")
    parser.add_argument("--model_nickname", type=str, nargs='?', default=None, help="Nickname for model")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()
    
    TRAIN_PLAN = args.train_plan
    TRAIN_PATH = Path(f"calyapo/data/final/{TRAIN_PLAN}_train.jsonl")
    VAL_PATH = Path(f"calyapo/data/final/{TRAIN_PLAN}_val.jsonl")
    TEST_PATH = Path(f"calyapo/data/final/{TRAIN_PLAN}_test.jsonl")
    OUTPUT_FOLDER = Path(f"inference_outputs/{TRAIN_PLAN}")
    LORA_ADAPTER_PATH = Path(f"calyapo/training/checkpoints/{TRAIN_PLAN}_dataset/{args.adapter_folder}")
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    engine_config = {
        "model": args.model_name, 
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "dtype": "float16",
        "max_model_len": 212, # prompts are not that long
        "max_num_seqs": 96,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True, 
        "seed": 42, 
        "enable_lora": True, # initialize the engine to have lora enabled, then toggle the path request
        "max_loras": 1
    }

    sampling_config = {
        "temperature": 0,
        "max_tokens": 2, # only need to generate one response (A, B, C or D) but give some flexibility
        "logprobs": 5 
    }

    path_conf = {
        'train' : TRAIN_PATH, 
        'val' : VAL_PATH, 
        'test' : TEST_PATH, 
        'output_folder' : OUTPUT_FOLDER, 
        'lora_weights_folder' : LORA_ADAPTER_PATH
    }

    run_inference(
        engine_params=engine_config, 
        sampling_params=sampling_config,  
        train_plan=args.train_plan, 
        path_conf=path_conf,
        verbose=True
    )

    