import json
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from pathlib import Path
import os
from datetime import datetime
import argparse
import time
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from openai import OpenAI

from calyapo.data_eval.correctness import pred_is_correct
load_dotenv()

# --- Configuration ---
TP_ABBREVIATIONS = {
        "presidents_to_abortion" : "p2a", 
        "opinion_school" : "os", 
        "test_plan" : "test"
    }

SOCRATES_QWEN_MODEL = "socratesft/socrates-qwen2.5-14b-dpo"
SOCRATES_LLAMA_MODEL = "socratesft/socrates-llama3-8b-sft"
SOCRATES_SYSTEM_PROMPT = (
    "You are simulating a survey respondent. Answer exactly as instructed, "
    "following the specified response format without additional commentary."
)

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

def format_chat_prompts(raw_data, model_name, system_prompt):
    """
    Formats prompts using the tokenizer-native chat template.
    Works for Qwen, Llama-3, etc.
    """
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatted = []
    for item in raw_data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]}
        ]
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

    print(f"Formatted {len(formatted)} prompts.")
    return formatted

def run_chat_template_inference(
    model_name,
    sampling_params,
    split,
    train_plan,
    input_path,
    output_folder,
    num_gpus=1,
    chunk_size=2000,
    verbose=False,
    system_prompt=SOCRATES_SYSTEM_PROMPT
):
    """
    Generic inference runner for chat-template models
    like SOCRATES-Qwen and SOCRATES-Llama.
    """

    if not os.path.exists(input_path):
        raise ValueError(f"Input path '{input_path}' does not exist")
    raw_data = load_data(input_path)
    if not raw_data:
        return

    # ---------------- Prompt Formatting ----------------
    prompts = format_chat_prompts(
        raw_data=raw_data,
        model_name=model_name,
        system_prompt=system_prompt
    )

    # ---------------- Engine Config ----------------
    engine_config = {
        "model": model_name,
        "tensor_parallel_size": num_gpus,
        "dtype": "bfloat16",
        "max_model_len": 1024,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True,
        "seed": 42
    }

    if verbose:
        print(f"\n------------------------Chat Model Stats------------------------")
        print(f"Model:                   {model_name}")
        print(f"dtype:                   {engine_config['dtype']}")
        print(f"num_gpus:                {engine_config['tensor_parallel_size']}")
        print(f"max_model_len:           {engine_config['max_model_len']}")
        print(f"max_num_seqs:            {engine_config['max_num_seqs']}")
        print(f"gpu_memory_utilization:  {engine_config['gpu_memory_utilization']}")
        print(f"----------------------------------------------------------------")

    llm = LLM(**engine_config)
    vllm_sampling_config = SamplingParams(**sampling_params)

    # ---------------- Inference ----------------
    print(f"Starting inference on {len(prompts)} prompts...")
    all_outputs = []
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]

        print(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} prompts)...")

        chunk_outputs = llm.generate(chunk, vllm_sampling_config)

        all_outputs.extend(chunk_outputs)

    # ---------------- Save ----------------
    ts = get_timestamp()
    save_dir = output_folder / Path(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model_name.split("/")[-1]
    results_file = (
        save_dir /
        f"results_{split}_{TP_ABBREVIATIONS[train_plan]}_{model_slug}_{ts}.jsonl"
    )
    config_file = (
        save_dir /
        f"config_{split}_{TP_ABBREVIATIONS[train_plan]}_{model_slug}_{ts}.json"
    )
    with open(config_file, "w") as cf:
        json.dump({
            "timestamp": ts,
            "model": model_name,
            "engine_params": engine_config,
            "sampling_params": sampling_params,
            "input_dataset": str(input_path)
        }, cf, indent=4)
    print(f"Config saved to: {config_file}")

    with open(results_file, "w") as f:

        for i, output in enumerate(all_outputs):

            generated_text = output.outputs[0].text.strip()

            true_label = raw_data[i].get("completion", "").strip()

            logprobs_data = output.outputs[0].logprobs

            result = {
                "index": i,
                "prediction": generated_text,
                "true_label": true_label,
                "is_correct": pred_is_correct(
                    llm_out=generated_text,
                    true_ans=true_label
                ),
                "logprobs": str(logprobs_data),
                "model": model_name
            }

            f.write(json.dumps(result) + "\n")
    print(f"Results saved to: {results_file}")

def run_gemini_inference(
    model_name,
    sampling_params,
    split,
    train_plan,
    input_path,
    output_folder,
    verbose=False
):
    """Runs inference using Gemini API"""

    if not os.path.exists(input_path):
        raise ValueError(f"Input path '{input_path}' does not exist")

    raw_data = load_data(input_path)

    if not raw_data:
        return

    # ---------------- API Setup ----------------
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")

    client = genai.Client(api_key=api_key)

    # Validate model is reachable before burning through the dataset
    try:
        test_resp = client.models.generate_content(
            model=model_name,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=1)
        )
        print(f"Model '{model_name}' reachable. Starting inference...")
    except Exception as e:
        raise RuntimeError(f"Model validation failed before inference — aborting. Error: {e}")

    # ---------------- Output Setup ----------------
    ts = get_timestamp()

    save_dir = output_folder / Path(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_file = (
        save_dir /
        f"results_{split}_{TP_ABBREVIATIONS[train_plan]}_gemini_{ts}.jsonl"
    )

    config_file = (
        save_dir /
        f"config_{split}_{TP_ABBREVIATIONS[train_plan]}_gemini_{ts}.json"
    )

    config_data = {
        "timestamp": ts,
        "model_name": model_name,
        "sampling_params": sampling_params,
        "input_dataset": str(input_path)
    }

    with open(config_file, "w") as cf:
        json.dump(config_data, cf, indent=4)

    print(f"Starting Gemini inference on {len(raw_data)} prompts → {results_file}")

    # ---------------- Inference Loop ----------------
    n_correct = 0
    n_errors = 0

    # FATAL errors that will never recover — bail immediately
    FATAL_EXCEPTIONS = (
        genai.errors.ClientError,   # covers 400 InvalidArgument, 403 PermissionDenied, 404 NotFound
    )

    with open(results_file, "w") as f:

        for i, item in enumerate(raw_data):

            prompt = item["prompt"]
            true_label = item.get("completion", "").strip()

            generated_text = ""
            success = False
            retries = 0

            while not success and retries < 5:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=sampling_params.get("temperature", 0),
                            max_output_tokens=sampling_params.get("max_tokens", 2),
                            top_k=1
                        )
                    )
                    generated_text = response.text.strip()
                    success = True

                except FATAL_EXCEPTIONS as e:
                    # These will never recover — surface immediately
                    raise RuntimeError(
                        f"Fatal Gemini API error at index={i} — aborting run.\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Details:    {e}"
                    )

                except Exception as e:
                    retries += 1
                    wait = 5 * retries  # escalating backoff: 5s, 10s, 15s, 20s, 25s
                    print(
                        f"[WARN] index={i} retry={retries}/5 "
                        f"({type(e).__name__}: {e}) — retrying in {wait}s"
                    )
                    time.sleep(wait)

            if not success:
                generated_text = "ERROR_FAILED_GENERATION"
                n_errors += 1
                print(f"[ERROR] index={i} exhausted retries — writing ERROR_FAILED_GENERATION")

            is_correct = pred_is_correct(llm_out=generated_text, true_ans=true_label)
            if is_correct:
                n_correct += 1

            result = {
                "index": i,
                "prediction": generated_text,
                "true_label": true_label,
                "is_correct": is_correct,
                "model": model_name
            }

            f.write(json.dumps(result) + "\n")
            f.flush()  # ensure writes hit disk even if the run is killed

            # Per-item progress (always printed, not gated on verbose)
            print(
                f"[{i+1}/{len(raw_data)}] "
                f"pred='{generated_text}' label='{true_label}' correct={is_correct}"
            )

            time.sleep(0.5)

    print(
        f"\nDone. {len(raw_data)} items | "
        f"correct={n_correct} ({100*n_correct/len(raw_data):.1f}%) | "
        f"errors={n_errors}"
    )
    print(f"Results saved to: {results_file}")

def run_inference(engine_params, sampling_params, split, train_plan, input_path, output_folder, chunk_size: int = 2000, lora_path = None, verbose=False):
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
        print(f"Number of Datapoints:    {len(raw_data)}")
        print(f"Training Plan:           {train_plan}")
        print(f"Plan using Abbreviation: {TP_ABBREVIATIONS.get(train_plan, 'no abbreviations found')}")
        print(f"chunk_size:              {chunk_size}")
        print(f"-------------------------------------------------------------")
        
        print(f"\n------------------------Engine Stats------------------------")
        print(f"Initializing vLLM engine for model: '{model_name}'")
        print(f"quantization:            {engine_params.get('quantization', None)}")
        print(f"num_gpus:                {engine_params.get('tensor_parallel_size', None)}")
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
    else:
        lora_request = None
 
    # chunking logic
    all_outputs = []
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} prompts)...")
        
        chunk_outputs = llm.generate(chunk, vllm_sampling_config, lora_request=lora_request)
        all_outputs.extend(chunk_outputs)
    outputs = all_outputs

    ts = get_timestamp()
    model_type = "lora" if engine_params.get('enable_lora', False) else "base"
    save_dir = output_folder / Path(model_name)
    os.makedirs(save_dir, exist_ok=True)

    results_file = save_dir / f"results_{split}_{TP_ABBREVIATIONS[train_plan]}_{model_type}_{ts}.jsonl"
    config_file = save_dir / f"config_{split}_{TP_ABBREVIATIONS[train_plan]}_{model_type}_{ts}.json"
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
                "is_correct": pred_is_correct(llm_out=generated_text, true_ans=true_label),
                "logprobs": str(logprobs_data) 
            }
            f.write(json.dumps(result) + "\n")
    print(f"Results saved to: {results_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully runs offline inference pipeline.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan to finetune on.")
    parser.add_argument("--model_name", type=str, nargs='?',  help="Name of model to finetune on.")
    parser.add_argument("--run_keyword", type=str, nargs='?', default='archon', help="Name of inference run")
    parser.add_argument("--adapter_folder", type=str, nargs='?', default=None, help="Folder with safetensor and json.")
    parser.add_argument("--model_type", type=str, choices=['lora', 'base', 'gemini', 'socrates_qwen', 'socrates_llama'], default='train')
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()
    
    TRAIN_PLAN = args.train_plan
    RUN_KEYWORD = args.run_keyword
    TRAIN_PATH = Path(f"calyapo/data/final_{RUN_KEYWORD}/{TRAIN_PLAN}_train.jsonl")
    VAL_PATH = Path(f"calyapo/data/final_{RUN_KEYWORD}/{TRAIN_PLAN}_val.jsonl")
    TEST_PATH = Path(f"calyapo/data/final_{RUN_KEYWORD}/{TRAIN_PLAN}_test.jsonl")
    OUTPUT_FOLDER = Path(f"inference_outputs/{TRAIN_PLAN}/outputs_{RUN_KEYWORD}")
    LORA_ADAPTER_PATH = Path(f"calyapo/training/checkpoints/{TRAIN_PLAN}_dataset/{args.adapter_folder}")
    
    USE_LORA = args.model_type.lower() == 'lora'
    SPLIT = args.split

    basic_inf_engine_config = {
        "model": args.model_name,
        "tensor_parallel_size": args.num_gpus, 
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "dtype": "float16",
        "max_model_len": 212, # prompts are not that long
        "max_num_seqs": 96,
        "gpu_memory_utilization": 0.75,
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

    if SPLIT == 'val':
        inf_split = "validation"
        input_path = VAL_PATH
    elif SPLIT == 'train':
        inf_split = "training"
        input_path = TRAIN_PATH
    elif SPLIT == 'test':
        inf_split = "test"
        input_path = TEST_PATH

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if args.model_type == 'gemini':
        run_gemini_inference(
            model_name=args.model_name,
            sampling_params=sampling_config,
            split=inf_split,
            train_plan=args.train_plan,
            input_path=input_path,
            output_folder=OUTPUT_FOLDER,
            verbose=True
        )
    elif args.model_type == 'socrates_qwen':
        run_chat_template_inference(
            model_name=SOCRATES_QWEN_MODEL,
            sampling_params=sampling_config,
            split=inf_split,
            train_plan=args.train_plan,
            input_path=input_path,
            output_folder=OUTPUT_FOLDER,
            num_gpus=args.num_gpus,
            chunk_size=args.chunk_size,
            verbose=args.verbose
        )
    elif args.model_type == 'socrates_llama':
        run_chat_template_inference(
            model_name=SOCRATES_LLAMA_MODEL,
            sampling_params=sampling_config,
            split=inf_split,
            train_plan=args.train_plan,
            input_path=input_path,
            output_folder=OUTPUT_FOLDER,
            num_gpus=args.num_gpus,
            chunk_size=args.chunk_size,
            verbose=args.verbose
        )
    else:
        run_inference(
            engine_params=engine_config, 
            sampling_params=sampling_config, 
            split=inf_split, 
            train_plan=args.train_plan, 
            input_path=input_path,
            output_folder=OUTPUT_FOLDER, 
            chunk_size=args.chunk_size, 
            lora_path=lora_path, 
            verbose=True
        )