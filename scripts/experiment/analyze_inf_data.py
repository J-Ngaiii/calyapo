import json
from pathlib import Path

OUTPUT_FOLDER = Path("inference_outputs")
RESULTS_PATH = OUTPUT_FOLDER / "results_20260324_171048.jsonl"
CONFIG_PATH = OUTPUT_FOLDER / "config_20260324_171048.json"

def main(results_path, config_path, verbose=False):
    if verbose:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        engine_params = config_data['engine_params']
        sampling_params = config_data['sampling_params']
        print(f"Calculating Accuracy for model with following configs:")
        print(f"model:                   {engine_params.get('model', None)}")
        print(f"quantization:            {engine_params.get('quantization', None)}")
        print(f"max_model_len:           {engine_params.get('max_model_len', None)}")
        print(f"max_num_seqs:            {engine_params.get('max_num_seqs', None)}")
        print(f"gpu_memory_utilization:  {engine_params.get('gpu_memory_utilization', None)}")
        print(f"seed:                    {engine_params.get('seed', None)}")
        print(f"--------------------------------------------------------------")

        print(f"\n---------------Sampler Stats---------------")
        print(f"temperature:             {sampling_params.get('temperature', None)}")
        print(f"max_tokens:              {sampling_params.get('max_tokens', None)}")
        print(f"logprobs:                {sampling_params.get('logprobs', None)}")
        print(f"--------------------------------------------")
    
    num_correct = 0
    num_datapoints = 0
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                num_datapoints += 1
                pred_correct = int(json_obj['is_correct'])
                num_correct += pred_correct
    
    acc = num_correct / num_datapoints
    if verbose: 
        print(f"Total number of datapoints loaded in: {num_datapoints}")
        print(f"Accuracy: {acc}")
    return acc

if __name__ == "__main__":
    main(RESULTS_PATH, CONFIG_PATH, verbose=True)