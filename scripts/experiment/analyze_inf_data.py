import json
from pathlib import Path

OUTPUT_FOLDER = Path("inference_outputs")
P2A_PATHS = {
    'train' : {
        'lora' : {
            'results' : "results_train_p2a_lora_20260324_222337.jsonl", 
            'config' : "config_train_p2a_lora_20260324_222337.json"
        }, 
        'base' : {
            'results' : "results_train_p2a_base_20260324_171048.jsonl", 
            'config' : "config_train_p2a_base_20260324_171048.json"
        }, 
    }, 
    'val' : {
        'lora' : {
            'results' : "results_validation_p2a_lora_20260325_084809.jsonl", 
            'config' : "config_validation_p2a_lora_20260325_084809.json"
        }, 
        'base' : {
            'results' : "results_validation_p2a_base_20260325_083628.jsonl", 
            'config' : "config_validation_p2a_base_20260325_083628.json"
        }, 
    }
}
PATH_CONFIG = {
    'presidents_to_abortion' : P2A_PATHS
}

def get_path(train_plan, verbose=False):
    if train_plan not in PATH_CONFIG:
        raise ValueError(f"inputted training plan '{train_plan}' not in path config, choose from: {PATH_CONFIG.keys()}")
    
    train_plan_conf = PATH_CONFIG[train_plan]
    out = {
        'train_lora' : {
            'results_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['train']['lora']['results'], 
            'config_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['train']['lora']['config']
        }, 
        'train_base' : {
            'results_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['train']['base']['results'], 
            'config_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['train']['base']['config']
        }, 
        'val_lora' : {
            'results_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['val']['lora']['results'], 
            'config_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['val']['lora']['config']
        }, 
        'val_base' : {
            'results_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['val']['base']['results'], 
            'config_path' : OUTPUT_FOLDER / Path(str(train_plan)) / train_plan_conf['val']['base']['config']
        }, 
    }
    return out

def calculate_accuracy(results_path, config_path, bootstrap=False, verbose=False):
    if verbose:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        engine_params = config_data['engine_params']
        sampling_params = config_data['sampling_params']
        print(f"model:                   {engine_params.get('model', None)}")
        print(f"quantization:            {engine_params.get('quantization', None)}")
        print(f"max_model_len:           {engine_params.get('max_model_len', None)}")
        print(f"max_num_seqs:            {engine_params.get('max_num_seqs', None)}")
        print(f"gpu_memory_utilization:  {engine_params.get('gpu_memory_utilization', None)}")
        print(f"LoRA Enabled:            {engine_params.get('enable_lora', None)}")
        print(f"seed:                    {engine_params.get('seed', None)}")

        print(f"temperature:             {sampling_params.get('temperature', None)}")
        print(f"max_tokens:              {sampling_params.get('max_tokens', None)}")
        print(f"logprobs:                {sampling_params.get('logprobs', None)}")
        print(f"----------------------------------------------")
    
    num_correct = 0
    num_datapoints = 0
    data = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                num_datapoints += 1
                pred_correct = int(json_obj['is_correct'])
                num_correct += pred_correct
                data.append(json_obj)
    
    acc = num_correct / num_datapoints
    # calculate confidence interval
    if verbose: 
        print(f"\n----------------------- Metrics -----------------------")
        print(f"Total number of datapoints loaded in: {num_datapoints}")
        print(f"Accuracy: {acc}")
        print(f"---------------------------------------------------------")
    return acc

def main(train_plan, verbose = False):
    splits_conf = get_path(train_plan, verbose=verbose)
    for k, v in splits_conf.items():
        print(f"-------------- Processing Split '{k}' --------------\n")
        calculate_accuracy(**v, verbose=verbose)

if __name__ == "__main__":
    tp = 'presidents_to_abortion'
    main(tp, verbose=True)