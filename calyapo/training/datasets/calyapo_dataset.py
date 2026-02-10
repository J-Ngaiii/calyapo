import copy
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

from calyapo.training.configs.datasets import calyapo_dataset_config

class Tokens():
    def __init__(self):
        """
        Literally only exists for labeling things and code readability.
        """
        pass


class CalyapoDataset(Dataset):
    def __init__(self, dataset_config: calyapo_dataset_config, tokenizer, partition="train"):
        """
        Args:
            dataset_config: Configuration object containing paths (train_split, test_split). Sourced from configs/datasets.py
            - takes in calyapo datasets under different training plans
            tokenizer: The HuggingFace tokenizer.
            partition: 'train' or 'test' (passed by finetuning.py).
        """
        self.tokenizer = tokenizer
        self.predict_eos = dataset_config.predict_eos

        if partition == "train":
            file_path = dataset_config.train_split
        else:
            file_path = dataset_config.test_split

        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")      
        print(f"Loading {partition} data from: {file_path}")

        # Alpaca uses json.load (for a single list), but we use line-by-line for JSONL
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip(): # Skip empty lines
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} examples.")

    def update_eos_pred(self, tf: bool):
        self.predict_eos = tf
        print(f"Updated EOS awarness: {'ENABLED' if self.predict_eos else 'DISABLED'}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignores this value
        
        # retrieve and construct raw texts
        ann = self.data[index]
        prompt_text = ann["prompt"]
        completion_text = ann["completion"]
        full_text = prompt_text + completion_text
        
        # tokenize prompt (Context) - add BOS
        # need the length of this to calculate loss
        prompt_ids: Tokens = self.tokenizer.encode(
            self.tokenizer.bos_token + prompt_text, 
            add_special_tokens=False
        )
        
        # tokenize full text (BOS + EOS)
        if self.predict_eos:
            example_ids: Tokens = self.tokenizer.encode(
                self.tokenizer.bos_token + full_text + self.tokenizer.eos_token, 
                add_special_tokens=False
            )
        else:
            example_ids: Tokens = self.tokenizer.encode(
                self.tokenizer.bos_token + full_text, 
                add_special_tokens=False
            )
        
        # make labels for masking
        # Start with a copy of the input indices
        labels: Tokens = copy.deepcopy(example_ids)
        
        # set the labels for the "Prompt" section to -100.
        # this tells the model: "Read this, but don't try to predict it."
        num_prompt_tokens = len(prompt_ids)
        assert num_prompt_tokens < len(labels), f"Prompt is longer (len {num_prompt_tokens}) than full text (len {len(labels)}). Printing both texts below\nPrompt:\n{prompt_text}\nFull Text:\n{full_text}"
        labels[:num_prompt_tokens] = [IGNORE_INDEX] * num_prompt_tokens

        # convert to tensors
        # rely on finetuning.py DataCollator
        return {
            "input_ids": example_ids,
            "labels": labels, # labels now has prompt tokens masked  
            "attention_mask": [1] * len(example_ids), 
            # 1 means the model sees it, 0 means its padding and the model doesn't see it
            # after the mask the model sees the labels --> -100 means read don't predict
        }

def get_calyapo_dataset(dataset_config, tokenizer, split):
    """
    Entry point function used by the Llama Cookbook loader.
    """
    return CalyapoDataset(dataset_config, tokenizer, split)

if __name__ == "__main__":
    from calyapo.training.configs.datasets import ideology_to_ideology_dataset, ideology_to_trump_dataset
    from transformers import AutoTokenizer

    def token_inspector(example_idx, tokenizer, partition = "train", num_start_tokens_to_test = 15, num_end_tokens_to_test = 8, eos_aware = False, dump_full_text = True):
        
        print(f"\nloading {partition} dataset...")
        ds = CalyapoDataset(ideology_to_trump_dataset, tokenizer, partition)
        if eos_aware:
            print("\nEOS aware test")
            ds.update_eos_pred(True)
        else:
            print("\nEOS absent test")
            ds.update_eos_pred(False)
        
        
        item = ds[example_idx]
        input_ids = item['input_ids']
        labels = item['labels']
        mask = item['attention_mask']
        
        print(f"\nexamining idx: {example_idx}")
        print(f"sequence length: {len(input_ids)}")
        
        # -iterate through tokens, check for masks (-100) 
        print("\nloading token-by-token inspection...")
        
        num_start_tokens_to_test = 15
        num_end_tokens_to_test = 8
        full_representation = ""
        
        print(f"{'IDX':<1} | {'TokenID':<6} | {'Status':<19} | {'String Representation'}")
        for i, (inp_id, lbl_id) in enumerate(zip(input_ids, labels)):
            token_str = tokenizer.decode([inp_id]).replace('\n', '\\n')
            full_representation += repr(token_str)

            if lbl_id == -100:
                status = "MASKED"
            else:
                status = "TRAIN"
                
            # only print first 10 tokens and the transition point (where masking stops)
            is_transition = (i > 0 and labels[i] != -100 and labels[i-1] == -100)
            
            if i < num_start_tokens_to_test or is_transition or i >= len(input_ids) - num_end_tokens_to_test:
                print(f"{i:<3} | {inp_id:<7} | {status:<19} | {repr(token_str)}")
            elif i == num_start_tokens_to_test:
                print("... (skipping middle tokens) ...")
        
        if dump_full_text:
            print(f"full representation below for reference:\n{full_representation}")

        print("\n--- token test finished ---")

    def main():
        example_idx = 0

        # setting up tokenizer
        print("loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        except:
            # fallback to GPT2 for logic testing
            print("Llama tokenizer not found, using GPT2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # custom make special tokens
            tokenizer.bos_token = "<|s|>" # start
            tokenizer.eos_token = "<|h|>" # halt
            tokenizer.pad_token = tokenizer.eos_token

        token_inspector(example_idx, tokenizer, partition="train", eos_aware=False)
        token_inspector(example_idx, tokenizer, partition="train", eos_aware=True)
        token_inspector(example_idx, tokenizer, partition="val", eos_aware=False)
        token_inspector(example_idx, tokenizer, partition="val", eos_aware=True)

        print(f"\n----- FULL TEST FINISHED -----")

    main()

    
    
    