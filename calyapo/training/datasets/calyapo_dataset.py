import copy
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class CalyapoDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        """
        Args:
            dataset_config: Configuration object containing paths (train_split, test_split). Sourced from configs/datasets.py
            tokenizer: The HuggingFace tokenizer.
            partition: 'train' or 'test' (passed by finetuning.py).
        """
        self.tokenizer = tokenizer

        if partition == "train":
            file_path = dataset_config.train_split
        elif partition == "val":
            file_path = dataset_config.val_split
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignores this value
        
        # 1. Get raw text
        ann = self.data[index]
        prompt_text = ann["prompt"]
        completion_text = ann["completion"]
        
        # 2. Construct Full Source
        # We manually manage special tokens to ensure alignment
        full_text = prompt_text + completion_text
        
        # 3. Tokenize Prompt (Context) - Add BOS
        # We need the length of this to know where to start calculating loss
        prompt_ids = self.tokenizer.encode(
            self.tokenizer.bos_token + prompt_text, 
            add_special_tokens=False
        )
        
        # 4. Tokenize Full Example - Add BOS + EOS
        example_ids = self.tokenizer.encode(
            self.tokenizer.bos_token + full_text + self.tokenizer.eos_token, 
            add_special_tokens=False
        )
        
        # 5. Create Labels (Masking)
        # Start with a copy of the input indices
        labels = copy.deepcopy(example_ids)
        
        # Set the labels for the "Prompt" section to -100.
        # This tells the model: "Read this, but don't try to predict it."
        # We mask everything up to the length of the prompt_ids.
        # Note: We check lengths to avoid index errors if tokenization gets weird.
        mask_len = len(prompt_ids)
        if mask_len < len(labels):
            labels[:mask_len] = [IGNORE_INDEX] * mask_len
        else:
            # Fallback if prompt somehow equals or exceeds full example (shouldn't happen)
            labels[:] = [IGNORE_INDEX] * len(labels)

        # 6. Convert to Tensors
        # We return lists here; the DataCollator (in finetuning.py) handles padding/stacking.
        return {
            "input_ids": example_ids,
            "labels": labels,
            "attention_mask": [1] * len(example_ids),
        }

def get_calyapo_dataset(dataset_config, tokenizer, split):
    """
    Entry point function used by the Llama Cookbook loader.
    """
    return CalyapoDataset(dataset_config, tokenizer, split)