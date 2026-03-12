import asyncio
import json
import os
from openai import AsyncOpenAI
from pathlib import Path

from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER

ROOT_DIR = Path(__file__).resolve().parents[2]

# 1. Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf" 
DATASET = "presidents_to_abortion"
DATA_FILES = {
    "train": Path(UNIVERSAL_FINAL_FOLDER / f"{DATASET}_train.jsonl") ,
    "val": Path(UNIVERSAL_FINAL_FOLDER / f"{DATASET}_val.jsonl")
}

# Limit concurrent requests to avoid memory/network issues
CONCURRENCY_LIMIT = 50
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

async def get_prediction(datapoint, i):
    """Processes a single line from the JSONL file."""
    async with semaphore:
        try:
            # Extract fields based on your provided format
            user_prompt = datapoint.get("prompt")
            # We strip the completion to handle leading spaces like " D"
            true_label = datapoint.get("completion", "").strip()

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=5, # We only need the letter (A, B, C, D, E)
                logprobs=True,
                top_logprobs=5
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Simple check: does the prediction start with the correct letter?
            is_correct = prediction.startswith(true_label)
            
            return {
                "index": i,
                "prediction": prediction,
                "true_label": true_label,
                "is_correct": is_correct,
                "logprobs": response.choices[0].logprobs.model_dump()
            }
        except Exception as e:
            print(f"Error on index {i}: {e}")
            return None

async def process_file(file_path, split_name):
    """Reads JSONL line-by-line and runs inference."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load JSONL
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Processing {split_name} ({len(data)} items)...")
    
    tasks = [get_prediction(item, i) for i, item in enumerate(data)]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    # Calculate Stats
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = (correct / len(results)) * 100 if results else 0
    print(f"Done {split_name}. Accuracy: {accuracy:.2f}%")

    # Save output for KL roll-up
    output_path = f"results_{split_name}.jsonl"
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

async def main():
    for split, path in DATA_FILES.items():
        await process_file(path, split)

if __name__ == "__main__":
    asyncio.run(main())