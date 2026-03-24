import asyncio
import json
import os
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.asyncio import tqdm
from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER

ROOT_DIR = Path(__file__).resolve().parents[2]

# configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf" 
DATASET = "presidents_to_abortion"
DATA_FILES = {
    "train": Path(UNIVERSAL_FINAL_FOLDER / f"{DATASET}_train.jsonl") ,
    "val": Path(UNIVERSAL_FINAL_FOLDER / f"{DATASET}_val.jsonl")
}

# limit concurrent requests to avoid memory/network issues
CONCURRENCY_LIMIT = 50
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# empty key is correct based on https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_client_with_tools/
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

            # change to just call client.completions rather than client.chats.completion since not using chat model
            response = await client.completions.create(
                model=MODEL_NAME,
                prompt=user_prompt, # Pass the string directly
                temperature=0,
                max_tokens=5, 
                logprobs=5 # Note: In Completions API, this is an integer, not a boolean
            )

            # access prediction via .text not .message
            prediction = response.choices[0].text.strip()
            # prediction = response.choices[0].message.content.strip()
            
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
    """Reads JSONL and runs inference with progress tracking."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Processing {split_name} ({len(data)} items)...")
    
    output_path = f"results_{split_name}.jsonl"
    results = []

    # Wrap your tasks in tqdm to see a progress bar
    tasks = [get_prediction(item, i) for i, item in enumerate(data)]
    
    with open(output_path, "w") as f:
        # as_completed allows us to handle results as they finish
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=split_name):
            result = await task
            if result:
                results.append(result)
                # Save incrementally so you don't lose progress
                f.write(json.dumps(result) + "\n")
                f.flush() 

    correct = sum(1 for r in results if r["is_correct"])
    accuracy = (correct / len(results)) * 100 if results else 0
    print(f"\nDone {split_name}. Accuracy: {accuracy:.2f}%")

async def main():
    for split, path in DATA_FILES.items():
        await process_file(path, split)

if __name__ == "__main__":
    asyncio.run(main())