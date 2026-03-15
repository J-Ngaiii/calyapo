from pathlib import Path

from calyapo.configurations.config import UNIVERSAL_FINAL_FOLDER

def direct_check_json(train_plan):
    train_path = UNIVERSAL_FINAL_FOLDER / f"{train_plan}_train.jsonl"
    val_path = UNIVERSAL_FINAL_FOLDER / f"{train_plan}_val.jsonl"

    results = {}

    for name, path in [("train", train_path), ("val", val_path)]:
        if path.exists():
            # 2. Count lines in the JSONL file
            with open(path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f)
            results[name] = count
            print(f"{name.capitalize()} set found: {count} datapoints.")
        else:
            results[name] = 0
            print(f"{name.capitalize()} file not found at: {path}")

    return results

if __name__ == "__main__":
    direct_check_json('presidents_to_abortion')