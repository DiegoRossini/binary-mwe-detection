#!/usr/bin/env python3
import os
import json
from pathlib import Path
from huggingface_hub import login
from datasets import load_dataset

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "coam_dataset"


def download_coam():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in with HF_TOKEN")
    else:
        print("No HF_TOKEN set, attempting without auth...")

    print("Downloading CoAM dataset...")
    dataset = load_dataset("yusuke196/CoAM")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in dataset.items():
        data_dict = split_data.to_dict()
        num_examples = len(next(iter(data_dict.values())))
        examples = [{k: v[i] for k, v in data_dict.items()} for i in range(num_examples)]

        output_file = OUTPUT_DIR / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(examples)} examples to {output_file}")

    print("Done!")


if __name__ == "__main__":
    download_coam()
