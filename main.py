#!/usr/bin/env python3

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import random
import datetime

import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file

from config import (
    DATA_ROOT, LOG_DIR, MODEL_NAME, SEED, BATCH_SIZE,
    USE_DEV, DEV_RATIO, USE_OVERSAMPLING, OVERSAMPLE_RATIO,
    THRESH_GRID, SAVE_MODEL, EPOCHS, LR, WEIGHT_DECAY,
    PATIENCE, DROPOUT, WINDOW_MAX, MAX_MEMBER_LEN,
    TRAIN_PROJ_FILE, TEST_PROJ_FILE
)
from data import load_dataset, load_projection_file, CoAMTokenDataset
from features import load_spacy_model, inject_chunking_and_dependency, apply_oversampling
from model import StartEndInsideModel
from train import train_model_with_seed, set_seed
from evaluate import evaluate_exact, tune_thresholds, collect_detailed_predictions


def main():
    set_seed(SEED)
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading dataset...")
    if not DATA_ROOT.exists():
        print(f"Error: {DATA_ROOT} not found. Run scripts/download_dataset.py first.")
        return

    dataset = load_dataset(DATA_ROOT)
    train_sentences = dataset["train"]
    test_sentences = dataset["test"]
    print(f"Loaded {len(train_sentences)} train, {len(test_sentences)} test sentences")

    print("Loading spaCy and injecting features...")
    nlp = load_spacy_model()
    inject_chunking_and_dependency(train_sentences, nlp)
    inject_chunking_and_dependency(test_sentences, nlp)

    if USE_OVERSAMPLING:
        train_sentences = apply_oversampling(train_sentences, OVERSAMPLE_RATIO, LOG_DIR)

    print("Loading projections...")
    if not TRAIN_PROJ_FILE.exists():
        print(f"Error: {TRAIN_PROJ_FILE} not found. Run scripts/generate_projections.py first.")
        return

    proj_train = load_projection_file(TRAIN_PROJ_FILE)
    proj_test = load_projection_file(TEST_PROJ_FILE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if USE_DEV:
        random.shuffle(train_sentences)
        dev_size = max(1, int(len(train_sentences) * DEV_RATIO))
        dev_sentences = train_sentences[:dev_size]
        train_sentences = train_sentences[dev_size:]
        print(f"Split: {len(train_sentences)} train, {len(dev_sentences)} dev")
    else:
        dev_sentences = None

    ds_train = CoAMTokenDataset(train_sentences, proj_train, tokenizer)
    ds_dev = CoAMTokenDataset(dev_sentences, proj_train, tokenizer) if dev_sentences else None
    ds_test = CoAMTokenDataset(test_sentences, proj_test, tokenizer)

    print("\nTraining...")
    model, history = train_model_with_seed(
        SEED, ds_train, ds_dev, StartEndInsideModel, MODEL_NAME, BATCH_SIZE,
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE
    )

    print("\nTuning thresholds...")
    tune_ds = ds_dev if ds_dev else ds_train
    best = tune_thresholds(model, tune_ds, THRESH_GRID)
    print(f"Best thresholds: {best['thresholds']}, F1: {best['f1']:.4f}")

    print("\nEvaluating...")
    train_eval = evaluate_exact(model, ds_train, best["thresholds"])
    test_eval = evaluate_exact(model, ds_test, best["thresholds"])
    print(f"Train F1: {train_eval['f1']:.4f}, Test F1: {test_eval['f1']:.4f}")

    predictions = collect_detailed_predictions(model, ds_test, best["thresholds"])
    with open(LOG_DIR / "predictions_detail.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    log = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "model": MODEL_NAME, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "seed": SEED, "use_dev": USE_DEV, "dev_ratio": DEV_RATIO,
            "threshold_grid": THRESH_GRID, "window_max": WINDOW_MAX,
            "max_member_len": MAX_MEMBER_LEN, "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
            "oversampling": USE_OVERSAMPLING, "oversample_ratio": OVERSAMPLE_RATIO
        },
        "train_history": history,
        "best_thresholds": list(best["thresholds"]),
        "final_train_eval": train_eval,
        "final_test_eval": test_eval
    }
    (LOG_DIR / "metrics.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))

    if SAVE_MODEL:
        torch.save(model.state_dict(), LOG_DIR / "model.pt")
        save_file(model.state_dict(), LOG_DIR / "model.safetensors")
        print(f"Model saved to {LOG_DIR}")

    print("Done.")


if __name__ == "__main__":
    main()
