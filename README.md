# Binary MWE Detection

Code for "Binary Token-Level Classification with DeBERTa for All-Type MWE Identification" (EACL 2026 Findings).

## Results

| Model | F1 |
|-------|-----|
| DLT+lo (ours) | 69.8% |
| Qwen-72B | 57.8% |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Inference

```bash
python inference.py "I'm looking forward to the meeting."
python inference.py --model outputs/model.safetensors "He kicked the bucket last night."
```

## Training

```bash
# 1. Download CoAM
export HF_TOKEN=your_token
python scripts/download_dataset.py

# 2. Generate projections
python scripts/generate_projections.py

# 3. Train
python main.py
```

## Structure

```
├── main.py           # Training
├── inference.py      # Single-text inference
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── features.py
└── scripts/
    ├── download_dataset.py
    ├── generate_projections.py
    └── run_training.sh
```

## Models

Pretrained: [huggingface.co/DiegoRossini](https://huggingface.co/DiegoRossini)

## Citation

```bibtex
@inproceedings{rossini2026binary,
    title = "Binary Token-Level Classification with {DeBERTa} for All-Type {MWE} Identification",
    author = "Rossini, Diego and van der Plas, Lonneke",
    booktitle = "Findings of EACL 2026",
    year = "2026"
}
```

## License

MIT
