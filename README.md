---
language: en
license: mit
tags:
  - multiword-expressions
  - mwe
  - token-classification
  - deberta
  - nlp
datasets:
  - yusuke196/CoAM
metrics:
  - f1
pipeline_tag: token-classification
---

# Binary MWE Detection (DLT+lo)

DeBERTa-v3-large fine-tuned for multiword expression identification using binary token-level classification. Handles both **continuous** and **discontinuous** MWEs.

**Paper:** "Binary Token-Level Classification with DeBERTa for All-Type MWE Identification" (EACL 2026 Findings)

## Results

| Model | Overall F1 | Continuous F1 | Discontinuous F1 |
|-------|------------|---------------|------------------|
| **DLT+lo (this)** | **69.8%** | **72.1%** | **53.8%** |
| Qwen-72B | 57.8% | — | — |

## Installation

```bash
pip install transformers torch spacy networkx
python -m spacy download en_core_web_lg
```

## Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("DiegoRossini/mwe-detection-deberta", trust_remote_code=True)

# Continuous MWE
mwes = model.detect("They made up their minds.")
print(mwes)  # ['made up']

# Discontinuous MWE
mwes = model.detect("I ran into an old friend yesterday.")
print(mwes)  # ['ran into']

# With detailed output
mwes = model.detect("He kicked the bucket last night.", return_details=True)
```

## Thresholds

Default thresholds (start, end, inside): `(0.5, 0.6, 0.2)`

Adjust based on your precision/recall needs:
- Lower thresholds → more MWEs detected (higher recall)
- Higher thresholds → fewer but more confident MWEs (higher precision)

## Training

Trained on [CoAM](https://huggingface.co/datasets/yusuke196/CoAM) with:
- Encoder: DeBERTa-v3-large
- Linguistic features: NP chunking, dependency distances
- Data augmentation: 30% oversampling

Code: [github.com/DiegoRossini/binary-mwe-detection](https://github.com/DiegoRossini/binary-mwe-detection)

## Citation

```bibtex
@inproceedings{rossini2026binary,
    title = "Binary Token-Level Classification with {DeBERTa} for All-Type {MWE} Identification",
    author = "Rossini, Diego and van der Plas, Lonneke",
    booktitle = "Findings of EACL 2026",
    year = "2026"
}
```
