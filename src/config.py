import os
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
PROJECT_ROOT = Path(os.environ.get("MWE_PROJECT_ROOT", Path(__file__).parent.parent))
DATA_ROOT = Path(os.environ.get("MWE_DATA_ROOT", PROJECT_ROOT / "data" / "coam_dataset"))
PROJECTION_DIR = Path(os.environ.get("MWE_PROJECTION_DIR", PROJECT_ROOT / "data" / "projection_artifacts"))
TRAIN_PROJ_FILE = PROJECTION_DIR / "projection_train_v2.json"
TEST_PROJ_FILE = PROJECTION_DIR / "projection_test_v2.json"
LOG_DIR = PROJECT_ROOT / "outputs"

# Model
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_SEQ_LEN = 256
DROPOUT = 0.3

# Training
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-5
SEED = 42
GRAD_ACCUM = 1
PATIENCE = 3
WEIGHT_DECAY = 0.01

# Data
USE_DEV = True
DEV_RATIO = 0.15
USE_OVERSAMPLING = True
OVERSAMPLE_RATIO = 0.30

# Reconstruction
MAX_MEMBER_LEN = 6
WINDOW_MAX = 13
THRESH_GRID = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]

SAVE_MODEL = True

TYPE_MAP = {
    "modifier/connective": "MOD/CONN",
    "noun": "NOUN",
    "verb": "VERB",
    "clause": "CLAUSE",
    "other_pos": "OTHER",
    "head_not_in_mwe": "OTHER"
}

CHUNK_MAP = {"O": 0, "NP": 1}
