from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import DATA_ROOT, TYPE_MAP, CHUNK_MAP, MAX_SEQ_LEN


@dataclass
class Token:
    idx: int
    surface: str
    lemma: str
    pos: str
    head: Optional[int]
    chunk_tag: Optional[str] = None


@dataclass(frozen=True)
class MWE:
    tokens: Tuple[int, ...]
    type: Optional[str] = None

    def __post_init__(self):
        if self.tokens != tuple(sorted(self.tokens)):
            raise ValueError(f"Unsorted tokens {self.tokens}")


@dataclass
class Sentence:
    sent_id: str
    text: str
    tokens: List[Token]
    mwes: List[MWE]
    dep_distances: Optional[Dict[Tuple[int, int], int]] = None


def _strip_comments(raw: str) -> str:
    raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
    return "\n".join([ln for ln in raw.splitlines()
                      if not ln.lstrip().startswith(("//", "#"))])


def _parse_top(text: str, path: Path) -> List[dict]:
    if not text.strip():
        raise ValueError(f"{path} empty.")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    recs = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {i} invalid JSON") from e
    return recs


def load_coam_file(path: Path) -> List[Sentence]:
    stripped = _strip_comments(path.read_text(encoding="utf-8"))
    records = _parse_top(stripped, path)
    out = []

    for rec in records:
        sid = rec.get("sent_id") or rec.get("id") or rec.get("sentence_id")
        toks_raw = rec.get("tokens") or []
        toks = []

        for t in toks_raw:
            toks.append(Token(
                idx=t["idx"],
                surface=t.get("surface") or t.get("form") or "",
                lemma=t.get("lemma", ""),
                pos=t.get("pos", ""),
                head=t.get("head")
            ))

        for i, tk in enumerate(toks):
            if i != tk.idx:
                raise ValueError(f"Idx mismatch {sid}")

        mwes = []
        for m in rec.get("mwes", []):
            idxs = m.get("indices") or []
            if idxs:
                mwes.append(MWE(
                    tokens=tuple(sorted(int(x) for x in idxs)),
                    type=TYPE_MAP.get(m.get("type"), m.get("type"))
                ))

        out.append(Sentence(sid, rec.get("text", ""), toks, mwes))

    return out


def load_dataset(root: Path = DATA_ROOT) -> Dict[str, List[Sentence]]:
    return {
        split: load_coam_file(root / f"{split}.json")
        for split in ("train", "test")
    }


def load_projection_file(path: Path) -> Dict[str, Dict[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["sentences"]


def bitstring_to_list(bits: str) -> List[int]:
    return [1 if c == '1' else 0 for c in bits]


class CoAMTokenDataset(Dataset):
    def __init__(self, sentences: List[Sentence], proj_map: Dict[str, Dict[str, str]],
                 tokenizer: AutoTokenizer, max_seq_len: int = MAX_SEQ_LEN):
        self.sents = []
        self.labels = []
        self.chunk_tags = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        for s in sentences:
            if s.sent_id.endswith("_oversample"):
                orig_id = s.sent_id[:-11]
                entry = proj_map.get(orig_id)
            else:
                entry = proj_map.get(s.sent_id)

            if entry is None:
                if not s.sent_id.endswith("_oversample"):
                    raise ValueError(f"Missing projection for {s.sent_id}")
                continue

            self.sents.append(s)
            self.labels.append({
                "start": bitstring_to_list(entry["start"]),
                "end": bitstring_to_list(entry["end"]),
                "inside": bitstring_to_list(entry["inside"])
            })
            self.chunk_tags.append([t.chunk_tag or "O" for t in s.tokens])

    def __len__(self) -> int:
        return len(self.sents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.sents[idx]
        labs = self.labels[idx]
        chunk_tags = self.chunk_tags[idx]
        words = [t.surface for t in s.tokens]

        enc = self.tokenizer(
            words, is_split_into_words=True, truncation=True,
            max_length=self.max_seq_len, add_special_tokens=True
        )
        word_ids = enc.word_ids()
        L = len(enc["input_ids"])

        start = torch.zeros(L)
        end = torch.zeros(L)
        inside = torch.zeros(L)
        mask = torch.zeros(L, dtype=torch.bool)
        chunk_feat = torch.zeros(L, dtype=torch.long)

        for ti, wid in enumerate(word_ids):
            if wid is None:
                continue
            if ti == 0 or word_ids[ti - 1] != wid:
                mask[ti] = True
                start[ti] = labs["start"][wid]
                end[ti] = labs["end"][wid]
                inside[ti] = labs["inside"][wid]
                chunk_feat[ti] = CHUNK_MAP.get(chunk_tags[wid], 0)

        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "start": start, "end": end, "inside": inside,
            "loss_mask": mask, "chunk_feat": chunk_feat, "sent_idx": idx
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["input_ids", "attention_mask", "start", "end", "inside", "loss_mask", "chunk_feat"]
    max_len = max(len(x["input_ids"]) for x in batch)
    out = {}

    for k in keys:
        padded = []
        for x in batch:
            t = x[k]
            if len(t) < max_len:
                pad = torch.zeros(max_len - len(t), dtype=t.dtype)
                t = torch.cat([t, pad], 0)
            padded.append(t)
        out[k] = torch.stack(padded)

    out["sent_idx"] = [x["sent_idx"] for x in batch]
    return out
