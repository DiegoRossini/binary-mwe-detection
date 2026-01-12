#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import hashlib
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set

REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "data" / "coam_dataset"
OUTPUT_DIR = REPO_ROOT / "data" / "projection_artifacts"

MAX_MEMBER_LEN = 6
WINDOW_MAX = 13
PROJECTION_VERSION = "v2"

TYPE_MAP = {
    "modifier/connective": "MOD/CONN",
    "noun": "NOUN",
    "verb": "VERB",
    "clause": "CLAUSE",
    "other_pos": "OTHER",
    "head_not_in_mwe": "OTHER"
}


@dataclass
class Token:
    idx: int
    surface: str
    lemma: str
    pos: str
    head: Optional[int]


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
        sent_id = rec.get("sent_id") or rec.get("id") or rec.get("sentence_id")
        toks = [Token(t["idx"], t.get("surface") or t.get("form") or "",
                      t.get("lemma", ""), t.get("pos", ""), t.get("head"))
                for t in rec.get("tokens", [])]
        mwes = []
        for m in rec.get("mwes", []):
            idx_list = m.get("indices") or []
            if idx_list:
                mwes.append(MWE(tuple(sorted(int(x) for x in idx_list)),
                                TYPE_MAP.get(m.get("type"), m.get("type"))))
        out.append(Sentence(sent_id, rec.get("text", ""), toks, mwes))
    return out


def load_dataset(root: Path) -> Dict[str, List[Sentence]]:
    return {split: load_coam_file(root / f"{split}.json") for split in ("train", "test")}


def project_sentence(sent: Sentence) -> Dict[str, List[int]]:
    n = len(sent.tokens)
    start, end, inside = [0] * n, [0] * n, [0] * n
    for m in sent.mwes:
        toks = list(m.tokens)
        if len(toks) < 2:
            continue
        start[toks[0]] = 1
        end[toks[-1]] = 1
        for t in toks[1:-1]:
            inside[t] = 1
    return {"start": start, "end": end, "inside": inside}


def reconstruct_candidates(sent, start_scores, end_scores, inside_scores,
                           th_start=0.5, th_end=0.5, th_inside=0.5) -> List[Tuple[int, ...]]:
    n = len(sent.tokens)
    starts = [i for i in range(n) if start_scores[i] >= th_start]
    ends = [i for i in range(n) if end_scores[i] >= th_end]
    out: Set[Tuple[int, ...]] = set()
    for s in starts:
        for e in ends:
            if e <= s or (e - s + 1) > WINDOW_MAX:
                continue
            members = {s, e}
            for t in range(s + 1, e):
                if inside_scores[t] >= th_inside:
                    members.add(t)
            if 2 <= len(members) <= MAX_MEMBER_LEN:
                out.add(tuple(sorted(members)))
    return sorted(out)


def ceiling_check(sentences: List[Sentence]) -> Dict[str, Any]:
    total_gold, recovered = 0, 0
    for sent in sentences:
        if not sent.mwes:
            continue
        proj = project_sentence(sent)
        preds = reconstruct_candidates(sent, proj["start"], proj["end"], proj["inside"])
        gold_set = set(m.tokens for m in sent.mwes)
        total_gold += len(gold_set)
        recovered += len(gold_set & set(preds))
    return {"ceiling": recovered / total_gold if total_gold else 1.0,
            "total_gold": total_gold, "recovered": recovered}


def bitstring(bits: List[int]) -> str:
    return "".join(str(b) for b in bits)


def build_artifact(sentences: List[Sentence], split: str) -> Dict[str, Any]:
    payload = {}
    for s in sentences:
        lab = project_sentence(s)
        payload[s.sent_id] = {
            "start": bitstring(lab["start"]),
            "end": bitstring(lab["end"]),
            "inside": bitstring(lab["inside"])
        }
    return {
        "meta": {
            "version": PROJECTION_VERSION, "split": split,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "spec": {"max_member_len": MAX_MEMBER_LEN, "window_max": WINDOW_MAX}
        },
        "sentences": payload
    }


def write_artifact(artifact: Dict[str, Any], path: Path) -> str:
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    if not DATA_ROOT.exists():
        print(f"Error: {DATA_ROOT} not found. Run download_dataset.py first.")
        exit(1)

    print(f"Loading from {DATA_ROOT}")
    dataset = load_dataset(DATA_ROOT)

    for split, sents in dataset.items():
        ceiling = ceiling_check(sents)
        print(f"{split}: {len(sents)} sentences, ceiling {ceiling['ceiling']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, sents in dataset.items():
        artifact = build_artifact(sents, split)
        path = OUTPUT_DIR / f"projection_{split}_{PROJECTION_VERSION}.json"
        h = write_artifact(artifact, path)
        print(f"Saved {path.name} [{h[:16]}...]")

    print("Done!")
