#!/usr/bin/env python3
"""
Simple MWE detection on a single sentence.

Usage:
    python inference.py "I'm looking forward to the meeting."
    python inference.py --model path/to/model.safetensors "Your text here"
"""
import argparse
import torch
import spacy
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from torch import nn
from safetensors.torch import load_file

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_SEQ_LEN = 256
WINDOW_MAX = 13
MAX_MEMBER_LEN = 6
CHUNK_MAP = {"O": 0, "NP": 1}


class StartEndInsideModel(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(h)
        self.chunk_emb = nn.Embedding(2, 16)
        self.fc = nn.Linear(h, h // 2)
        self.head_start = nn.Linear(h // 2 + 16, 1)
        self.head_end = nn.Linear(h // 2 + 16, 1)
        self.head_inside = nn.Linear(h // 2 + 16, 1)

    def forward(self, input_ids, attention_mask, chunk_feat):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        x = self.layer_norm(x)
        x = self.drop(x)
        h = torch.relu(self.fc(x))
        h = self.drop(h)
        x_cat = torch.cat([h, self.chunk_emb(chunk_feat)], dim=-1)
        start = torch.sigmoid(self.head_start(x_cat)).squeeze(-1)
        end = torch.sigmoid(self.head_end(x_cat)).squeeze(-1)
        inside = torch.sigmoid(self.head_inside(x_cat)).squeeze(-1)
        return start, end, inside


def load_model(model_path, device):
    model = StartEndInsideModel(MODEL_NAME).to(device)
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(text, nlp):
    doc = nlp(text)
    tokens = [{"surface": t.text, "idx": i} for i, t in enumerate(doc)]
    chunk_tags = ["O"] * len(doc)
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunk_tags[i] = "NP"

    dep_distances = {}
    edges = [(t.i, t.head.i) for t in doc if t.head != t]
    if edges:
        G = nx.Graph(edges)
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                try:
                    dep_distances[(i, j)] = min(nx.shortest_path_length(G, i, j), 5)
                except:
                    dep_distances[(i, j)] = 5

    return tokens, chunk_tags, dep_distances


def reconstruct(tokens, start_scores, end_scores, inside_scores, dep_distances, thresholds):
    th_s, th_e, th_i = thresholds
    starts = [i for i, v in enumerate(start_scores) if v >= th_s]
    ends = [i for i, v in enumerate(end_scores) if v >= th_e]

    candidates = []
    for s in starts:
        for e in ends:
            if e <= s or (e - s + 1) > WINDOW_MAX:
                continue
            members = {s, e}
            for t in range(s + 1, e):
                if inside_scores[t] >= th_i:
                    members.add(t)

            if len(members) < (e - s + 1) and dep_distances:
                member_list = sorted(members)
                max_dist = max(dep_distances.get((member_list[i], member_list[i + 1]), 10)
                               for i in range(len(member_list) - 1))
                if max_dist > 3:
                    continue

            if 2 <= len(members) <= MAX_MEMBER_LEN:
                candidates.append(tuple(sorted(members)))

    return list(set(candidates))


def detect_mwes(text, model, tokenizer, nlp, device, thresholds=(0.5, 0.6, 0.2)):
    tokens, chunk_tags, dep_distances = preprocess(text, nlp)
    words = [t["surface"] for t in tokens]

    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=MAX_SEQ_LEN, add_special_tokens=True)

    input_ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(enc["attention_mask"]).unsqueeze(0).to(device)

    word_ids = enc.word_ids()
    chunk_feat = torch.zeros(len(enc["input_ids"]), dtype=torch.long)
    for ti, wid in enumerate(word_ids):
        if wid is not None:
            chunk_feat[ti] = CHUNK_MAP.get(chunk_tags[wid], 0)
    chunk_feat = chunk_feat.unsqueeze(0).to(device)

    with torch.no_grad():
        start_p, end_p, inside_p = model(input_ids, attention_mask, chunk_feat)

    start_scores = [0.0] * len(tokens)
    end_scores = [0.0] * len(tokens)
    inside_scores = [0.0] * len(tokens)
    used = set()

    for ti, wid in enumerate(word_ids):
        if wid is not None and wid not in used:
            start_scores[wid] = float(start_p[0, ti])
            end_scores[wid] = float(end_p[0, ti])
            inside_scores[wid] = float(inside_p[0, ti])
            used.add(wid)

    mwe_indices = reconstruct(tokens, start_scores, end_scores, inside_scores,
                              dep_distances, thresholds)

    mwes = []
    for indices in mwe_indices:
        mwe_text = " ".join(words[i] for i in indices)
        mwes.append({"text": mwe_text, "indices": list(indices)})

    return mwes


def main():
    parser = argparse.ArgumentParser(description="Detect MWEs in text")
    parser.add_argument("text", help="Text to analyze")
    parser.add_argument("--model", default="outputs/model.safetensors", help="Path to model")
    parser.add_argument("--thresholds", nargs=3, type=float, default=[0.5, 0.6, 0.2],
                        help="start end inside thresholds")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_lg")

    print("Loading model...")
    model = load_model(args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\nText: {args.text}")
    mwes = detect_mwes(args.text, model, tokenizer, nlp, device, tuple(args.thresholds))

    print(f"\nFound {len(mwes)} MWE(s):")
    for mwe in mwes:
        print(f"  - {mwe['text']} (indices: {mwe['indices']})")


if __name__ == "__main__":
    main()
