from __future__ import annotations
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import spacy
import networkx as nx
from tqdm import tqdm

from data import Token, Sentence
from config import OVERSAMPLE_RATIO


def load_spacy_model(model_name: str = "en_core_web_lg"):
    return spacy.load(model_name)


def align_tokens(our_tokens: List[Token], spacy_doc) -> List[Optional[int]]:
    mapping = []
    spacy_tokens = [t.text for t in spacy_doc]
    spacy_idx = 0

    for our_tok in our_tokens:
        found = False
        for idx in range(spacy_idx, len(spacy_tokens)):
            if spacy_tokens[idx] == our_tok.surface:
                mapping.append(idx)
                spacy_idx = idx + 1
                found = True
                break
        if not found:
            mapping.append(None)

    return mapping


def compute_dep_distance(tok1, tok2) -> int:
    edges = []
    for token in tok1.doc:
        if token.head != token:
            edges.append((token.i, token.head.i))

    if not edges:
        return abs(tok2.i - tok1.i)

    G = nx.Graph()
    G.add_edges_from(edges)

    if not (G.has_node(tok1.i) and G.has_node(tok2.i)):
        return 10

    try:
        return nx.shortest_path_length(G, tok1.i, tok2.i)
    except nx.NetworkXNoPath:
        return 10


def inject_chunking_and_dependency(sentences: List[Sentence], nlp=None,
                                   show_progress: bool = True) -> None:
    if nlp is None:
        nlp = load_spacy_model()

    iterator = tqdm(sentences, desc="Injecting features") if show_progress else sentences

    for sent in iterator:
        doc = nlp(sent.text)
        chunk_tags = ["O"] * len(sent.tokens)
        mapping = align_tokens(sent.tokens, doc)

        for chunk in doc.noun_chunks:
            for i, idx in enumerate(mapping):
                if idx is not None and chunk.start <= idx < chunk.end:
                    chunk_tags[i] = "NP"

        for i, tok in enumerate(sent.tokens):
            tok.chunk_tag = chunk_tags[i]

        dep_distances = {}
        for i in range(len(sent.tokens)):
            for j in range(i + 1, len(sent.tokens)):
                if mapping[i] is not None and mapping[j] is not None:
                    dist = compute_dep_distance(doc[mapping[i]], doc[mapping[j]])
                    dep_distances[(i, j)] = min(dist, 5)
                else:
                    dep_distances[(i, j)] = 5

        sent.dep_distances = dep_distances


def oversample_via_augmentation(sent: Sentence, p: float = 0.1) -> Tuple[Sentence, List[str]]:
    new_tokens = []
    changes = []

    for tok in sent.tokens:
        new_tok = Token(tok.idx, tok.surface, tok.lemma, tok.pos, tok.head, tok.chunk_tag)
        if random.random() < p and tok.pos in ['NOUN', 'VERB', 'ADJ']:
            changes.append(f"token_{tok.idx}")
        new_tokens.append(new_tok)

    new_sent = Sentence(
        sent_id=sent.sent_id + "_oversample",
        text=" ".join([t.surface for t in new_tokens]),
        tokens=new_tokens,
        mwes=sent.mwes,
        dep_distances=sent.dep_distances
    )
    return new_sent, changes


def apply_oversampling(train_sentences: List[Sentence],
                       oversample_ratio: float = OVERSAMPLE_RATIO,
                       log_dir: Optional[Path] = None) -> List[Sentence]:
    oversampled = []
    log_entries = []
    mwe_types = Counter()

    for sent in train_sentences:
        if random.random() < oversample_ratio:
            new_sent, changes = oversample_via_augmentation(sent)
            oversampled.append(new_sent)

            for mwe in sent.mwes:
                mwe_types[mwe.type or "UNKNOWN"] += 1

            log_entries.append({
                "original_id": sent.sent_id,
                "num_mwes": len(sent.mwes),
                "mwe_types": [mwe.type for mwe in sent.mwes],
                "modified": len(changes) > 0
            })

    result = train_sentences + oversampled

    print(f"Oversampling: {len(train_sentences)} -> {len(result)} sentences")

    if log_dir is not None:
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "oversampling_details.json", "w") as f:
            json.dump({"details": log_entries}, f, indent=2)

    return result
