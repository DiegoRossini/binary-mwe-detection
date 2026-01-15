import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from .configuration_mwe import MWEConfig

CHUNK_MAP = {"O": 0, "NP": 1}
MAX_SEQ_LEN = 256
WINDOW_MAX = 13
MAX_MEMBER_LEN = 6


class MWEModel(PreTrainedModel):
    config_class = MWEConfig

    def __init__(self, config: MWEConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        h = config.hidden_size

        self.drop = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(h)
        self.chunk_emb = nn.Embedding(config.chunk_vocab_size, config.chunk_embedding_dim)

        self.fc = nn.Linear(h, h // 2)
        self.head_start = nn.Linear(h // 2 + config.chunk_embedding_dim, 1)
        self.head_end = nn.Linear(h // 2 + config.chunk_embedding_dim, 1)
        self.head_inside = nn.Linear(h // 2 + config.chunk_embedding_dim, 1)

        self._tokenizer = None
        self._nlp = None

        self.post_init()

    def forward(self, input_ids, attention_mask, chunk_feat):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state

        x = self.layer_norm(x)
        x = self.drop(x)

        h = torch.relu(self.fc(x))
        h = self.drop(h)

        chunk_emb = self.chunk_emb(chunk_feat)
        x_cat = torch.cat([h, chunk_emb], dim=-1)

        start = torch.sigmoid(self.head_start(x_cat)).squeeze(-1)
        end = torch.sigmoid(self.head_end(x_cat)).squeeze(-1)
        inside = torch.sigmoid(self.head_inside(x_cat)).squeeze(-1)

        return {"start": start, "end": end, "inside": inside}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        return self._tokenizer

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_lg")
        return self._nlp

    def _preprocess(self, text):
        import networkx as nx

        doc = self.nlp(text)
        tokens = [{"surface": t.text} for t in doc]
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

    def _reconstruct(self, tokens, start_scores, end_scores, inside_scores, dep_distances, thresholds):
        th_start, th_end, th_inside = thresholds

        starts = [i for i, v in enumerate(start_scores) if v >= th_start]
        ends = [i for i, v in enumerate(end_scores) if v >= th_end]

        candidates = []
        for s in starts:
            for e in ends:
                if e <= s or (e - s + 1) > WINDOW_MAX:
                    continue

                members = {s, e}
                for t in range(s + 1, e):
                    if inside_scores[t] >= th_inside:
                        members.add(t)

                if len(members) < (e - s + 1) and dep_distances:
                    member_list = sorted(members)
                    max_dep_dist = 0
                    for i in range(len(member_list) - 1):
                        key = (member_list[i], member_list[i + 1])
                        dist = dep_distances.get(key, 10)
                        max_dep_dist = max(max_dep_dist, dist)
                    if max_dep_dist > 3:
                        continue

                if 2 <= len(members) <= MAX_MEMBER_LEN:
                    candidates.append(tuple(sorted(members)))

        return list(set(candidates))

    def detect(self, text, thresholds=(0.5, 0.6, 0.2), return_details=False):
        """
        Detect MWEs in text.

        Args:
            text: Input text string
            thresholds: Tuple of (start, end, inside) thresholds
            return_details: If True, return detailed info including scores

        Returns:
            List of MWE strings, or list of dicts if return_details=True
        """
        tokens, chunk_tags, dep_distances = self._preprocess(text)
        words = [t["surface"] for t in tokens]

        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                             max_length=MAX_SEQ_LEN, add_special_tokens=True)

        device = next(self.parameters()).device
        input_ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(enc["attention_mask"]).unsqueeze(0).to(device)

        word_ids = enc.word_ids()
        chunk_feat = torch.zeros(len(enc["input_ids"]), dtype=torch.long)
        for ti, wid in enumerate(word_ids):
            if wid is not None:
                chunk_feat[ti] = CHUNK_MAP.get(chunk_tags[wid], 0)
        chunk_feat = chunk_feat.unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, chunk_feat)

        start_scores = [0.0] * len(tokens)
        end_scores = [0.0] * len(tokens)
        inside_scores = [0.0] * len(tokens)
        used = set()

        for ti, wid in enumerate(word_ids):
            if wid is not None and wid not in used:
                start_scores[wid] = float(out["start"][0, ti])
                end_scores[wid] = float(out["end"][0, ti])
                inside_scores[wid] = float(out["inside"][0, ti])
                used.add(wid)

        mwe_indices = self._reconstruct(tokens, start_scores, end_scores, inside_scores,
                                        dep_distances, thresholds)

        mwes = []
        for indices in mwe_indices:
            mwe_text = " ".join(words[i] for i in indices)
            if return_details:
                mwes.append({
                    "text": mwe_text,
                    "indices": list(indices),
                    "scores": {
                        "start": [start_scores[i] for i in indices],
                        "end": [end_scores[i] for i in indices],
                        "inside": [inside_scores[i] for i in indices]
                    }
                })
            else:
                mwes.append(mwe_text)

        return mwes
