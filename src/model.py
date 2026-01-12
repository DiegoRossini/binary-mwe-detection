import torch
from torch import nn
from transformers import AutoModel

from config import DROPOUT, CHUNK_MAP


class StartEndInsideModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.chunk_emb = nn.Embedding(len(CHUNK_MAP), 16)

        self.fc = nn.Linear(hidden_size, hidden_size // 2)
        self.head_start = nn.Linear(hidden_size // 2 + 16, 1)
        self.head_end = nn.Linear(hidden_size // 2 + 16, 1)
        self.head_inside = nn.Linear(hidden_size // 2 + 16, 1)

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

        return start, end, inside, out.last_hidden_state

    def freeze_encoder(self, freeze: bool = True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
