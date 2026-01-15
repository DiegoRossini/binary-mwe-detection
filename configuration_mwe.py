from transformers import PretrainedConfig


class MWEConfig(PretrainedConfig):
    model_type = "mwe-deberta"

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-large",
        hidden_size: int = 1024,
        dropout: float = 0.3,
        chunk_vocab_size: int = 2,
        chunk_embedding_dim: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.chunk_vocab_size = chunk_vocab_size
        self.chunk_embedding_dim = chunk_embedding_dim
