from torch import Tensor, nn


class NeuralCharModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.context_length = context_length
        
        activation_types = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }

        if activation not in activation_types:
            raise ValueError(
                f"Unknown activation {activation!r}; "
                f"choose from {sorted(activation_types)}"
            )
            
        self.activation_name = activation
        activation_layer = activation_types[activation]()
            
        self.network = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Flatten(start_dim=1),
            nn.Linear(context_length * embedding_dim, hidden_dim),
            activation_layer,
            nn.Linear(hidden_dim, vocab_size),
        )
        
    def forward(self, context_ids: Tensor) -> Tensor:
        return self.network(context_ids)
