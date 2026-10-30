import math

from torch import Tensor, nn


class NeuralCharModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        activation: str = "relu",
        num_hidden_layers: int = 1,
        normalization: str = "none",
        dropout: float = 0.0,
    ):
        super().__init__()

        for name, value in (
            ("vocab_size", vocab_size),
            ("context_length", context_length),
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
            ("num_hidden_layers", num_hidden_layers),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if normalization not in ("none", "layer_norm", "batch_norm"):
            raise ValueError("Unknown normalization")

        if (
            isinstance(dropout, bool)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")

        self.context_length = context_length
        activation_types = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }

        if activation not in activation_types:
            raise ValueError(
                f"Unknown activation {activation!r}; "
                f"choose from {sorted(activation_types)}"
            )

        self.activation_name = activation

        layers: list[nn.Module] = [
            nn.Embedding(vocab_size, embedding_dim),
            nn.Flatten(start_dim=1),
        ]

        input_dim = context_length * embedding_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))

            if normalization == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dim))

            elif normalization == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation_types[activation]())

            if dropout:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.network = nn.Sequential(*layers)

    def forward(self, context_ids: Tensor) -> Tensor:
        """Embed contexts and produce one score per vocabulary token.

        Args:
            context_ids: Integer tensor shaped batch by context length.
        Returns:
            Logits shaped batch by vocabulary size.
        """
        return self.network(context_ids)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = NeuralCharModel(
        vocab_size=6, context_length=3, embedding_dim=4, hidden_dim=8
    ).eval()
    contexts = torch.tensor([[0, 0, 4], [0, 4, 3]], dtype=torch.long)
    print("Two contexts of character IDs:", contexts.tolist())
    print("Model: untrained")

    with torch.no_grad():
        values = contexts

        for layer in model.network:
            values = layer(values)
            print(f"{type(layer).__name__}: {tuple(values.shape)}")

        print("Single-context output:", tuple(model(contexts[:1]).shape))
