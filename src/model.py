"""Decoder-style causal transformer for next-token MIDI prediction."""

import torch
from torch import nn


class MidiTransformer(nn.Module):
    """Predict the next MIDI token at every position in a sequence."""

    def __init__(
        self,
        vocabulary_size=268,
        context_length=256,
        embedding_dimension=128,
        number_of_heads=4,
        number_of_layers=3,
        feedforward_dimension=512,
        dropout=0.1,
    ):
        super().__init__()

        if embedding_dimension % number_of_heads != 0:
            raise ValueError(
                "embedding_dimension must be divisible by number_of_heads"
            )

        self.context_length = context_length
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
        )
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dimension,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=number_of_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=number_of_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_layer = nn.Linear(
            embedding_dimension,
            vocabulary_size,
        )

    def forward(self, token_ids):
        """Return next-token logits with shape (batch, sequence, vocabulary)."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")

        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        ).unsqueeze(0)
        positions = positions.expand(batch_size, sequence_length)

        hidden_states = self.token_embedding(token_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)

        # True values above the diagonal hide future tokens from attention.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )

        hidden_states = self.transformer(
            hidden_states,
            mask=causal_mask,
            is_causal=True,
        )
        hidden_states = self.final_norm(hidden_states)
        return self.output_layer(hidden_states)
