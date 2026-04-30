from typing import Optional

import torch
import torch.nn as nn


class AcquisitionConditionEmbedding(nn.Module):
    """
    Converts acquisition parameters into a single conditioning vector.

    Conditions:
    - manufacturer_label: discrete class label
    - field_strength_label: discrete class label
    - echo_time_norm: continuous scalar in [0, 1]

    The module also supports classifier-free guidance training through
    condition dropout. When a condition is dropped, the model receives a
    learned unconditional embedding instead of the true condition embedding.
    """

    def __init__(
        self,
        num_manufacturers: int = 2,
        num_field_strengths: int = 2,
        manufacturer_embedding_dim: int = 32,
        field_strength_embedding_dim: int = 32,
        echo_embedding_dim: int = 32,
        condition_dim: int = 128,
        dropout_prob: float = 0.10,
    ):
        super().__init__()

        if num_manufacturers <= 0:
            raise ValueError("num_manufacturers must be positive.")

        if num_field_strengths <= 0:
            raise ValueError("num_field_strengths must be positive.")

        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0 and 1.")

        self.num_manufacturers = num_manufacturers
        self.num_field_strengths = num_field_strengths
        self.condition_dim = condition_dim
        self.dropout_prob = dropout_prob

        self.manufacturer_embedding = nn.Embedding(
            num_embeddings=num_manufacturers,
            embedding_dim=manufacturer_embedding_dim,
        )

        self.field_strength_embedding = nn.Embedding(
            num_embeddings=num_field_strengths,
            embedding_dim=field_strength_embedding_dim,
        )

        self.echo_time_mlp = nn.Sequential(
            nn.Linear(1, echo_embedding_dim),
            nn.SiLU(),
            nn.Linear(echo_embedding_dim, echo_embedding_dim),
        )

        combined_dim = (
            manufacturer_embedding_dim
            + field_strength_embedding_dim
            + echo_embedding_dim
        )

        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.unconditional_embedding = nn.Parameter(
            torch.randn(condition_dim) * 0.02
        )

    def _validate_inputs(
        self,
        manufacturer_label: torch.Tensor,
        field_strength_label: torch.Tensor,
        echo_time_norm: torch.Tensor,
    ) -> None:
        if manufacturer_label.ndim != 1:
            raise ValueError(
                f"manufacturer_label must have shape [B], got {manufacturer_label.shape}"
            )

        if field_strength_label.ndim != 1:
            raise ValueError(
                f"field_strength_label must have shape [B], got {field_strength_label.shape}"
            )

        if echo_time_norm.ndim != 1:
            raise ValueError(
                f"echo_time_norm must have shape [B], got {echo_time_norm.shape}"
            )

        batch_size = manufacturer_label.shape[0]

        if field_strength_label.shape[0] != batch_size:
            raise ValueError("field_strength_label batch size mismatch.")

        if echo_time_norm.shape[0] != batch_size:
            raise ValueError("echo_time_norm batch size mismatch.")

    def forward(
        self,
        manufacturer_label: torch.Tensor,
        field_strength_label: torch.Tensor,
        echo_time_norm: torch.Tensor,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            condition_embedding: tensor of shape [B, condition_dim]

        During training, classifier-free condition dropout is applied randomly.
        During inference, use force_unconditional=True to get unconditional
        embeddings for classifier-free guidance.
        """
        self._validate_inputs(
            manufacturer_label=manufacturer_label,
            field_strength_label=field_strength_label,
            echo_time_norm=echo_time_norm,
        )

        device = manufacturer_label.device
        batch_size = manufacturer_label.shape[0]

        if force_unconditional:
            return self.unconditional_embedding.unsqueeze(0).expand(batch_size, -1)

        manufacturer_label = manufacturer_label.long()
        field_strength_label = field_strength_label.long()
        echo_time_norm = echo_time_norm.float().view(batch_size, 1)

        manufacturer_emb = self.manufacturer_embedding(manufacturer_label)
        field_strength_emb = self.field_strength_embedding(field_strength_label)
        echo_emb = self.echo_time_mlp(echo_time_norm)

        combined = torch.cat(
            [manufacturer_emb, field_strength_emb, echo_emb],
            dim=1,
        )

        condition_embedding = self.condition_mlp(combined)

        if self.training and self.dropout_prob > 0.0:
            drop_mask = (
                torch.rand(batch_size, device=device) < self.dropout_prob
            )

            if drop_mask.any():
                unconditional = self.unconditional_embedding.unsqueeze(0).expand(
                    batch_size, -1
                )
                condition_embedding = torch.where(
                    drop_mask.view(batch_size, 1),
                    unconditional,
                    condition_embedding,
                )

        return condition_embedding