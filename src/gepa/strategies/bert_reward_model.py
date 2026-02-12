# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Lightweight reward model for fast prompt candidate scoring.

Uses TF-IDF features + a small 2-layer MLP instead of a full transformer
so that both training and inference are near-instantaneous.  The model is
trained fully online — it learns from every new (prompt, score) pair
without any minimum-sample gate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TFIDF_FEATURES = 512
DEFAULT_HIDDEN_DIM = 64
DEFAULT_TRAIN_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Reward model  (TF-IDF → MLP → scalar score)
# ---------------------------------------------------------------------------
class BERTRewardModel(nn.Module):
    """Lightweight TF-IDF + MLP reward model.

    Despite the class name (kept for backward-compat), this does **not** use
    BERT.  It encodes prompts via scikit-learn TF-IDF and feeds the sparse
    vectors through a tiny 2-layer MLP.

    Usage::

        rm = BERTRewardModel()
        rm.add_training_data(prompts, scores)
        rm.train_on_buffer()
        predictions = rm.predict(candidate_prompts)
    """

    def __init__(
        self,
        max_features: int = DEFAULT_TFIDF_FEATURES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        min_samples: int = 1,           # kept for API compat, default=1
    ):
        super().__init__()
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        self.min_samples = min_samples

        # TF-IDF vectoriser (rebuilt on each train pass with full vocab)
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            sublinear_tf=True,
        )

        # Small MLP: input_dim → hidden → 1
        self._mlp = nn.Sequential(
            nn.Linear(max_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mlp.to(self.device)

        # Online training buffer
        self._training_prompts: list[str] = []
        self._training_scores: list[float] = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Forward (operates on dense TF-IDF feature tensors)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, max_features) dense float tensor → (batch,) scores."""
        return self._mlp(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def add_training_data(self, prompts: list[str], scores: list[float]) -> None:
        """Append new (prompt, score) pairs to the training buffer."""
        assert len(prompts) == len(scores)
        self._training_prompts.extend(prompts)
        self._training_scores.extend(scores)

    def train_on_buffer(
        self,
        epochs: int = DEFAULT_TRAIN_EPOCHS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> float | None:
        """Train on all accumulated data.  Returns final epoch loss, or None
        if there isn't enough data yet (< min_samples)."""
        if len(self._training_prompts) < self.min_samples:
            return None

        # Re-fit TF-IDF on entire buffer
        tfidf_matrix = self._vectorizer.fit_transform(self._training_prompts)
        # Convert sparse → dense numpy → torch tensor
        X = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32).to(self.device)
        y = torch.tensor(self._training_scores, dtype=torch.float32).to(self.device)

        n = X.shape[0]
        optimizer = torch.optim.Adam(self._mlp.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self._mlp.train()
        avg_loss = 0.0
        for _epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X[idx]
                yb = y[idx]

                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

        self._is_trained = True
        self._mlp.eval()
        return avg_loss

    @torch.no_grad()
    def predict(self, prompts: list[str]) -> list[float]:
        """Predict reward scores for a list of prompt texts."""
        self._mlp.eval()
        tfidf_matrix = self._vectorizer.transform(prompts)
        X = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32).to(self.device)
        scores = self(X)
        return scores.cpu().tolist()
