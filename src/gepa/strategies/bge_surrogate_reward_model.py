"""
BGE-M3 Reward Model for GEPA prompt optimisation.

Drop-in replacement for BERTRewardModel — same interface
(add_training_data(prompts, scores), predict(prompts), predict_ucb(prompts))
but uses BAAI/bge-m3 with Matryoshka embedding truncation under the hood.

The GEPA engine's reward_model interface works on prompt-level aggregate
scores (one score per prompt), NOT (prompt, question) pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_SAMPLES = 5
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 8
DEFAULT_MATRYOSHKA_DIM = 256  # Truncate bge-m3 embeddings to this dim


# ---------------------------------------------------------------------------
# BGE-M3 Embedding Cache
# ---------------------------------------------------------------------------
class _BGEEmbedder:
    """Lazily loads BAAI/bge-m3 and caches embeddings."""

    def __init__(self, matryoshka_dim: int = DEFAULT_MATRYOSHKA_DIM):
        self.matryoshka_dim = matryoshka_dim
        self._model = None
        self._cache: dict[str, torch.Tensor] = {}

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        print("  [BGE-M3] Loading BAAI/bge-m3 model ...")
        self._model = SentenceTransformer("BAAI/bge-m3")
        print(f"  [BGE-M3] Loaded. Matryoshka dim = {self.matryoshka_dim}")

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts into Matryoshka-truncated embeddings.

        Returns a tensor of shape [len(texts), matryoshka_dim].
        Uses a cache to avoid re-encoding identical texts.
        """
        self._load_model()

        # Find which texts need encoding
        uncached_texts = []
        for t in texts:
            if t not in self._cache:
                uncached_texts.append(t)

        # Batch-encode uncached texts
        if uncached_texts:
            raw_embeddings = self._model.encode(
                uncached_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Matryoshka truncation: just slice the first N dims
            truncated = raw_embeddings[:, : self.matryoshka_dim]
            # Re-normalise after truncation
            truncated = torch.nn.functional.normalize(truncated, p=2, dim=1)

            for j, text in enumerate(uncached_texts):
                self._cache[text] = truncated[j].cpu()

        # Assemble results
        result = torch.stack([self._cache[t] for t in texts])
        return result


# ---------------------------------------------------------------------------
# BGE-M3 Reward Model — drop-in replacement for BERTRewardModel
# ---------------------------------------------------------------------------
class BGESurrogateRewardModel(nn.Module):
    """
    BGE-M3 encoder + regression head  →  scalar score ∈ [0, 1].

    Drop-in replacement for BERTRewardModel.
    Same interface: add_training_data(prompts, scores), predict(prompts),
    predict_ucb(prompts), train_on_buffer().
    Uses BAAI/bge-m3 with Matryoshka truncation instead of BERT.
    """

    def __init__(
        self,
        matryoshka_dim: int = DEFAULT_MATRYOSHKA_DIM,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        super().__init__()
        self.matryoshka_dim = matryoshka_dim
        self.min_samples = min_samples

        # Shared BGE-M3 embedder (lazy loaded, cached)
        self._embedder = _BGEEmbedder(matryoshka_dim=matryoshka_dim)

        # Regression head: embedding → hidden → 1
        self.regression_head = nn.Sequential(
            nn.Linear(matryoshka_dim, matryoshka_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(matryoshka_dim // 2, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Online training buffer
        self._training_prompts: list[str] = []
        self._training_scores: list[float] = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return a batch of scalar scores ∈ [0, 1].

        Args:
            embeddings: [B, matryoshka_dim]
        """
        logit = self.regression_head(embeddings).squeeze(-1)  # [B]
        return torch.sigmoid(logit)

    # ------------------------------------------------------------------
    # Public API (matches BERTRewardModel interface)
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
        """
        Fine-tune on all accumulated data.

        Returns the final epoch's average loss, or ``None`` if there was
        not enough data to train.
        """
        if len(self._training_prompts) < self.min_samples:
            return None

        # Pre-compute all embeddings (cached, so fast for repeated texts)
        embeddings = self._embedder.encode(self._training_prompts).to(self.device)
        targets = torch.tensor(self._training_scores, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(embeddings, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.regression_head.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.regression_head.train()
        avg_loss = 0.0
        for _epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for emb, tgt in loader:
                optimizer.zero_grad()
                preds = self(emb)
                loss = loss_fn(preds, tgt)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

        self._is_trained = True
        self.regression_head.eval()
        return avg_loss

    @torch.no_grad()
    def predict(self, prompts: list[str]) -> list[float]:
        """Predict reward scores for a list of prompt texts."""
        self.regression_head.eval()
        embeddings = self._embedder.encode(prompts).to(self.device)
        scores = self(embeddings)
        return scores.cpu().tolist()

    @torch.no_grad()
    def predict_ucb(self, prompts: list[str], n_samples: int = 5) -> tuple[list[float], list[float]]:
        """
        Predict reward scores using MC Dropout to get mean and std for UCB ranking.
        Requires the model to have dropout layers active (train mode).
        """
        # Set to train mode to enable dropout
        self.regression_head.train()

        embeddings = self._embedder.encode(prompts).to(self.device)

        all_scores = []
        for _ in range(n_samples):
            scores = self(embeddings)
            all_scores.append(scores.cpu().unsqueeze(0))

        # Shape: [n_samples, batch_size]
        stacked_scores = torch.cat(all_scores, dim=0)

        # Calculate mean and std
        means = stacked_scores.mean(dim=0).tolist()
        stds = stacked_scores.std(dim=0).tolist()

        # Revert to eval mode
        self.regression_head.eval()

        return means, stds
