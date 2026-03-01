from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_SAMPLES = 5
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 256


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PromptRewardDataset(Dataset):
    """Simple dataset mapping (prompt_text, score) → tokenised input + target."""

    def __init__(
        self,
        prompts: list[str],
        scores: list[float],
        tokenizer: BertTokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.prompts = prompts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------
class BERTRewardModel(nn.Module):
    """
    BERT encoder + linear regression head  →  scalar score ∈ [0, 1].

    Usage::

        rm = BERTRewardModel()
        # ... optimiser discovers (prompt, valset_avg_score) pairs ...
        rm.add_training_data(prompts, scores)
        rm.train_on_buffer()
        predictions = rm.predict(candidate_prompts)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = DEFAULT_MAX_LENGTH,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        super().__init__()
        self.max_length = max_length
        self.min_samples = min_samples

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert: BertModel = BertModel.from_pretrained(model_name)
        self.regression_head = nn.Linear(self.bert.config.hidden_size, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Online training buffer
        self._training_prompts: list[str] = []
        self._training_scores: list[float] = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a batch of scalar scores ∈ [0, 1]."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] pooling
        logit = self.regression_head(cls_embedding).squeeze(-1)
        return torch.sigmoid(logit)

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
        """
        Fine-tune on all accumulated data.

        Returns the final epoch's average loss, or ``None`` if there was
        not enough data to train.
        """
        if len(self._training_prompts) < self.min_samples:
            return None

        dataset = PromptRewardDataset(
            self._training_prompts,
            self._training_scores,
            self.tokenizer,
            self.max_length,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.train()
        avg_loss = 0.0
        for _epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                targets = batch["score"].to(self.device)

                optimizer.zero_grad()
                preds = self(ids, mask)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

        self._is_trained = True
        self.eval()
        return avg_loss

    @torch.no_grad()
    def predict(self, prompts: list[str]) -> list[float]:
        """Predict reward scores for a list of prompt texts."""
        self.eval()
        encodings = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = encodings["input_ids"].to(self.device)
        mask = encodings["attention_mask"].to(self.device)
        scores = self(ids, mask)
        return scores.cpu().tolist()

    @torch.no_grad()
    def predict_ucb(self, prompts: list[str], n_samples: int = 5) -> tuple[list[float], list[float]]:
        """
        Predict reward scores using MC Dropout to get mean and std for UCB ranking.
        Requires the model to have dropout layers active (train mode).
        """
        # Set to train mode to enable dropout
        self.train()
        
        encodings = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = encodings["input_ids"].to(self.device)
        mask = encodings["attention_mask"].to(self.device)
        
        all_scores = []
        for _ in range(n_samples):
            # Forward pass with dropout active
            scores = self(ids, mask)
            all_scores.append(scores.cpu().unsqueeze(0))
            
        # Shape: [n_samples, batch_size]
        stacked_scores = torch.cat(all_scores, dim=0)
        
        # Calculate mean and std
        means = stacked_scores.mean(dim=0).tolist()
        stds = stacked_scores.std(dim=0).tolist()
        
        # Revert to eval mode
        self.eval()
        
        return means, stds
