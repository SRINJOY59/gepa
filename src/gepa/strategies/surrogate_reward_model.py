"""
Surrogate Reward Model for GEPA prompt optimisation.

Scores (prompt_text, validation_question) pairs cheaply using BERT
embeddings fed through an MLP.  The overall reward for a prompt is the
mean predicted score across all validation questions.

Training data is collected online: each fully-validated prompt yields
(prompt, question, actual_score) triples that are appended to an
ever-growing buffer and used to retrain the MLP via SGD.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_SAMPLES = 5
DEFAULT_TRAIN_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256
DEFAULT_HIDDEN_DIM = 256


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PromptQuestionDataset(Dataset):
    """Maps (prompt, question, score) → tokenised inputs + target."""

    def __init__(
        self,
        prompts: list[str],
        questions: list[str],
        scores: list[float],
        tokenizer: BertTokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        assert len(prompts) == len(questions) == len(scores)
        self.prompts = prompts
        self.questions = questions
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt_enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        question_enc = self.tokenizer(
            self.questions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "question_input_ids": question_enc["input_ids"].squeeze(0),
            "question_attention_mask": question_enc["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------
class SurrogateRewardModel(nn.Module):
    """BERT embeddings of (prompt, question) → MLP → scalar score ∈ [0, 1].

    The MLP takes ``phi(prompt, question) = concat(BERT_CLS(prompt),
    BERT_CLS(question))`` and outputs a predicted score.

    Public API
    ----------
    add_training_data(prompts, questions, scores)
        Append triples to the training buffer.
    train_on_buffer(...)
        Retrain the MLP using SGD + MSE on all accumulated data.
    predict(prompts, questions)
        Return predicted scores for (prompt, question) pairs.
    predict_with_ucb(prompts, questions, exploration_weight)
        Return UCB scores: predicted mean + exploration bonus.
    predict_prompt_scores(prompt_texts, validation_questions)
        For each prompt text return the mean predicted score across all
        validation questions.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = DEFAULT_MAX_LENGTH,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        super().__init__()
        self.max_length = max_length
        self.min_samples = min_samples

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert: BertModel = BertModel.from_pretrained(model_name)

        # Freeze BERT parameters — only train the MLP head
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_dim = self.bert.config.hidden_size  # 768 for base

        # MLP: concat([CLS]_prompt, [CLS]_question) → hidden → 1
        self.mlp = nn.Sequential(
            nn.Linear(bert_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Online training buffer
        self._training_prompts: list[str] = []
        self._training_questions: list[str] = []
        self._training_scores: list[float] = []
        self._is_trained: bool = False

        # Visit counts for UCB (keyed by prompt text)
        self._visit_counts: dict[str, int] = defaultdict(int)
        self._total_visits: int = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return batch of scalar scores ∈ [0, 1]."""
        with torch.no_grad():
            prompt_out = self.bert(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
            )
            question_out = self.bert(
                input_ids=question_input_ids,
                attention_mask=question_attention_mask,
            )

        prompt_cls = prompt_out.last_hidden_state[:, 0, :]    # [B, 768]
        question_cls = question_out.last_hidden_state[:, 0, :]  # [B, 768]

        phi = torch.cat([prompt_cls, question_cls], dim=-1)  # [B, 1536]
        logit = self.mlp(phi).squeeze(-1)                    # [B]
        return torch.sigmoid(logit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def add_training_data(
        self,
        prompts: list[str],
        questions: list[str],
        scores: list[float],
    ) -> None:
        """Append (prompt, question, score) triples to the training buffer."""
        assert len(prompts) == len(questions) == len(scores)
        self._training_prompts.extend(prompts)
        self._training_questions.extend(questions)
        self._training_scores.extend(scores)

    def train_on_buffer(
        self,
        epochs: int = DEFAULT_TRAIN_EPOCHS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> float | None:
        """Retrain the MLP on all accumulated data using SGD + MSE.

        Returns the final-epoch average loss, or ``None`` if insufficient
        data.
        """
        if len(self._training_prompts) < self.min_samples:
            return None

        dataset = PromptQuestionDataset(
            self._training_prompts,
            self._training_questions,
            self._training_scores,
            self.tokenizer,
            self.max_length,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # SGD as specified by the user
        optimizer = torch.optim.SGD(self.mlp.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.mlp.train()
        avg_loss = 0.0
        for _epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                p_ids = batch["prompt_input_ids"].to(self.device)
                p_mask = batch["prompt_attention_mask"].to(self.device)
                q_ids = batch["question_input_ids"].to(self.device)
                q_mask = batch["question_attention_mask"].to(self.device)
                targets = batch["score"].to(self.device)

                optimizer.zero_grad()
                preds = self(p_ids, p_mask, q_ids, q_mask)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

        self._is_trained = True
        self.mlp.eval()
        return avg_loss

    @torch.no_grad()
    def predict(self, prompts: list[str], questions: list[str]) -> list[float]:
        """Predict scores for a list of (prompt, question) pairs."""
        self.mlp.eval()
        prompt_enc = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        question_enc = self.tokenizer(
            questions,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        scores = self(
            prompt_enc["input_ids"].to(self.device),
            prompt_enc["attention_mask"].to(self.device),
            question_enc["input_ids"].to(self.device),
            question_enc["attention_mask"].to(self.device),
        )
        return scores.cpu().tolist()

    def predict_with_ucb(
        self,
        prompts: list[str],
        questions: list[str],
        exploration_weight: float = 1.0,
    ) -> list[float]:
        """Return UCB scores: ``predicted_mean + c * sqrt(ln(N) / n_i)``."""
        pred_scores = self.predict(prompts, questions)
        ucb_scores: list[float] = []
        for prompt_text, score in zip(prompts, pred_scores):
            n_i = max(self._visit_counts[prompt_text], 1)
            N = max(self._total_visits, 1)
            bonus = exploration_weight * math.sqrt(math.log(N + 1) / n_i)
            ucb_scores.append(score + bonus)
        return ucb_scores

    def predict_prompt_scores(
        self,
        prompt_texts: list[str],
        validation_questions: list[str],
        use_ucb: bool = False,
        exploration_weight: float = 1.0,
    ) -> list[float]:
        """For each prompt, compute the mean predicted score across all
        validation questions.

        Returns one score per prompt.
        """
        all_scores: list[float] = []
        for prompt_text in prompt_texts:
            # Tile prompt_text across all validation questions
            prompts_tiled = [prompt_text] * len(validation_questions)
            if use_ucb and self._is_trained:
                scores = self.predict_with_ucb(
                    prompts_tiled,
                    validation_questions,
                    exploration_weight=exploration_weight,
                )
            elif self._is_trained:
                scores = self.predict(prompts_tiled, validation_questions)
            else:
                # Untrained: return random exploration score
                import random as _rng
                scores = [_rng.random() for _ in validation_questions]

            all_scores.append(sum(scores) / max(len(scores), 1))
        return all_scores

    def record_visit(self, prompt_text: str) -> None:
        """Increment visit counter (for UCB exploration bonus)."""
        self._visit_counts[prompt_text] += 1
        self._total_visits += 1
