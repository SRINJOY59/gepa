from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gepa.core.callbacks import GEPACallback, ValsetEvaluatedEvent


class DetailedValidationLogger(GEPACallback):
    """Logs detailed scores for every validation example to a JSONL file.
    
    Optionally computes and logs the score on the FULL validation set as well,
    to measure the variation/gap between the subset (used for optimization) 
    and the true full validation set.
    """

    def __init__(self, log_file: str | Path, full_valset: list[dict] | None = None, adapter: Any | None = None):
        self.log_file = Path(log_file)
        self.full_valset = full_valset
        self.adapter = adapter
        
        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear file on init
        with open(self.log_file, "w") as f:
            pass

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        """Log scores when validation set is evaluated."""
        
        # 1. Base record with subset scores (from the event)
        record = {
            "iteration": event["iteration"],
            "candidate_idx": event["candidate_idx"],
            # "candidate": event["candidate"], # Optional: include full candidate if needed
            "subset_average_score": event["average_score"],
            "subset_num_examples": event["num_examples_evaluated"],
            "subset_scores_by_id": event["scores_by_val_id"], # dict[id, score]
        }

        # 2. Optional: Compute full validation score if provided
        if self.full_valset is not None and self.adapter is not None:
            # We must evaluate manually here. Note: this is synchronous and will slow down the loop!
            # Use capture_traces=False for speed.
            full_eval = self.adapter.evaluate(
                inputs=self.full_valset,
                candidate=event["candidate"],
                capture_traces=False
            )
            
            full_avg = sum(full_eval.scores) / len(full_eval.scores) if full_eval.scores else 0.0
            
            record["full_average_score"] = full_avg
            record["full_num_examples"] = len(self.full_valset)
            record["score_variation"] = full_avg - event["average_score"] # Positive means full set is better, negative means subset overestimated
            # record["full_scores_detailed"] = full_eval.scores # Optional: log all full scores (could be large)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
