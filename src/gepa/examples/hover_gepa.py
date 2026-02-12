
#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Multi-Minibatch GEPA + BERT Reward Model on the HoVer dataset.

HoVer (HOppy VERification) is a multi-hop fact verification dataset.
Each example contains a claim and a label (SUPPORTED / NOT_SUPPORTED).
GEPA optimises a ``verification_prompt`` that instructs an LLM to
classify claims correctly.
"""

from __future__ import annotations

# ── stdlib imports (BEFORE any sys.path manipulation) ──
import datetime
import logging
import os
import sys


# ──────────────────────────────────────────────────────────────────────
# Tee: duplicate stdout to a file
# ──────────────────────────────────────────────────────────────────────
class _TeeStream:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, text):
        self._original.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    # Forward any other attribute lookups to the original stream
    def __getattr__(self, name):
        return getattr(self._original, name)

# Ensure the gepa package is importable regardless of CWD.
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import argparse
import random
from dataclasses import dataclass
from typing import Any

import litellm

litellm.verbose = False

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from gepa import optimize
from gepa.strategies.bert_reward_model import BERTRewardModel
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
def load_hover_dataset(
    train_size: int = 200,
    val_size: int = 100,
    test_size: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the HoVer dataset (``hover-nlp/hover``).

    Each example is a dict with keys:
      - ``claim``  (str): the claim to verify
      - ``label``  (str): ``SUPPORTED`` or ``NOT_SUPPORTED``

    We combine the train and validation splits from HuggingFace (the
    test split has no labels) and then re-split to the requested sizes.
    """
    from datasets import load_dataset

    ds = load_dataset("hover-nlp/hover", trust_remote_code=True)

    # Combine train + validation (test has label = -1)
    examples = []
    for split_name in ("train", "validation"):
        if split_name not in ds:
            continue
        for x in ds[split_name]:
            label = x.get("label")
            # Skip unlabelled entries
            if label is None or label == -1 or label == "-1":
                continue
            # Normalise label to string
            if isinstance(label, int):
                label_str = "SUPPORTED" if label == 0 else "NOT_SUPPORTED"
            else:
                label_str = str(label).strip().upper()
            examples.append({
                "claim": x["claim"],
                "label": label_str,
            })

    random.Random(seed).shuffle(examples)

    total_needed = train_size + val_size + test_size
    if len(examples) < total_needed:
        print(
            f"Warning: Only {len(examples)} labelled examples available, "
            f"requested {total_needed}. Adjusting sizes proportionally."
        )
        ratio = len(examples) / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(examples) - train_size - val_size

    trainset = examples[:train_size]
    valset = examples[train_size : train_size + val_size]
    testset = examples[train_size + val_size : train_size + val_size + test_size]

    print(
        f"Loaded {len(trainset)} train, {len(valset)} val, "
        f"{len(testset)} test examples from HoVer"
    )
    return trainset, valset, testset


# ──────────────────────────────────────────────────────────────────────
# HoVer Adapter (GEPA framework native)
# ──────────────────────────────────────────────────────────────────────
def create_hover_adapter(task_lm: str, api_key: str):
    """Build a GEPAAdapter for the HoVer claim verification task.

    One optimisable prompt component:
      ``verification_prompt``  – instructs the LLM how to verify claims
    """
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    @dataclass
    class HoVerTrajectory:
        input_data: dict
        output_data: dict
        trace_info: dict

    class HoVerAdapter(GEPAAdapter):
        def __init__(self, task_lm: str, api_key: str):
            self.task_lm = task_lm
            self.api_key = api_key

        # ── evaluate ────────────────────────────────────────────────
        def evaluate(
            self,
            inputs: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch:
            outputs: list[dict] = []
            scores: list[float] = []
            trajectories: list | None = [] if capture_traces else None

            verification_prompt = candidate.get("verification_prompt", "")

            for example in inputs:
                try:
                    claim = example["claim"]
                    gold_label = example["label"]

                    # Single LLM call: verify the claim
                    resp = litellm.completion(
                        model=self.task_lm,
                        messages=[
                            {"role": "system", "content": verification_prompt},
                            {"role": "user", "content": f"Claim: {claim}"},
                        ],
                        api_key=self.api_key,
                    )
                    raw_response = resp.choices[0].message.content.strip()

                    # Parse prediction
                    pred_label = self._parse_label(raw_response)

                    # Binary scoring
                    score = 1.0 if pred_label == gold_label else 0.0

                    output = {
                        "claim": claim,
                        "gold_label": gold_label,
                        "predicted_label": pred_label,
                        "raw_response": raw_response,
                    }
                    outputs.append(output)
                    scores.append(score)

                    if capture_traces:
                        feedback = (
                            f"Predicted: {pred_label}, Gold: {gold_label}, "
                            f"{'✓ Correct' if score == 1.0 else '✗ Incorrect'}"
                        )
                        trajectories.append(
                            HoVerTrajectory(
                                input_data=example,
                                output_data=output,
                                trace_info={
                                    "verification_prompt": verification_prompt,
                                    "feedback": feedback,
                                },
                            )
                        )

                except Exception as e:
                    outputs.append({"error": str(e)})
                    scores.append(0.0)
                    if capture_traces:
                        trajectories.append(None)

            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
            )

        # ── label parsing ───────────────────────────────────────────
        @staticmethod
        def _parse_label(raw: str) -> str:
            """Extract SUPPORTED / NOT_SUPPORTED from LLM response.

            The seed prompt asks for SUPPORTED/REFUTED/NOT ENOUGH INFO.
            We map REFUTED and NOT ENOUGH INFO → NOT_SUPPORTED for
            scoring against HoVer's binary gold labels.
            """
            upper = raw.upper()
            # Check for NOT_SUPPORTED / NOT ENOUGH INFO / REFUTED first
            if "NOT_SUPPORTED" in upper or "NOT SUPPORTED" in upper:
                return "NOT_SUPPORTED"
            if "NOT ENOUGH INFO" in upper or "NOT_ENOUGH_INFO" in upper:
                return "NOT_SUPPORTED"
            if "REFUTE" in upper:
                return "NOT_SUPPORTED"
            if "SUPPORTED" in upper:
                return "SUPPORTED"
            # Fallback heuristics
            if "FALSE" in upper or "NOT" in upper:
                return "NOT_SUPPORTED"
            return "SUPPORTED"

        # ── reflective dataset ──────────────────────────────────────
        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_result: EvaluationBatch,
            components_to_update: list[str],
        ) -> dict[str, list[dict]]:
            """Build per-component feedback for GEPA reflection."""
            reflective_data: dict[str, list[dict]] = {}
            trajs = eval_result.trajectories or [None] * len(eval_result.outputs)

            for component in components_to_update:
                entries: list[dict] = []
                for out, score, traj in zip(
                    eval_result.outputs, eval_result.scores, trajs
                ):
                    if traj is None:
                        continue
                    entries.append(
                        {
                            "input": traj.input_data.get("claim", ""),
                            "current_instruction": candidate.get(component, ""),
                            "output": out,
                            "score": score,
                            "feedback": traj.trace_info.get("feedback", ""),
                        }
                    )
                reflective_data[component] = entries

            return reflective_data

    return HoVerAdapter(task_lm, api_key)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Minibatch GEPA + BERT Reward Model on HoVer"
    )
    # LLM
    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY", ""),
    )
    parser.add_argument("--task_lm", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--reflection_lm", type=str, default="gemini/gemini-2.0-flash")

    # Dataset
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Multi-minibatch config
    parser.add_argument("--num_minibatches", type=int, default=5,
                        help="Number of minibatches per iteration (M)")
    parser.add_argument("--candidates_per_minibatch", type=int, default=4,
                        help="Candidates generated per minibatch (N) for surrogate scoring")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top-K candidates selected by surrogate for actual evaluation")
    
    # Budget
    parser.add_argument("--max_metric_calls", type=int, default=500)

    # BERT Reward Model
    parser.add_argument("--bert_min_samples", type=int, default=1,
                        help="Minimum training samples before reward model starts predicting (default=1 for online)")

    # Output
    parser.add_argument("--output_file", type=str,
                        default="results_hover_multi_minibatch.txt",
                        help="Path to the output .txt file for all results")

    args = parser.parse_args()

    # ── Set up output file (tee stdout + stderr) ──
    log_file = open(args.output_file, "w")
    log_file.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
    log_file.write(f"{'='*60}\n\n")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    print(f"Logging all output to: {args.output_file}")

    if not args.google_api_key:
        raise ValueError(
            "Provide --google_api_key or set GOOGLE_API_KEY env variable"
        )

    # ── Load dataset ──
    trainset, valset, testset = load_hover_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # ── Seed candidate (from GEPA paper, arXiv 2507.19457) ──
    seed_candidate = {
        "verification_prompt": (
            "You are a fact-checking assistant.\n"
            "Given a claim and a set of evidence passages, determine whether "
            "the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO based only "
            "on the evidence.\n"
            "Carefully reason over the evidence before answering.\n"
            "Output the final label only."
        ),
    }

    # ── Create adapter ──
    adapter = create_hover_adapter(
        task_lm=args.task_lm,
        api_key=args.google_api_key,
    )

    # ── Reflection LM ──
    def reflection_lm(prompt: str) -> str:
        resp = litellm.completion(
            model=args.reflection_lm,
            messages=[{"role": "user", "content": prompt}],
            api_key=args.google_api_key,
        )
        return resp.choices[0].message.content

    # ── Reward Model (BERT embeddings + MLP head) ──
    reward_model = BERTRewardModel(min_samples=args.bert_min_samples)

    # ── Logger ──
    from gepa.logging.logger import StdOutLogger

    logger = StdOutLogger()

    # ── Print config ──
    print(f"\n{'='*60}")
    print("Multi-Minibatch GEPA + BERT Reward Model on HoVer")
    print(f"{'='*60}")
    print(f"Task LM:             {args.task_lm}")
    print(f"Reflection LM:       {args.reflection_lm}")
    print(f"Minibatches (M):     {args.num_minibatches}")
    print(f"Candidates/MB (N):   {args.candidates_per_minibatch}")
    print(f"Top-K:               {args.top_k}")
    print(f"Max metric calls:    {args.max_metric_calls}")
    print(f"RM min samples:      {args.bert_min_samples}")
    print(f"{'='*60}\n")

    # ── Evaluate seed on test set ──
    print(f"{'='*60}")
    print("Evaluating SEED candidate on Test Set")
    print(f"{'='*60}")
    seed_eval = adapter.evaluate(testset, seed_candidate)
    seed_avg = (
        sum(seed_eval.scores) / len(seed_eval.scores)
        if seed_eval.scores
        else 0.0
    )
    print(f"Seed Average Score (10 samples): {seed_avg:.4f}")
    print(f"{'='*60}\n")

    # ── Run GEPA with multi-minibatch + BERT ──
    print("Starting multi-minibatch GEPA optimization...\n")
    result = optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        # Multi-minibatch settings
        use_multi_minibatch=True,
        num_minibatches=args.num_minibatches,
        candidates_per_minibatch=args.candidates_per_minibatch,
        top_k=args.top_k,
        reward_model=reward_model,
        # General settings
        candidate_selection_strategy="pareto",
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
        logger=logger,
        display_progress_bar=True,
        track_best_outputs=True,
    )

    # ── Results ──
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"Best Validation Score: {best_score:.4f}")
    print(f"Best Candidate (Index {result.best_idx}):")
    for k, v in result.best_candidate.items():
        print(f"\nComponent '{k}':\n{v}")

    # ── Evaluate optimised on test set ──
    print(f"\n{'='*60}")
    print("Evaluating OPTIMIZED candidate on Test Set")
    print(f"{'='*60}")
    best_eval = adapter.evaluate(testset, result.best_candidate)
    best_avg = (
        sum(best_eval.scores) / len(best_eval.scores)
        if best_eval.scores
        else 0.0
    )
    print(f"Optimized Average Score: {best_avg:.4f}")
    print(f"Improvement over seed:   {best_avg - seed_avg:+.4f}")
    print(f"{'='*60}\n")

    # ── Reward model stats ──
    if reward_model.is_trained:
        print(f"BERT Reward Model: trained on {len(reward_model._training_prompts)} samples")
    else:
        print("BERT Reward Model: never reached training threshold")

    # ── Close log file ──
    print(f"\nRun completed: {datetime.datetime.now().isoformat()}")
    print(f"Full output saved to: {args.output_file}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return result


if __name__ == "__main__":
    main()