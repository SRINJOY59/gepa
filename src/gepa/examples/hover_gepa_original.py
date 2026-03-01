#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Original GEPA on the HoVer dataset.

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
# Robust LLM Wrapper
# ──────────────────────────────────────────────────────────────────────
def robust_completion(model: str, messages: list[dict], api_keys: list[str], **kwargs) -> Any:
    """Wrapper around litellm.completion with API key fallback."""
    import litellm
    import time
    import logging

    if not api_keys:
        # Fallback to default Litellm behavior if list is empty
        return litellm.completion(model=model, messages=messages, **kwargs)

    last_err = None
    while api_keys:
        key = api_keys[0]
        try:
            return litellm.completion(model=model, messages=messages, api_key=key, **kwargs)
        except Exception as e:
            last_err = e
            logging.warning(f"API call failed with current key: {e}")
            logging.warning("Removing failed key from rotation and retrying with the next one...")
            api_keys.pop(0) # Remove the depleted key from the list for all future calls
            time.sleep(1) # tiny sleep before retry

    logging.error("All API keys failed.")
    raise last_err

# ──────────────────────────────────────────────────────────────────────
# HoVer Adapter (GEPA framework native)
# ──────────────────────────────────────────────────────────────────────
def create_hover_adapter(task_lm: str, api_keys: list[str]):
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
        def __init__(self, task_lm: str, api_keys: list[str]):
            self.task_lm = task_lm
            self.api_keys = api_keys

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
                    resp = robust_completion(
                        model=self.task_lm,
                        messages=[
                            {"role": "system", "content": verification_prompt},
                            {"role": "user", "content": f"Claim: {claim}"},
                        ],
                        api_keys=self.api_keys,
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

    return HoVerAdapter(task_lm, api_keys)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Original GEPA on HoVer"
    )
    # LLM
    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY", ""),
    )
    parser.add_argument("--task_lm", type=str, default="openrouter/qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--reflection_lm", type=str, default="openrouter/qwen/qwen-2.5-72b-instruct")

    # Dataset
    parser.add_argument("--train_size", type=int, default=20)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Multi-minibatch config - REMOVED

    # Budget
    parser.add_argument("--max_metric_calls", type=int, default=500)

    # Output
    parser.add_argument("--output_file", type=str,
                        default="results_hover_original.txt",
                        help="Path to the output .txt file for all results")
    
    # Validation Subset & Logging
    parser.add_argument("--val_subset_size", type=int, default=None,
                        help="Number of validation examples to use (random subset). If None, use full valset.")
    parser.add_argument("--detailed_log_file", type=str, default="validation_scores.jsonl",
                        help="Path to log detailed validation scores (JSONL).")

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
    
    # Extract all OPENROUTER_API_KEYs for fallback logic
    openrouter_keys = []
    for k, v in os.environ.items():
        if k.startswith("OPENROUTER_API_KEY") and v.strip():
            # sort by key name to maintain order 1,2,3... if possible
            openrouter_keys.append((k, v.strip()))
            
    openrouter_keys.sort(key=lambda x: x[0])
    api_keys_list = [v for k, v in openrouter_keys]
    
    if not api_keys_list:
        print("Warning: No defined OPENROUTER_API_KEYs found. Falling back to google_api_key.")
        api_keys_list = [args.google_api_key]
    else:
        print(f"Loaded {len(api_keys_list)} OpenRouter API keys for fallback.")

    # ── Load dataset ──
    trainset, valset, testset = load_hover_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # ── Subset Validation Set if requested ──
    full_valset = None
    if args.val_subset_size is not None:
        full_valset = valset # Keep reference to full set
        if args.val_subset_size < len(valset):
            print(f"Subsetting validation set from {len(valset)} to {args.val_subset_size} examples.")
            # Use a fixed seed for reproducibility of the subset
            rng = random.Random(args.seed)
            valset = rng.sample(valset, args.val_subset_size)
    
        # If Val subset is NOT requested, full_valset is just valset (but logic below handles None/Not None)
        # Actually logic: if val_subset_size is passed, we want to log variation.
        # If val_subset_size is passed but >= len(valset), subset == full set, variation is 0.

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
        api_keys=api_keys_list,
    )

    # ── Reflection LM ──
    def reflection_lm(prompt: str) -> str:
        resp = robust_completion(
            model=args.reflection_lm,
            messages=[{"role": "user", "content": prompt}],
            api_keys=api_keys_list,
        )
        return resp.choices[0].message.content

    # ── Callbacks ──
    from gepa.callbacks.detailed_logger import DetailedValidationLogger
    
    # We pass full_valset to logger IF we are subsetting, to compute the variation.
    # If args.val_subset_size is None, we are running on full set anyway, so no variation to log (or variation is 0).
    # But user might want to log "full score" explicitly even if it's the same.
    # Let's pass it if available.
    
    callbacks = [
        DetailedValidationLogger(
            log_file=args.detailed_log_file,
            full_valset=full_valset if args.val_subset_size is not None else None,
            adapter=adapter
        )
    ]
    print(f"Logging detailed validation scores to: {args.detailed_log_file}")

    # ── Logger ──
    from gepa.logging.logger import StdOutLogger

    logger = StdOutLogger()

    # ── Print config ──
    print(f"\n{'='*60}")
    print("Original GEPA on HoVer")
    print(f"{'='*60}")
    print(f"Task LM:             {args.task_lm}")
    print(f"Reflection LM:       {args.reflection_lm}")
    print(f"Max metric calls:    {args.max_metric_calls}")
    print(f"Val Subset Size:     {args.val_subset_size if args.val_subset_size else 'Full'}")
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

    # ── Create BERT Reward Model ──
    from gepa.strategies.bert_reward_model import BERTRewardModel
    reward_model = BERTRewardModel(min_samples=2)

    # ── Run GEPA - Original ──
    print("Starting Original GEPA optimization...\n")
    result = optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        # Multi-minibatch settings
        use_multi_minibatch=True,
        num_minibatches=3,
        candidates_per_minibatch=10, # Generate N=15 mutations
        use_bandit_mutation=True, # Critical to generate multi prompts
        top_k=3, # Run full inference only on top 3
        reward_model=reward_model,
        # Callbacks
        callbacks=callbacks,
        # General settings
        candidate_selection_strategy="pareto", # or whatever default
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

    # ── Close log file ──
    print(f"\nRun completed: {datetime.datetime.now().isoformat()}")
    print(f"Full output saved to: {args.output_file}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return result


if __name__ == "__main__":
    main()
