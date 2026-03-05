#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
GEPA on the HoVer dataset — BGE-M3 Surrogate + Batch Eval.

Identical to hover_gepa_original.py except:
  1. BGE-M3 (Matryoshka) embeddings instead of BERT
  2. Batch evaluation (ThreadPoolExecutor) instead of sequential
  3. Comprehensive metrics logging (Rounds, LLM calls, Acc, Loss, Top-K)
  4. Output saved to hover_gepa_main.txt
"""

from __future__ import annotations

# ── stdlib imports (BEFORE any sys.path manipulation) ──
import datetime
import logging
import os
import sys
import time
import concurrent.futures
from functools import partial


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
litellm.suppress_debug_info = True

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from gepa import optimize
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# LLM call counter (thread-safe)
# ──────────────────────────────────────────────────────────────────────
import threading

class LLMCallCounter:
    """Thread-safe counter for tracking total LLM API calls."""
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def increment(self, n: int = 1):
        with self._lock:
            self._count += n

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

llm_call_counter = LLMCallCounter()


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

    ds = load_dataset("vincentkoc/hover-parquet")

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
        llm_call_counter.increment()
        return litellm.completion(model=model, messages=messages, **kwargs)

    last_err = None
    while api_keys:
        key = api_keys[0]
        try:
            llm_call_counter.increment()
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
# HoVer Adapter (GEPA framework native) — with BATCH evaluation
# ──────────────────────────────────────────────────────────────────────
def create_hover_adapter(task_lm: str, api_keys: list[str], batch_workers: int = 5):
    """Build a GEPAAdapter for the HoVer claim verification task.

    Uses ThreadPoolExecutor for parallel (batch) LLM evaluation.
    """
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    @dataclass
    class HoVerTrajectory:
        input_data: dict
        output_data: dict
        trace_info: dict

    class HoVerAdapter(GEPAAdapter):
        def __init__(self, task_lm: str, api_keys: list[str], batch_workers: int = 5):
            self.task_lm = task_lm
            self.api_keys = api_keys
            self.batch_workers = batch_workers

        # ── evaluate (BATCH — ThreadPoolExecutor) ────────────────────
        def evaluate(
            self,
            inputs: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch:
            verification_prompt = candidate.get("verification_prompt", "")

            def _eval_single(example):
                """Evaluate one claim. Runs in a thread."""
                try:
                    claim = example["claim"]
                    gold_label = example["label"]

                    resp = robust_completion(
                        model=self.task_lm,
                        messages=[
                            {"role": "system", "content": verification_prompt},
                            {"role": "user", "content": f"Claim: {claim}"},
                        ],
                        api_keys=list(self.api_keys),  # copy to avoid pop issues across threads
                    )
                    raw_response = resp.choices[0].message.content.strip()

                    pred_label = self._parse_label(raw_response)
                    score = 1.0 if pred_label == gold_label else 0.0

                    output = {
                        "claim": claim,
                        "gold_label": gold_label,
                        "predicted_label": pred_label,
                        "raw_response": raw_response,
                    }

                    trajectory = None
                    if capture_traces:
                        feedback = (
                            f"Predicted: {pred_label}, Gold: {gold_label}, "
                            f"{'✓ Correct' if score == 1.0 else '✗ Incorrect'}"
                        )
                        trajectory = HoVerTrajectory(
                            input_data=example,
                            output_data=output,
                            trace_info={
                                "verification_prompt": verification_prompt,
                                "feedback": feedback,
                            },
                        )

                    return output, score, trajectory

                except Exception as e:
                    return {"error": str(e)}, 0.0, None

            # ── Run all evaluations in parallel ──
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
                futures = [executor.submit(_eval_single, ex) for ex in inputs]

            # Collect results in original order
            outputs = []
            scores = []
            trajectories = [] if capture_traces else None

            for future in futures:
                output, score, traj = future.result()
                outputs.append(output)
                scores.append(score)
                if capture_traces:
                    trajectories.append(traj)

            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
            )

        # ── label parsing ───────────────────────────────────────────
        @staticmethod
        def _parse_label(raw: str) -> str:
            """Extract SUPPORTED / NOT_SUPPORTED from LLM response."""
            upper = raw.upper()
            if "NOT_SUPPORTED" in upper or "NOT SUPPORTED" in upper:
                return "NOT_SUPPORTED"
            if "NOT ENOUGH INFO" in upper or "NOT_ENOUGH_INFO" in upper:
                return "NOT_SUPPORTED"
            if "REFUTE" in upper:
                return "NOT_SUPPORTED"
            if "SUPPORTED" in upper:
                return "SUPPORTED"
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

    return HoVerAdapter(task_lm, api_keys, batch_workers)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GEPA on HoVer (BGE-M3 Surrogate + Batch Eval)"
    )
    # LLM
    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY", ""),
    )
    parser.add_argument("--task_lm", type=str, default="openrouter/qwen/qwen3-8b")
    parser.add_argument("--reflection_lm", type=str, default="openrouter/qwen/qwen3-8b")

    # Dataset
    parser.add_argument("--train_size", type=int, default=30)
    parser.add_argument("--val_size", type=int, default=15)
    parser.add_argument("--test_size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # Budget
    parser.add_argument("--max_metric_calls", type=int, default=500)

    # Batch eval
    parser.add_argument("--batch_workers", type=int, default=5,
                        help="Number of concurrent LLM calls for batch evaluation")

    # Output
    parser.add_argument("--output_file", type=str,
                        default="hover_gepa_main.txt",
                        help="Path to the output .txt file for all results")
    
    # Validation Subset & Logging
    parser.add_argument("--val_subset_size", type=int, default=None,
                        help="Number of validation examples to use (random subset). If None, use full valset.")
    parser.add_argument("--detailed_log_file", type=str, default="validation_scoresv2.jsonl",
                        help="Path to log detailed validation scores (JSONL).")

    args = parser.parse_args()

    # ── Set up output file (tee stdout + stderr) ──
    log_file = open(args.output_file, "w", encoding="utf-8")
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

    # ── Create adapter (with batch evaluation) ──
    adapter = create_hover_adapter(
        task_lm=args.task_lm,
        api_keys=api_keys_list,
        batch_workers=args.batch_workers,
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

    # ── Verbose Prompt Logger (with comprehensive metrics) ──
    class VerbosePromptLogger:
        """Callback that prints every prompt, score, selection decision,
        AND comprehensive metrics (Rounds, LLM calls, Acc, Loss, Top-K stats)."""

        def __init__(self):
            self.iteration_count = 0
            self.best_score = 0.0
            self.best_prompt = ""

        def on_optimization_start(self, event):
            print(f"\n{'='*70}")
            print("VERBOSE LOG: Optimization Starting")
            print(f"{'='*70}")
            print(f"  Train set size: {event['trainset_size']}")
            print(f"  Val set size:   {event['valset_size']}")
            print(f"\n  SEED PROMPT:")
            for comp, text in event['seed_candidate'].items():
                print(f"    [{comp}]:")
                for line in text.split('\n'):
                    print(f"      {line}")
            print(f"{'='*70}\n")

        def on_iteration_start(self, event):
            self.iteration_count = event['iteration']
            print(f"\n{'─'*70}")
            print(f"  ITERATION {event['iteration']}")
            print(f"{'─'*70}")

        def on_candidate_selected(self, event):
            print(f"\n  ▶ PARENT SELECTED (idx={event['candidate_idx']}, score={event['score']:.4f})")
            for comp, text in event['candidate'].items():
                print(f"    [{comp}]:")
                for line in text.split('\n'):
                    print(f"      {line}")

        def on_proposal_end(self, event):
            print(f"\n  ▶ NEW MUTATION PROPOSED (Iteration {event['iteration']}):")
            for comp, text in event['new_instructions'].items():
                print(f"    [{comp}]:")
                for line in text.split('\n'):
                    print(f"      {line}")

        def on_valset_evaluated(self, event):
            avg = event['average_score']
            is_best = event.get('is_best_program', False)
            marker = " ★ NEW BEST" if is_best else ""
            print(f"\n  ▶ VALIDATION SCORE: {avg:.4f}  (candidate idx={event['candidate_idx']}){marker}")
            print(f"    Examples evaluated: {event['num_examples_evaluated']}/{event['total_valset_size']}")
            # Show per-example scores
            scores_dict = event.get('scores_by_val_id', {})
            if scores_dict:
                correct = sum(1 for s in scores_dict.values() if s >= 1.0)
                total = len(scores_dict)
                print(f"    Correct: {correct}/{total}")
            # Print the full prompt being evaluated
            print(f"    PROMPT EVALUATED:")
            for comp, text in event['candidate'].items():
                print(f"      [{comp}]:")
                for line in text.split('\n'):
                    print(f"        {line}")

        def on_candidate_accepted(self, event):
            self.best_score = event['new_score']
            print(f"\n  ✅ CANDIDATE ACCEPTED (new idx={event['new_candidate_idx']}, score={event['new_score']:.4f})")

        def on_candidate_rejected(self, event):
            print(f"\n  ❌ CANDIDATE REJECTED")
            print(f"     Old score: {event['old_score']:.4f}, New score: {event['new_score']:.4f}")
            print(f"     Reason: {event['reason']}")

        def on_iteration_end(self, event):
            accepted = event['proposal_accepted']
            status = "ACCEPTED" if accepted else "REJECTED"
            print(f"\n  ── Iteration {event['iteration']} finished: {status}")

            # ── COMPREHENSIVE METRICS ──
            print(f"\n  {'='*55}")
            print(f"  METRICS SUMMARY  -  Round {event['iteration']}")
            print(f"  {'='*55}")
            print(f"  {'Round:':<30} {event['iteration']}")
            print(f"  {'Total LLM Calls:':<30} {llm_call_counter.count}")
            print(f"  {'Total Metric Calls:':<30} {event.get('total_metric_calls', 'N/A')}")

            # Validation accuracy (from event data)
            val_scores = event.get('val_aggregate_scores', [])
            if val_scores:
                current_best = max(val_scores)
                print(f"  {'Best Val Acc (current):':<30} {current_best:.4f}")
                print(f"  {'Top-K Avg Score:':<30} {sum(val_scores[-3:])/max(len(val_scores[-3:]),1):.4f}")
                print(f"  {'Top-K Max Score:':<30} {max(val_scores[-3:]):.4f}")
                print(f"  {'Top-K Min Score:':<30} {min(val_scores[-3:]):.4f}")

            print(f"  {'='*55}")
            print(f"{'─'*70}\n")

        def on_optimization_end(self, event):
            print(f"\n{'='*70}")
            print("VERBOSE LOG: Optimization Complete")
            print(f"{'='*70}")
            print(f"  Total iterations:   {event['total_iterations']}")
            print(f"  Total metric calls: {event['total_metric_calls']}")
            print(f"  Total LLM calls:    {llm_call_counter.count}")
            print(f"  Best candidate idx: {event['best_candidate_idx']}")
            print(f"{'='*70}\n")

    verbose_logger = VerbosePromptLogger()

    callbacks = [
        DetailedValidationLogger(
            log_file=args.detailed_log_file,
            full_valset=full_valset if args.val_subset_size is not None else None,
            adapter=adapter
        ),
        verbose_logger,
    ]
    print(f"Logging detailed validation scores to: {args.detailed_log_file}")

    # ── Logger ──
    from gepa.logging.logger import StdOutLogger

    logger = StdOutLogger()

    # ── Print config ──
    print(f"\n{'='*60}")
    print("GEPA on HoVer (BGE-M3 Surrogate + Batch Eval)")
    print(f"{'='*60}")
    print(f"Task LM:             {args.task_lm}")
    print(f"Reflection LM:       {args.reflection_lm}")
    print(f"Max metric calls:    {args.max_metric_calls}")
    print(f"Batch workers:       {args.batch_workers}")
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
    print(f"Seed Average Score ({len(testset)} samples): {seed_avg:.4f}")
    print(f"{'='*60}\n")

    # ── Create BGE-M3 Reward Model (instead of BERT) ──
    from gepa.strategies.bge_surrogate_reward_model import BGESurrogateRewardModel
    reward_model = BGESurrogateRewardModel(min_samples=2)

    # ── Run GEPA ──
    print("Starting GEPA optimization (BGE-M3 Surrogate + Batch Eval)...\n")
    result = optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        # Multi-minibatch settings
        use_multi_minibatch=True,
        num_minibatches=2,
        candidates_per_minibatch=5, # Generate N=5 mutations per minibatch
        use_bandit_mutation=True, # Critical to generate multi prompts
        top_k=3, # Run full inference only on top 3
        reward_model=reward_model,
        # Callbacks
        callbacks=callbacks,
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

    # ── Final Statistics ──
    print(f"{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total Rounds (iterations):    {len(result.val_aggregate_scores)}")
    print(f"Total LLM Calls:              {llm_call_counter.count}")
    print(f"Seed test score:              {seed_avg:.4f}")
    print(f"Optimized test score:         {best_avg:.4f}")
    print(f"Best validation score:        {best_score:.4f}")
    print(f"Embedding model:              BAAI/bge-m3 (Matryoshka)")
    print(f"Evaluation mode:              Batch ({args.batch_workers} workers)")
    print(f"{'='*60}")

    # ── Close log file ──
    print(f"\nRun completed: {datetime.datetime.now().isoformat()}")
    print(f"Full output saved to: {args.output_file}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return result


if __name__ == "__main__":
    main()
