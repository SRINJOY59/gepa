#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
GEPA on the PUPA dataset — Local HF Model (Llama-3.1-8B) + BGE-M3 Surrogate + Metrics.

Based on pupa_gepa_original.py with the following changes:
  1. Uses a local HuggingFace model (meta-llama/Llama-3.1-8B-Instruct) instead of OpenRouter/litellm
  2. BGE-M3 (Matryoshka) reward model instead of BERT (like hover_gepa_main.py)
  3. Comprehensive metrics logging (Rounds, LLM calls, Acc, Loss, Top-K)
  4. VerbosePromptLogger callback (from hover_gepa_main.py)
"""

from __future__ import annotations

# ── stdlib imports ──
import datetime
import logging
import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Any

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

    def __getattr__(self, name):
        return getattr(self._original, name)


# Ensure the gepa package is importable regardless of CWD.
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import argparse
import random

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("tokenizers").setLevel(logging.WARNING)

from gepa import optimize
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# LLM call counter (thread-safe)
# ──────────────────────────────────────────────────────────────────────
class LLMCallCounter:
    """Thread-safe counter for tracking total local LLM inference calls."""
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
# Local HF Model Loader
# ──────────────────────────────────────────────────────────────────────
_local_model = None
_local_tokenizer = None
_model_lock = threading.Lock()


def load_local_model(model_name: str):
    """Lazily load the local HF model and tokenizer (once)."""
    global _local_model, _local_tokenizer
    with _model_lock:
        if _local_model is not None:
            return _local_model, _local_tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n[LOCAL-LLM] Loading model: {model_name} ...")
        start = time.time()

        _local_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token

        _local_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        _local_model.eval()

        elapsed = time.time() - start
        print(f"[LOCAL-LLM] Model loaded in {elapsed:.1f}s")
        print(f"[LOCAL-LLM] Device map: {_local_model.hf_device_map if hasattr(_local_model, 'hf_device_map') else 'N/A'}")
        print()

        return _local_model, _local_tokenizer


def local_completion(
    model_name: str,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Run inference on the local HF model using chat template.

    Args:
        model_name: HF model identifier (used for lazy loading).
        messages: List of {"role": ..., "content": ...} dicts.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to sample or use greedy decoding.

    Returns:
        The generated text (assistant response only).
    """
    import torch

    model, tokenizer = load_local_model(model_name)

    llm_call_counter.increment()

    # Apply chat template to convert messages to model input
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
def load_pupa_dataset(
    train_size: int = 225,
    val_size: int = 225,
    test_size: int = 214,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    from datasets import load_dataset

    pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")
    examples = [
        {
            "input": x["user_query"],
            "target_response": x["target_response"],
            "pii_units": x["pii_units"],
        }
        for x in pupa_new["train"]
    ]

    random.Random(seed).shuffle(examples)

    trainset = examples[:train_size]
    valset = examples[train_size : train_size + val_size]
    testset = examples[train_size + val_size : train_size + val_size + test_size]

    print(
        f"Loaded {len(trainset)} train, {len(valset)} val, "
        f"{len(testset)} test examples"
    )
    return trainset, valset, testset


# ──────────────────────────────────────────────────────────────────────
# PAPILLON Adapter (with local HF model)
# ──────────────────────────────────────────────────────────────────────
def create_papillon_adapter(model_name: str, max_new_tokens: int = 1024):
    """Build a GEPAAdapter for the PUPA task using a local HF model."""
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    @dataclass
    class PapillonTrajectory:
        input_data: dict
        output_data: dict
        trace_info: dict

    class PAPILLONAdapter(GEPAAdapter):
        def __init__(self, model_name: str, max_new_tokens: int = 1024):
            self.model_name = model_name
            self.max_new_tokens = max_new_tokens

        def _call_llm(self, messages: list[dict]) -> str:
            """Call the local HF model."""
            return local_completion(
                model_name=self.model_name,
                messages=messages,
                max_new_tokens=self.max_new_tokens,
            )

        def evaluate(
            self,
            inputs: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch:
            outputs: list[dict] = []
            scores: list[float] = []
            trajectories: list | None = [] if capture_traces else None

            redaction_prompt = candidate.get("redaction_prompt", "")
            response_prompt = candidate.get("response_prompt", "")

            for example in inputs:
                try:
                    user_query = example["input"]

                    # Step 1: Redact PII
                    redacted_request = self._call_llm([
                        {"role": "system", "content": redaction_prompt},
                        {"role": "user", "content": f"User query: {user_query}"},
                    ])

                    # Step 2: Untrusted LM (same local model)
                    external_response = self._call_llm([
                        {"role": "user", "content": redacted_request},
                    ])

                    # Step 3: Final response with context
                    final_text = self._call_llm([
                        {"role": "system", "content": response_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Original user query: {user_query}\n\n"
                                f"External LM request: {redacted_request}\n\n"
                                f"External LM response: {external_response}\n\n"
                                "Generate the final response for the user."
                            ),
                        },
                    ])

                    output = {
                        "redacted_request": redacted_request,
                        "external_response": external_response,
                        "final_response": final_text,
                    }

                    score, feedback = self._evaluate_single(example, output)
                    outputs.append(output)
                    scores.append(score)

                    if capture_traces:
                        trajectories.append(
                            PapillonTrajectory(
                                input_data=example,
                                output_data=output,
                                trace_info={
                                    "redaction_prompt": redaction_prompt,
                                    "response_prompt": response_prompt,
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

        def _llm_judge_quality(
            self,
            user_query: str,
            target_response: str,
            pred_response: str,
        ) -> bool:
            prompt = (
                "You are comparing the quality of two responses.\n"
                "Is response A at least as good as response B?\n\n"
                f"User Query: {user_query}\n\n"
                f"Response A (Prediction): {pred_response}\n\n"
                f"Response B (Target): {target_response}\n\n"
                "Return ONLY 'true' or 'false'."
            )
            try:
                resp = self._call_llm([{"role": "user", "content": prompt}])
                return "true" in resp.lower()
            except Exception:
                return False

        def _llm_judge_leakage(
            self,
            pii_units: str,
            redacted_request: str,
        ) -> float:
            pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
            if not pii_list:
                return 0.0

            prompt = (
                "Count the number of information pieces in `pii` that "
                "(i) are forms of PII *and* (ii) are leaked into the prompt "
                "given. Don't count celebrity names. Don't count things that "
                "do not appear in the prompt.\n\n"
                f"pii: {pii_list}\n"
                f"prompt: {redacted_request}\n\n"
                "Return ONLY a number."
            )
            try:
                resp = self._call_llm([{"role": "user", "content": prompt}])
                leaked = int(resp.strip())
            except Exception:
                leaked = sum(
                    1
                    for p in pii_list
                    if p.lower() in redacted_request.lower()
                )

            return leaked / len(pii_list)

        def _evaluate_single(
            self, example: dict, output: dict
        ) -> tuple[float, dict]:
            pii_units = example.get("pii_units", "")
            redacted_request = output.get("redacted_request", "")
            final_response = output.get("final_response", "")
            target_response = example.get("target_response", "")

            leakage = self._llm_judge_leakage(pii_units, redacted_request)
            privacy_score = 1.0 - leakage

            is_good = self._llm_judge_quality(
                example["input"], target_response, final_response
            )
            quality_score = 1.0 if is_good else 0.0

            overall = (privacy_score + quality_score) / 2.0

            feedback_text = (
                f"Overall={overall:.2f} "
                f"(quality={quality_score:.2f}, privacy={privacy_score:.2f}). "
                f"{'Reduce PII leakage!' if privacy_score < 0.8 else 'Good privacy!'} "
                f"{'Improve response quality!' if quality_score < 0.5 else 'Good quality!'}"
            )

            return overall, {
                "privacy": privacy_score,
                "quality": quality_score,
                "overall": overall,
                "feedback": feedback_text,
            }

        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_result: EvaluationBatch,
            components_to_update: list[str],
        ) -> dict[str, list[dict]]:
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
                            "input": traj.input_data.get("input", ""),
                            "current_instruction": candidate.get(component, ""),
                            "output": out,
                            "score": score,
                            "feedback": traj.trace_info.get("feedback", {}),
                        }
                    )
                reflective_data[component] = entries

            return reflective_data

    return PAPILLONAdapter(model_name, max_new_tokens)


# ──────────────────────────────────────────────────────────────────────
# Verbose Prompt Logger (from hover_gepa_main.py — with comprehensive metrics)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GEPA on PUPA — Local HF Model (Llama-3.1-8B) + BGE-M3 Surrogate + Metrics"
    )

    # Local HF model
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name/path for local inference")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max new tokens for generation")

    # Dataset
    parser.add_argument("--train_size", type=int, default=20)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Budget
    parser.add_argument("--max_metric_calls", type=int, default=500)

    # Output
    parser.add_argument("--output_file", type=str,
                        default="results_pupa_local_llm.txt",
                        help="Path to the output .txt file for all results")

    # Validation Subset & Logging
    parser.add_argument("--val_subset_size", type=int, default=None,
                        help="Number of validation examples to use (random subset). If None, use full valset.")
    parser.add_argument("--detailed_log_file", type=str,
                        default="validation_scores_pupa_local.jsonl",
                        help="Path to log detailed validation scores (JSONL).")

    args = parser.parse_args()

    # ── Set up output file (tee stdout + stderr) ──
    log_file = open(args.output_file, "w", encoding="utf-8")
    log_file.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
    log_file.write(f"{'='*60}\n\n")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    print(f"Logging all output to: {args.output_file}")

    # ── Pre-load the local model ──
    print(f"\nPre-loading local HF model: {args.model_name}")
    load_local_model(args.model_name)

    # ── Load dataset ──
    trainset, valset, testset = load_pupa_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    full_valset = None
    if args.val_subset_size is not None:
        full_valset = valset
        if args.val_subset_size < len(valset):
            print(f"Subsetting validation set from {len(valset)} to {args.val_subset_size} examples.")
            rng = random.Random(args.seed)
            valset = rng.sample(valset, args.val_subset_size)

    # ── Seed candidate (from DSPy tutorial) ──
    seed_candidate = {
        "redaction_prompt": (
            "Given a private user query, create a privacy-preserving request "
            "for a powerful external LLM. The LLM may assist without learning "
            "private information about the user.\n\n"
            "IMPORTANT: Output ONLY the privacy-preserving request. "
            "Do not include any reasoning, explanations, or conversational filler."
        ),
        "response_prompt": (
            "Respond to a user query. For inspiration, we found a potentially "
            "related request to a powerful external LLM and its response.\n\n"
            "Input:\n"
            "1. related_llm_request: The privacy-preserving request sent to "
            "the external LLM.\n"
            "2. related_llm_response: Information from a powerful LLM "
            "responding to the related request.\n"
            "3. user_query: The user's original request you need to fulfill.\n\n"
            "Instruction: Generate your final response to the user's request. "
            "Output ONLY the response."
        ),
    }

    # ── Create adapter (local HF model) ──
    adapter = create_papillon_adapter(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Reflection LM (also uses local model) ──
    def reflection_lm(prompt: str) -> str:
        return local_completion(
            model_name=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=args.max_new_tokens,
        )

    # ── Callbacks ──
    from gepa.callbacks.detailed_logger import DetailedValidationLogger

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

    # ── Create BGE-M3 Reward Model (like hover_gepa_main.py) ──
    from gepa.strategies.bge_surrogate_reward_model import BGESurrogateRewardModel
    reward_model = BGESurrogateRewardModel(min_samples=2)

    # ── Print config ──
    print(f"\n{'='*60}")
    print("GEPA on PUPA — Local HF Model + BGE-M3 Surrogate + Metrics")
    print(f"{'='*60}")
    print(f"Local Model:         {args.model_name}")
    print(f"Max New Tokens:      {args.max_new_tokens}")
    print(f"Max metric calls:    {args.max_metric_calls}")
    print(f"Val Subset Size:     {args.val_subset_size if args.val_subset_size else 'Full'}")
    print(f"Reward Model:        BGE-M3 Surrogate (Matryoshka)")
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

    # ── Run GEPA ──
    print("Starting GEPA optimization (Local HF + BGE-M3 Surrogate)...\n")
    result = optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        use_multi_minibatch=True,
        num_minibatches=3,
        candidates_per_minibatch=10,
        use_bandit_mutation=True,
        top_k=3,
        reward_model=reward_model,
        callbacks=callbacks,
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

    # ── Final Statistics (comprehensive, like hover_gepa_main.py) ──
    print(f"{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total Rounds (iterations):    {len(result.val_aggregate_scores)}")
    print(f"Total LLM Calls:              {llm_call_counter.count}")
    print(f"Seed test score:              {seed_avg:.4f}")
    print(f"Optimized test score:         {best_avg:.4f}")
    print(f"Best validation score:        {best_score:.4f}")
    print(f"Local model:                  {args.model_name}")
    print(f"Reward model:                 BAAI/bge-m3 (Matryoshka)")
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
