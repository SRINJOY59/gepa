#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Surrogate-Guided GEPA on the HoVer dataset.

Implements the 8-step pipeline:
  1. Select parent prompt
  2. Generate N mutations (20-50)
  3. Surrogate scores all (prompt, validation_question) pairs
  4. Select top-K candidates via UCB
  5. Run full validation on top-K only
  6. Collect training data for surrogate
  7. Update the surrogate model (SGD + MSE)
  8. Repeat
"""

from __future__ import annotations

# ── stdlib imports (BEFORE any sys.path manipulation) ──
import datetime
import logging
import os
import sys
import time


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
from dataclasses import dataclass
from typing import Any

import litellm

litellm.verbose = False
litellm.suppress_debug_info = True

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.strategies.surrogate_reward_model import SurrogateRewardModel
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# Load all OpenRouter API keys from .env  (OPENROUTER_API_KEY_1 … _6)
# ──────────────────────────────────────────────────────────────────────
def _load_openrouter_keys() -> list[str]:
    """Collect OPENROUTER_API_KEY_1 … OPENROUTER_API_KEY_6 from env."""
    keys: list[str] = []
    for i in range(1, 7):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "")
        if k:
            keys.append(k)
    if not keys:
        # Fallback: try the old single-key variable
        single = os.environ.get("OPENROUTER_API_KEY", "")
        if single:
            keys.append(single)
    return keys


# ──────────────────────────────────────────────────────────────────────
# LLM call with fallback retries — rotates through API keys
# ──────────────────────────────────────────────────────────────────────
def _call_with_fallback(
    model: str,
    messages: list[dict],
    api_keys: list[str],
    base_delay: float = 2.0,
) -> str:
    """Call litellm.completion, rotating through `api_keys` on failure.

    Each key is tried once (so total attempts == len(api_keys)).
    Uses exponential backoff between retries.  On all failures returns
    an empty string so the pipeline never crashes.
    """
    max_retries = len(api_keys)
    last_error: Exception | None = None
    for attempt, key in enumerate(api_keys, 1):
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                api_key=key,
                timeout=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            wait = base_delay * (2 ** (attempt - 1))  # exponential backoff
            print(
                f"  [Fallback] API key {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                print(f"  [Fallback] Switching to next key, retrying in {wait:.1f}s ...")
                time.sleep(wait)

    # All keys exhausted — return a safe fallback string
    print(
        f"  [Fallback] All {max_retries} API keys failed. "
        f"Last error: {last_error}. Using empty fallback response."
    )
    return ""


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
def load_hover_dataset(
    train_size: int = 200,
    val_size: int = 100,
    test_size: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the HoVer dataset (``hover-nlp/hover``)."""
    from datasets import load_dataset

    ds = load_dataset("hover-nlp/hover", trust_remote_code=True)

    examples = []
    for split_name in ("train", "validation"):
        if split_name not in ds:
            continue
        for x in ds[split_name]:
            label = x.get("label")
            if label is None or label == -1 or label == "-1":
                continue
            if isinstance(label, int):
                label_str = "SUPPORTED" if label == 0 else "NOT_SUPPORTED"
            else:
                label_str = str(label).strip().upper()
            examples.append({"claim": x["claim"], "label": label_str})

    random.Random(seed).shuffle(examples)

    total_needed = train_size + val_size + test_size
    if len(examples) < total_needed:
        print(
            f"Warning: Only {len(examples)} labelled examples available, "
            f"requested {total_needed}. Adjusting proportionally."
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
# HoVer Adapter
# ──────────────────────────────────────────────────────────────────────
def create_hover_adapter(task_lm: str, api_keys: list[str]):
    """Build a GEPAAdapter for the HoVer claim verification task."""

    _adapter_api_keys = api_keys

    @dataclass
    class HoVerTrajectory:
        input_data: dict
        output_data: dict
        trace_info: dict

    class HoVerAdapter(GEPAAdapter):
        def __init__(self, task_lm: str, api_keys: list[str]):
            self.task_lm = task_lm
            self.api_keys = api_keys

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

                    raw_response = _call_with_fallback(
                        model=self.task_lm,
                        messages=[
                            {"role": "system", "content": verification_prompt},
                            {"role": "user", "content": f"Claim: {claim}"},
                        ],
                        api_keys=self.api_keys,
                    )

                    pred_label = self._parse_label(raw_response)
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

        @staticmethod
        def _parse_label(raw: str) -> str:
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
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _generate_mutation(
    parent_prompt: str,
    reflective_dataset: list[dict],
    reflection_lm,
) -> str:
    """Use the GEPA reflective proposer to generate one mutated prompt."""
    new_texts = InstructionProposalSignature.run(
        lm=reflection_lm,
        input_dict={
            "current_instruction_doc": parent_prompt,
            "dataset_with_feedback": reflective_dataset,
            "prompt_template": None,  # use default
        },
    )
    return new_texts["new_instruction"]


def _evaluate_prompt_on_set(
    adapter,
    candidate: dict[str, str],
    dataset: list[dict],
) -> tuple[float, list[float]]:
    """Evaluate a candidate on a dataset and return (avg_score, per_example_scores)."""
    result = adapter.evaluate(dataset, candidate, capture_traces=False)
    avg = sum(result.scores) / max(len(result.scores), 1)
    return avg, result.scores


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Surrogate-Guided GEPA on HoVer"
    )
    # LLM (API keys are loaded from .env automatically)
    parser.add_argument("--task_lm", type=str, default="openrouter/qwen/qwen3-8b")
    parser.add_argument("--reflection_lm", type=str, default="openrouter/qwen/qwen3-8b")

    # Dataset
    parser.add_argument("--train_size", type=int, default=10)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Pipeline config
    parser.add_argument(
        "--max_iterations", type=int, default=15,
        help="Maximum number of iterations (safety cap)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping: stop after this many iterations with no improvement",
    )
    parser.add_argument(
        "--num_mutations", type=int, default=20,
        help="N: number of prompt mutations to generate per iteration (20-50)",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="K: number of top candidates to fully validate per iteration",
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=5,
        help="Size of train minibatch for generating reflective feedback",
    )
    parser.add_argument(
        "--exploration_weight", type=float, default=1.0,
        help="UCB exploration weight (c): score + c * sqrt(ln(N)/n_i)",
    )

    # Surrogate model
    parser.add_argument("--surrogate_hidden_dim", type=int, default=256)
    parser.add_argument("--surrogate_lr", type=float, default=1e-3)
    parser.add_argument("--surrogate_epochs", type=int, default=5)
    parser.add_argument("--surrogate_min_samples", type=int, default=3)

    # Output
    parser.add_argument(
        "--output_file", type=str,
        default="results_hover_surrogate_fixed.txt",
        help="Path to the output .txt file for all results",
    )

    args = parser.parse_args()

    # ── Set up output file (tee stdout + stderr) ──
    output_path = args.output_file
    try:
        log_file = open(output_path, "w", encoding="utf-8")
    except PermissionError:
        # File may be locked by a shell redirect — use a timestamped name
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_hover_surrogate_final_fixed.txt"
        print(f"WARNING: Cannot open '{args.output_file}', using '{output_path}' instead.")
        log_file = open(output_path, "w", encoding="utf-8")
    log_file.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
    log_file.write(f"{'='*60}\n\n")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    print(f"Logging all output to: {output_path}")

    # ── Load OpenRouter API keys ──
    openrouter_keys = _load_openrouter_keys()
    if not openrouter_keys:
        raise ValueError(
            "Set OPENROUTER_API_KEY_1 … OPENROUTER_API_KEY_6 in .env "
            "(or at least OPENROUTER_API_KEY)"
        )
    print(f"Loaded {len(openrouter_keys)} OpenRouter API key(s)")

    # ── Load dataset ──
    trainset, valset, testset = load_hover_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # ── Seed candidate ──
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
        api_keys=openrouter_keys,
    )

    # ── Reflection LM callable (rotates through all keys) ──
    def reflection_lm(prompt: str) -> str:
        return _call_with_fallback(
            model=args.reflection_lm,
            messages=[{"role": "user", "content": prompt}],
            api_keys=openrouter_keys,
        )

    # ── Surrogate model ──
    surrogate = SurrogateRewardModel(
        hidden_dim=args.surrogate_hidden_dim,
        min_samples=args.surrogate_min_samples,
    )

    # Validation questions (claims) — used as the "question" input to the
    # surrogate.  The surrogate scores (prompt, question) pairs.
    val_questions = [ex["claim"] for ex in valset]

    # ── RNG ──
    rng = random.Random(args.seed)

    # ── Print config ──
    print(f"\n{'='*60}")
    print("Surrogate-Guided GEPA on HoVer")
    print(f"{'='*60}")
    print(f"Task LM:             {args.task_lm}")
    print(f"Reflection LM:       {args.reflection_lm}")
    print(f"Max iterations:      {args.max_iterations}")
    print(f"Early stop patience: {args.patience}")
    print(f"Mutations per iter:  {args.num_mutations}")
    print(f"Top-K:               {args.top_k}")
    print(f"Minibatch size:      {args.minibatch_size}")
    print(f"UCB weight:          {args.exploration_weight}")
    print(f"Surrogate hidden:    {args.surrogate_hidden_dim}")
    print(f"Surrogate LR (SGD):  {args.surrogate_lr}")
    print(f"Surrogate epochs:    {args.surrogate_epochs}")
    print(f"Surrogate min samp:  {args.surrogate_min_samples}")
    print(f"{'='*60}\n")

    # ── Evaluate seed on test set ──
    print(f"{'='*60}")
    print("Evaluating SEED candidate on Test Set")
    print(f"{'='*60}")
    seed_avg, _ = _evaluate_prompt_on_set(adapter, seed_candidate, testset)
    print(f"Seed Average Score (test): {seed_avg:.4f}")
    print(f"{'='*60}\n")

    # ──────────────────────────────────────────────────────────────────
    # Candidate pool
    # ──────────────────────────────────────────────────────────────────
    candidate_pool: list[dict[str, str]] = [seed_candidate.copy()]
    candidate_scores: list[float] = [0.0]  # will be filled after first eval

    # Evaluate seed on valset
    seed_val_avg, seed_val_scores = _evaluate_prompt_on_set(
        adapter, seed_candidate, valset
    )
    candidate_scores[0] = seed_val_avg
    print(f"Seed validation score: {seed_val_avg:.4f}\n")

    # Collect initial surrogate training data from seed
    prompt_text_seed = seed_candidate["verification_prompt"]
    surrogate.add_training_data(
        prompts=[prompt_text_seed] * len(valset),
        questions=val_questions,
        scores=seed_val_scores,
    )
    # Initial train of surrogate
    loss = surrogate.train_on_buffer(
        lr=args.surrogate_lr,
        epochs=args.surrogate_epochs,
    )
    if loss is not None:
        print(f"Initial surrogate training loss: {loss:.6f}\n")

    total_llm_evals = len(testset) + len(valset)  # seed evals

    # ──────────────────────────────────────────────────────────────────
    # Main loop (with early stopping)
    # ──────────────────────────────────────────────────────────────────
    iteration = 0
    no_improve_count = 0
    global_best_score = candidate_scores[0]  # seed val score

    while True:
        iteration += 1

        # Safety cap
        if iteration > args.max_iterations:
            print(f"\n  Reached max iterations ({args.max_iterations}). Stopping.")
            break

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} (no-improve streak: {no_improve_count}/{args.patience})")
        print(f"{'='*60}")

        # ── STEP 1: Select parent prompt ──
        best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
        parent = candidate_pool[best_idx]
        parent_prompt = parent["verification_prompt"]
        print(f"[Step 1] Selected parent #{best_idx} (val score: {candidate_scores[best_idx]:.4f})")
        print(f"{'─'*40}")
        print(f"PARENT PROMPT:\n{parent_prompt}")
        print(f"{'─'*40}")

        # ── STEP 2: Generate N mutations ──
        print(f"[Step 2] Generating {args.num_mutations} mutations ...")

        # Sample a minibatch from trainset for reflective feedback
        minibatch = rng.sample(trainset, min(args.minibatch_size, len(trainset)))

        # Evaluate parent on minibatch (with traces) for feedback
        parent_eval = adapter.evaluate(minibatch, parent, capture_traces=True)
        total_llm_evals += len(minibatch)

        # Build reflective dataset
        reflective_data = adapter.make_reflective_dataset(
            parent, parent_eval, ["verification_prompt"]
        )
        ref_entries = reflective_data.get("verification_prompt", [])

        mutations: list[str] = []
        for m_idx in range(args.num_mutations):
            try:
                mutated = _generate_mutation(
                    parent_prompt, ref_entries, reflection_lm
                )
                mutations.append(mutated)
                print(f"  Mutation {m_idx+1}/{args.num_mutations} generated ✓")
            except Exception as e:
                print(f"  Mutation {m_idx+1} failed: {e}")
                mutations.append(parent_prompt)  # fallback to parent

        # Deduplicate (keep order)
        seen = set()
        unique_mutations: list[str] = []
        for m in mutations:
            if m not in seen:
                seen.add(m)
                unique_mutations.append(m)
        mutations = unique_mutations
        print(f"\n  Generated {len(mutations)} unique mutations")
        print(f"{'─'*40}")
        for m_idx, m_text in enumerate(mutations):
            print(f"  [Mutation {m_idx+1}]")
            print(f"  {m_text}")
            print(f"  {'·'*35}")
        print(f"{'─'*40}")

        # ── STEP 3: Surrogate scores all mutations ──
        print(f"[Step 3] Surrogate scoring {len(mutations)} mutations "
              f"(trained={surrogate.is_trained}) ...")
        surrogate_scores = surrogate.predict_prompt_scores(
            prompt_texts=mutations,
            validation_questions=val_questions,
            use_ucb=True,
            exploration_weight=args.exploration_weight,
        )

        print(f"{'─'*40}")
        print(f"  {'Rank':<6} {'UCB Score':<12} {'Prompt (first 120 chars)'}")
        print(f"  {'─'*6} {'─'*12} {'─'*50}")
        # Display sorted by surrogate score
        scored_list = list(enumerate(zip(mutations, surrogate_scores)))
        scored_list.sort(key=lambda x: x[1][1], reverse=True)
        for rank, (m_idx, (m_text, s_score)) in enumerate(scored_list, 1):
            marker = " ★" if rank <= args.top_k else ""
            print(f"  {rank:<6} {s_score:<12.4f} {m_text[:120]}...{marker}")
        print(f"{'─'*40}")

        # ── STEP 4: Select top-K via UCB ──
        ranked = sorted(
            zip(mutations, surrogate_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_k_mutations = [text for text, _ in ranked[: args.top_k]]
        print(f"[Step 4] Selected top-{args.top_k} candidates by UCB score")
        print(f"{'─'*40}")
        for tk_idx, prompt_text in enumerate(top_k_mutations):
            tk_score = ranked[tk_idx][1]
            print(f"  ★ Top-{tk_idx+1} (UCB={tk_score:.4f}):")
            print(f"  {prompt_text}")
            print(f"  {'·'*35}")
        print(f"{'─'*40}")

        # ── STEP 5: Run full validation on top-K ──
        print(f"[Step 5] Full validation on {len(top_k_mutations)} candidates ...")
        iteration_training_prompts: list[str] = []
        iteration_training_questions: list[str] = []
        iteration_training_scores: list[float] = []

        best_iter_score = -1.0
        best_iter_candidate: dict[str, str] | None = None

        for k_idx, prompt_text in enumerate(top_k_mutations):
            candidate = {"verification_prompt": prompt_text}
            val_avg, val_scores = _evaluate_prompt_on_set(adapter, candidate, valset)
            total_llm_evals += len(valset)

            print(f"  Candidate {k_idx+1}/{len(top_k_mutations)}: val_avg={val_avg:.4f}")
            print(f"  {'─'*35}")
            print(f"  PROMPT:\n  {prompt_text}")
            print(f"  {'─'*35}")

            # ── STEP 6: Collect training data ──
            iteration_training_prompts.extend([prompt_text] * len(valset))
            iteration_training_questions.extend(val_questions)
            iteration_training_scores.extend(val_scores)

            # Record visit for UCB
            surrogate.record_visit(prompt_text)

            # Track best
            if val_avg > best_iter_score:
                best_iter_score = val_avg
                best_iter_candidate = candidate

            # Add to candidate pool
            candidate_pool.append(candidate)
            candidate_scores.append(val_avg)

        # ── STEP 7: Update surrogate model ──
        print(f"[Step 7] Updating surrogate model ...")
        surrogate.add_training_data(
            prompts=iteration_training_prompts,
            questions=iteration_training_questions,
            scores=iteration_training_scores,
        )
        train_loss = surrogate.train_on_buffer(
            lr=args.surrogate_lr,
            epochs=args.surrogate_epochs,
        )
        buffer_size = len(surrogate._training_prompts)
        print(
            f"  Surrogate buffer: {buffer_size} samples | "
            f"loss: {train_loss:.6f}" if train_loss is not None
            else f"  Surrogate buffer: {buffer_size} samples | not enough data to train"
        )

        # ── STEP 8: Early stopping check ──
        global_best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
        current_global_best = candidate_scores[global_best_idx]

        if current_global_best > global_best_score + 1e-6:
            # Improvement found
            global_best_score = current_global_best
            no_improve_count = 0
            print(
                f"\n  ✓ Iteration {iteration} best val: {best_iter_score:.4f} | "
                f"NEW global best: {current_global_best:.4f} (#{global_best_idx}) | "
                f"Total LLM evals: {total_llm_evals}"
            )
        else:
            # No improvement
            no_improve_count += 1
            print(
                f"\n  ✗ Iteration {iteration} best val: {best_iter_score:.4f} | "
                f"Global best unchanged: {current_global_best:.4f} (#{global_best_idx}) | "
                f"No-improve: {no_improve_count}/{args.patience} | "
                f"Total LLM evals: {total_llm_evals}"
            )

        if no_improve_count >= args.patience:
            print(f"\n  Early stopping triggered after {iteration} iterations "
                  f"(no improvement for {args.patience} consecutive rounds).")
            break

    # ──────────────────────────────────────────────────────────────────
    # Final Results
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")

    global_best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
    best_candidate = candidate_pool[global_best_idx]
    best_val_score = candidate_scores[global_best_idx]

    print(f"Best Validation Score: {best_val_score:.4f}")
    print(f"Best Candidate (Index {global_best_idx}):")
    for k, v in best_candidate.items():
        print(f"\nComponent '{k}':\n{v}")

    # ── Evaluate optimised on test set ──
    print(f"\n{'='*60}")
    print("Evaluating OPTIMIZED candidate on Test Set")
    print(f"{'='*60}")
    best_test_avg, _ = _evaluate_prompt_on_set(adapter, best_candidate, testset)
    total_llm_evals += len(testset)
    print(f"Optimized Average Score (test): {best_test_avg:.4f}")
    print(f"Improvement over seed:          {best_test_avg - seed_avg:+.4f}")
    print(f"{'='*60}\n")

    # ── Surrogate model stats ──
    print(f"Surrogate Model: trained on {len(surrogate._training_prompts)} samples")
    print(f"Total LLM evaluations: {total_llm_evals}")

    # ── Close log file ──
    print(f"\nRun completed: {datetime.datetime.now().isoformat()}")
    print(f"Full output saved to: {args.output_file}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
