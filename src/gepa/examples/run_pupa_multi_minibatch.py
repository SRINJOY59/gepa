#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from __future__ import annotations

# ── stdlib imports (BEFORE any sys.path manipulation) ──
# Must import logging here to avoid local gepa/logging/ shadowing.
import logging
import os
import sys

# Ensure the gepa package is importable regardless of CWD.
# __file__ → examples/run_pupa_multi_minibatch.py
# We need src/ on sys.path (parent of the gepa package dir).
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
def load_pupa_dataset(
    train_size: int = 225,
    val_size: int = 225,
    test_size: int = 214,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the Columbia-NLP/PUPA dataset (``pupa_new`` split).

    Each example is a dict with keys:
      - ``input``          (str): the user query
      - ``target_response`` (str): gold reference response
      - ``pii_units``       (str): pipe-separated PII items

    Follows the split used in the DSPy GEPA PAPILLON tutorial.
    """
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
# PAPILLON Adapter (GEPA framework native)
# ──────────────────────────────────────────────────────────────────────
def create_papillon_adapter(task_lm: str, untrusted_lm: str, api_key: str):
    """Build a GEPAAdapter for the PAPILLON privacy-preserving pipeline.

    Two optimisable prompt components:
      ``redaction_prompt``  – instructs the local LM to strip PII
      ``response_prompt``   – instructs the local LM to answer the user
    """
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    @dataclass
    class PapillonTrajectory:
        input_data: dict
        output_data: dict
        trace_info: dict

    class PAPILLONAdapter(GEPAAdapter):
        def __init__(self, task_lm: str, untrusted_lm: str, api_key: str):
            self.task_lm = task_lm
            self.untrusted_lm = untrusted_lm
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

            redaction_prompt = candidate.get("redaction_prompt", "")
            response_prompt = candidate.get("response_prompt", "")

            for example in inputs:
                try:
                    user_query = example["input"]

                    # Step 1: Craft privacy-preserving request (local LM)
                    redact_resp = litellm.completion(
                        model=self.task_lm,
                        messages=[
                            {"role": "system", "content": redaction_prompt},
                            {"role": "user", "content": f"User query: {user_query}"},
                        ],
                        api_key=self.api_key,
                    )
                    redacted_request = redact_resp.choices[0].message.content

                    # Step 2: Send to untrusted LM
                    untrusted_resp = litellm.completion(
                        model=self.untrusted_lm,
                        messages=[{"role": "user", "content": redacted_request}],
                        api_key=self.api_key,
                    )
                    external_response = untrusted_resp.choices[0].message.content

                    # Step 3: Generate final response (local LM)
                    final_resp = litellm.completion(
                        model=self.task_lm,
                        messages=[
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
                        ],
                        api_key=self.api_key,
                    )
                    final_text = final_resp.choices[0].message.content

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

        # ── LLM judge (quality) ─────────────────────────────────────
        def _llm_judge_quality(
            self,
            user_query: str,
            target_response: str,
            pred_response: str,
        ) -> bool:
            """Return True if pred is at least as good as target."""
            prompt = (
                "You are comparing the quality of two responses.\n"
                "Is response A at least as good as response B?\n\n"
                f"User Query: {user_query}\n\n"
                f"Response A (Prediction): {pred_response}\n\n"
                f"Response B (Target): {target_response}\n\n"
                "Return ONLY 'true' or 'false'."
            )
            try:
                resp = litellm.completion(
                    model=self.task_lm,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.api_key,
                )
                return "true" in resp.choices[0].message.content.lower()
            except Exception:
                return False

        # ── PII leakage counting (LLM judge) ───────────────────────
        def _llm_judge_leakage(
            self,
            pii_units: str,
            redacted_request: str,
        ) -> float:
            """Return fraction of PII units leaked into the redacted request.

            Uses an LLM judge following the DSPy tutorial's JudgeLeakage
            pattern: count PII items that are *actual PII* **and** appear
            in the prompt.  Returns 0.0 (no leak) to 1.0 (all leaked).
            """
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
                resp = litellm.completion(
                    model=self.task_lm,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.api_key,
                )
                leaked = int(resp.choices[0].message.content.strip())
            except Exception:
                # Fallback: simple substring matching
                leaked = sum(
                    1
                    for p in pii_list
                    if p.lower() in redacted_request.lower()
                )

            return leaked / len(pii_list)

        # ── single-example scoring ──────────────────────────────────
        def _evaluate_single(
            self, example: dict, output: dict
        ) -> tuple[float, dict]:
            pii_units = example.get("pii_units", "")
            redacted_request = output.get("redacted_request", "")
            final_response = output.get("final_response", "")
            target_response = example.get("target_response", "")

            # Privacy score (LLM judge leakage)
            leakage = self._llm_judge_leakage(pii_units, redacted_request)
            privacy_score = 1.0 - leakage

            # Quality score (LLM judge)
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
                            "input": traj.input_data.get("input", ""),
                            "current_instruction": candidate.get(component, ""),
                            "output": out,
                            "score": score,
                            "feedback": traj.trace_info.get("feedback", {}),
                        }
                    )
                reflective_data[component] = entries

            return reflective_data

    return PAPILLONAdapter(task_lm, untrusted_lm, api_key)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Minibatch GEPA + BERT Reward Model on PUPA"
    )
    # LLM
    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY", ""),
    )
    parser.add_argument("--task_lm", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--untrusted_lm", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--reflection_lm", type=str, default="gemini/gemini-2.0-flash")

    # Dataset
    parser.add_argument("--train_size", type=int, default=225)
    parser.add_argument("--val_size", type=int, default=225)
    parser.add_argument("--test_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Multi-minibatch config
    parser.add_argument("--num_minibatches", type=int, default=15,
                        help="Number of minibatches per iteration (M)")
    parser.add_argument("--candidates_per_minibatch", type=int, default=20,
                        help="Candidates generated per minibatch (N) for surrogate scoring")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-K candidates selected by surrogate for actual evaluation")

    # Budget
    parser.add_argument("--max_metric_calls", type=int, default=500)

    # Lightweight Reward Model (TF-IDF + MLP)
    parser.add_argument("--bert_min_samples", type=int, default=1,
                        help="Minimum training samples before reward model starts predicting (default=1 for online)")

    args = parser.parse_args()

    if not args.google_api_key:
        raise ValueError(
            "Provide --google_api_key or set GOOGLE_API_KEY env variable"
        )


    # ── Load dataset ──
    trainset, valset, testset = load_pupa_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

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

    # ── Create adapter ──
    adapter = create_papillon_adapter(
        task_lm=args.task_lm,
        untrusted_lm=args.untrusted_lm,
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

    # ── Reward Model ──
    # Note: Uses lightweight TF-IDF + MLP despite the class name
    reward_model = BERTRewardModel(min_samples=args.bert_min_samples)

    # ── Logger ──
    from gepa.logging.logger import StdOutLogger

    logger = StdOutLogger()

    # ── Print config ──
    print(f"\n{'='*60}")
    print("Multi-Minibatch GEPA + TF-IDF Reward Model on PUPA")
    print(f"{'='*60}")
    print(f"Task LM:             {args.task_lm}")
    print(f"Untrusted LM:        {args.untrusted_lm}")
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
    seed_eval = adapter.evaluate(testset[:10], seed_candidate)
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

    return result


if __name__ == "__main__":
    main()
