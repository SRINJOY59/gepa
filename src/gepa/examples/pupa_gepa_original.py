#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import datetime
import logging
import os
import sys

class _TeeStream:
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


def robust_completion(model: str, messages: list[dict], openrouter_keys: list[str], **kwargs) -> Any:
    """Try OpenRouter keys with sleep to avoid rate limits."""
    import litellm
    import time
    import logging

    # Cap max_tokens to avoid reserving ~4K tokens against TPM limit
    kwargs.setdefault("max_tokens", 1024)

    if not openrouter_keys:
        return litellm.completion(model=model, messages=messages, **kwargs)

    last_err = None
    keys = list(openrouter_keys) # copy to avoid modifying original list in place
    
    while keys:
        key = keys[0]
        try:
            resp = litellm.completion(model=model, messages=messages, api_key=key, **kwargs)
            time.sleep(2) # Small sleep after success to stay under rate limits
            return resp
        except Exception as e:
            last_err = e
            logging.warning(f"API call failed with current key: {e}")
            logging.warning("Removing failed key from rotation and retrying with the next one...")
            keys.pop(0)
            time.sleep(3)

    logging.error("All OpenRouter API keys failed.")
    raise last_err


def create_papillon_adapter(task_lm: str, untrusted_lm: str, api_keys: list[str]):
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    @dataclass
    class PapillonTrajectory:
        input_data: dict
        output_data: dict
        trace_info: dict

    class PAPILLONAdapter(GEPAAdapter):
        def __init__(self, task_lm: str, untrusted_lm: str, api_keys: list[str]):
            self.task_lm = task_lm
            self.untrusted_lm = untrusted_lm
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

            redaction_prompt = candidate.get("redaction_prompt", "")
            response_prompt = candidate.get("response_prompt", "")

            for example in inputs:
                try:
                    user_query = example["input"]

                    redact_resp = robust_completion(
                        model=self.task_lm,
                        messages=[
                            {"role": "system", "content": redaction_prompt},
                            {"role": "user", "content": f"User query: {user_query}"},
                        ],
                        openrouter_keys=self.api_keys,
                    )
                    redacted_request = redact_resp.choices[0].message.content

                    untrusted_resp = robust_completion(
                        model=self.untrusted_lm,
                        messages=[{"role": "user", "content": redacted_request}],
                        openrouter_keys=self.api_keys,
                    )
                    external_response = untrusted_resp.choices[0].message.content

                    final_resp = robust_completion(
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
                        openrouter_keys=self.api_keys,
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
                resp = robust_completion(
                    model=self.task_lm,
                    messages=[{"role": "user", "content": prompt}],
                    openrouter_keys=self.api_keys,
                )
                return "true" in resp.choices[0].message.content.lower()
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
                resp = robust_completion(
                    model=self.task_lm,
                    messages=[{"role": "user", "content": prompt}],
                    openrouter_keys=self.api_keys,
                )
                leaked = int(resp.choices[0].message.content.strip())
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

    return PAPILLONAdapter(task_lm, untrusted_lm, api_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Original GEPA on PUPA"
    )
    # LLM 
    parser.add_argument("--task_lm", type=str, default="openrouter/meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--untrusted_lm", type=str, default="openrouter/meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--reflection_lm", type=str, default="openrouter/meta-llama/llama-3.1-8b-instruct")

    # Dataset
    parser.add_argument("--train_size", type=int, default=20)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Budget
    parser.add_argument("--max_metric_calls", type=int, default=500)

    # Output
    parser.add_argument("--output_file", type=str,
                        default="results_pupa_original.txt",
                        help="Path to the output .txt file for all results")
    
    # Validation Subset & Logging
    parser.add_argument("--val_subset_size", type=int, default=None,
                        help="Number of validation examples to use (random subset). If None, use full valset.")
    parser.add_argument("--detailed_log_file", type=str, default="validation_scores_pupa.jsonl",
                        help="Path to log detailed validation scores (JSONL).")

    args = parser.parse_args()

    log_file = open(args.output_file, "w", encoding="utf-8")
    log_file.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
    log_file.write(f"{'='*60}\n\n")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    print(f"Logging all output to: {args.output_file}")

    # ── Collect OpenRouter API keys  ──
    openrouter_keys = []
    for k, v in os.environ.items():
        if k.startswith("OPENROUTER_API_KEY") and v.strip():
            openrouter_keys.append((k, v.strip()))
    openrouter_keys.sort(key=lambda x: x[0])
    openrouter_keys_list = [v for _, v in openrouter_keys]
    
    if not openrouter_keys_list:
        print("Warning: No defined OPENROUTER_API_KEYs found. Make sure to set OPENROUTER_API_KEY env variables.")
    else:
        print(f"Loaded {len(openrouter_keys_list)} OpenRouter API keys.")

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

    # ── Create adapter ──
    adapter = create_papillon_adapter(
        task_lm=args.task_lm,
        untrusted_lm=args.untrusted_lm,
        api_keys=openrouter_keys_list,
    )

    # ── Reflection LM ──
    def reflection_lm(prompt: str) -> str:
        resp = robust_completion(
            model=args.reflection_lm,
            messages=[{"role": "user", "content": prompt}],
            openrouter_keys=openrouter_keys_list,
        )
        return resp.choices[0].message.content

    # ── Callbacks ──
    from gepa.callbacks.detailed_logger import DetailedValidationLogger
    
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

    # ── Create BERT Reward Model ──
    from gepa.strategies.bert_reward_model import BERTRewardModel
    reward_model = BERTRewardModel(min_samples=2)

    # ── Print config ──
    print(f"\n{'='*60}")
    print("Original GEPA on PUPA (PAPILLON)")
    print(f"{'='*60}")
    print(f"Task LM:             {args.task_lm}")
    print(f"Untrusted LM:        {args.untrusted_lm}")
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

    # ── Run GEPA - Original ──
    print("Starting Original GEPA optimization...\n")
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

    print(f"\nRun completed: {datetime.datetime.now().isoformat()}")
    print(f"Full output saved to: {args.output_file}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return result


if __name__ == "__main__":
    main()
