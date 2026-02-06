#!/usr/bin/env python3
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Bandit GEPA Example: Privacy-Preserving Delegation with Gemini

This example demonstrates how to use GEPA with:
1. Thompson Sampling bandit for mutation strategy selection
2. K=10 prompt variant generation
3. Gemini as the task and reflection LLM

Based on the PAPILLON (Privacy-aware AI Language Learning and Instruction Optimization Network)
concept from DSPy tutorials.

Usage:
    python bandit_papillon_gemini.py --google_api_key YOUR_GEMINI_API_KEY

Environment:
    export GOOGLE_API_KEY=your_gemini_api_key
"""

import argparse
import os
import random
from typing import Any

import litellm

# Enable verbose logging for litellm to see all API calls
litellm.verbose = False

import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from gepa import optimize


def load_pupa_dataset(train_size: int = 50, val_size: int = 50, test_size: int = 50, seed: int = 42):
    """
    Load the PUPA (Privacy User Preference Alignment) dataset.
    
    This dataset contains user queries with privacy-sensitive information (PII)
    and target responses that preserve privacy.
    """
    from datasets import load_dataset
    
    # Load PUPA dataset
    # Load PUPA dataset directly (fail if not found)
    pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")
    examples = [
        {
            "input": x["user_query"],
            "target_response": x["target_response"],
            "pii_units": x["pii_units"],
        }
        for x in pupa_new["train"]
    ]
    
    # Shuffle and split
    random.Random(seed).shuffle(examples)
    
    trainset = examples[:train_size]
    valset = examples[train_size:train_size + val_size]
    testset = examples[train_size + val_size:train_size + val_size + test_size]
    
    print(f"Loaded {len(trainset)} train, {len(valset)} val, {len(testset)} test examples")
    return trainset, valset, testset





def create_evaluator(task_lm_fn, untrusted_lm_fn):
    """
    Create an evaluator that measures:
    1. Quality: How well the response meets the user's needs
    2. Privacy: How much PII is leaked to the untrusted model
    
    Returns a combined score and feedback for GEPA.
    """
    
    def evaluate_privacy_and_quality(example: dict, output: Any) -> tuple[float, dict]:
        """Evaluate a single example."""
        user_query = example["input"]
        pii_units = example.get("pii_units", "")
        
        # Parse the output (expecting redacted_request and final_response)
        if isinstance(output, dict):
            redacted_request = output.get("redacted_request", "")
            final_response = output.get("final_response", "")
        else:
            redacted_request = str(output)
            final_response = str(output)
        
        # Calculate privacy score (1.0 = no PII leaked, 0.0 = all PII leaked)
        if pii_units:
            pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
            leaked = sum(1 for p in pii_list if p.lower() in redacted_request.lower())
            privacy_score = 1.0 - (leaked / len(pii_list)) if pii_list else 1.0
        else:
            privacy_score = 1.0
        
        # Simple quality check (response length and relevance)
        quality_score = min(1.0, len(final_response) / 100) if final_response else 0.0
        
        # Combined score
        overall_score = (privacy_score + quality_score) / 2.0
        
        feedback = {
            "privacy_score": privacy_score,
            "quality_score": quality_score,
            "overall_score": overall_score,
            "feedback_text": (
                f"Privacy: {privacy_score:.2f} (lower means more PII leaked), "
                f"Quality: {quality_score:.2f}. "
                f"{'Reduce PII leakage!' if privacy_score < 0.8 else 'Good privacy!'} "
                f"{'Improve response quality!' if quality_score < 0.5 else 'Good quality!'}"
            ),
        }
        
        return overall_score, feedback
    
    return evaluate_privacy_and_quality


def create_papillon_adapter(task_lm: str, untrusted_lm: str, api_key: str):
    """
    Create a GEPA adapter for the PAPILLON privacy-preserving system.
    
    The system has two prompts:
    1. redaction_prompt: Instructs the local LM to craft a privacy-preserving request
    2. response_prompt: Instructs the local LM to respond using the external LM's output
    """
    from dataclasses import dataclass
    from typing import TypeVar
    
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter
    
    # Define trajectory type for this adapter
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
        
        def evaluate(
            self,
            inputs: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch:
            """Evaluate the PAPILLON system on a batch of inputs."""
            outputs = []
            scores = []
            trajectories = [] if capture_traces else None
            
            redaction_prompt = candidate.get("redaction_prompt", "")
            response_prompt = candidate.get("response_prompt", "")
            
            for example in inputs:
                try:
                    # Step 1: Craft privacy-preserving request using local LM
                    user_query = example["input"]
                    redact_messages = [
                        {"role": "system", "content": redaction_prompt},
                        {"role": "user", "content": f"User query: {user_query}"},
                    ]
                    
                    redact_response = litellm.completion(
                        model=self.task_lm,
                        messages=redact_messages,
                        api_key=self.api_key,
                    )
                    redacted_request = redact_response.choices[0].message.content
                    

                    
                    # Step 2: Send to untrusted LM
                    untrusted_messages = [
                        {"role": "user", "content": redacted_request},
                    ]
                    untrusted_response = litellm.completion(
                        model=self.untrusted_lm,
                        messages=untrusted_messages,
                        api_key=self.api_key,
                    )
                    external_response = untrusted_response.choices[0].message.content
                    
                    # Step 3: Generate final response using local LM
                    response_messages = [
                        {"role": "system", "content": response_prompt},
                        {"role": "user", "content": (
                            f"Original user query: {user_query}\n\n"
                            f"External LM request: {redacted_request}\n\n"
                            f"External LM response: {external_response}\n\n"
                            f"Generate the final response for the user."
                        )},
                    ]
                    final_response = litellm.completion(
                        model=self.task_lm,
                        messages=response_messages,
                        api_key=self.api_key,
                    )
                    final_text = final_response.choices[0].message.content
                    

                    
                    output = {
                        "redacted_request": redacted_request,
                        "external_response": external_response,
                        "final_response": final_text,
                    }
                    
                    # Score
                    score, feedback = self._evaluate_single(example, output)
                    
                    outputs.append(output)
                    scores.append(score)
                    
                    if capture_traces:
                        trajectories.append(PapillonTrajectory(
                            input_data=example,
                            output_data=output,
                            trace_info={
                                "redaction_prompt": redaction_prompt,
                                "response_prompt": response_prompt,
                                "feedback": feedback,
                            },
                        ))
                
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
        
        def _llm_judge_quality(self, user_query: str, target_response: str, pred_response: str) -> bool:
            """LLM Judge: Is response A (pred) at least as good as response B (target)?"""
            prompt = f"""You are comparing the quality of two responses, given a user query.
Is response A at least as good as response B?

User Query: {user_query}

Response A (Prediction): {pred_response}

Response B (Target): {target_response}

Return ONLY 'true' or 'false'.
"""
            try:
                # Use task_lm as judge for simplicity in this example
                response = litellm.completion(
                    model=self.task_lm,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.api_key,
                )
                text = response.choices[0].message.content.lower().strip()
                return "true" in text
            except Exception:
                return False

        def _evaluate_single(self, example: dict, output: dict) -> tuple[float, dict]:
            """Evaluate privacy and quality for a single example."""
            pii_units = example.get("pii_units", "")
            redacted_request = output.get("redacted_request", "")
            final_response = output.get("final_response", "")
            
            # Privacy score
            if pii_units:
                pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
                leaked = sum(1 for p in pii_list if p.lower() in redacted_request.lower())
                privacy_score = 1.0 - (leaked / len(pii_list)) if pii_list else 1.0
            else:
                privacy_score = 1.0
            
            # Quality score (LLM Judge)
            target_response = example.get("target_response", "")
            is_good = self._llm_judge_quality(example["input"], target_response, final_response)
            quality_score = 1.0 if is_good else 0.0
            
            overall = (privacy_score + quality_score) / 2.0
            
            return overall, {
                "privacy": privacy_score,
                "quality": quality_score,
                "feedback": f"Privacy={privacy_score:.2f}, Quality={quality_score:.2f} ({'Good' if is_good else 'Bad'})",
            }
        
        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_result: EvaluationBatch,
            components_to_update: list[str],
        ) -> dict[str, list[dict]]:
            """Create reflective dataset from evaluation for GEPA to learn from."""
            reflective_data = {}
            
            for component in components_to_update:
                component_data = []
                
                # Use trajectories if available, otherwise skip (this shouldn't happen if captured)
                trajs = eval_result.trajectories if eval_result.trajectories else [None] * len(eval_result.outputs)
                
                for i, (out, score, traj) in enumerate(zip(
                    eval_result.outputs,
                    eval_result.scores,
                    trajs,
                )):
                    if traj is None:
                        continue
                    
                    entry = {
                        "input": traj.input_data.get("input", ""),
                        "current_instruction": candidate.get(component, ""),
                        "output": out,
                        "score": score,
                        "feedback": traj.trace_info.get("feedback", {}),
                    }
                    component_data.append(entry)
                
                reflective_data[component] = component_data
            
            return reflective_data
    
    return PAPILLONAdapter(task_lm, untrusted_lm, api_key)


def main():
    parser = argparse.ArgumentParser(description="Bandit GEPA Example with Gemini")
    parser.add_argument("--google_api_key", type=str, default=os.environ.get("GOOGLE_API_KEY", ""))
    parser.add_argument("--task_lm", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--untrusted_lm", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--reflection_lm", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--train_size", type=int, default=50)
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=20)
    parser.add_argument("--max_metric_calls", type=int, default=1000)
    parser.add_argument("--use_bandit", action="store_true", default=True, help="Use bandit-based mutation")
    parser.add_argument("--num_variants", type=int, default=10, help="Number of prompt variants (K)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if not args.google_api_key:
        raise ValueError("Please provide --google_api_key or set GOOGLE_API_KEY environment variable")
    
    # Set API key for litellm
    os.environ["GEMINI_API_KEY"] = args.google_api_key
    
    # Load dataset
    trainset, valset, testset = load_pupa_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    
    # Define seed candidate with initial prompts matching DSPy Tutorial
    # Tutorial uses ChainOfThought for Redaction, so we add reasoning steps.
    seed_candidate = {
        "redaction_prompt": """Given a private user query, create a privacy-preserving request for a powerful external LLM.
The LLM may assist without learning private information about the user.

Let's think step by step to identify PII and redact it effectively.""",
        
        "response_prompt": """Respond to a user query. For inspiration, we found a potentially related request to a powerful external LLM and its response.

Input:
1. related_llm_request: The privacy-preserving request sent to the external LLM.
2. related_llm_response: Information from a powerful LLM responding to the related request.
3. user_query: The user's original request you need to fulfill.

Instruction: Generate your final response to the user's request.""",
    }
    
    # Create adapter
    adapter = create_papillon_adapter(
        task_lm=args.task_lm,
        untrusted_lm=args.untrusted_lm,
        api_key=args.google_api_key,
    )
    
    # Create reflection LM function
    def reflection_lm(prompt: str) -> str:
        response = litellm.completion(
            model=args.reflection_lm,
            messages=[{"role": "user", "content": prompt}],
            api_key=args.google_api_key,
        )
        return response.choices[0].message.content
    
    from gepa.logging.logger import StdOutLogger

    # ...
    
    # Create logger
    logger = StdOutLogger()
    
    print(f"\n{'='*60}")
    print("Bandit GEPA Optimization with Gemini")
    print(f"{'='*60}")
    print(f"Task LM: {args.task_lm}")
    print(f"Reflection LM: {args.reflection_lm}")
    print(f"Use Bandit Mutation: {args.use_bandit}")
    print(f"Number of Variants (K): {args.num_variants}")
    print(f"Max Metric Calls: {args.max_metric_calls}")
    print(f"{'='*60}\n")

    # Evaluate seed candidate on test set
    print(f"\n{'='*60}")
    print("Evaluating SEED candidate on Test Set (Score Check)")
    print(f"{'='*60}")
    seed_eval_batch = adapter.evaluate(testset, seed_candidate)
    seed_score = sum(seed_eval_batch.scores) / len(seed_eval_batch.scores) if seed_eval_batch.scores else 0.0
    print(f"Seed Average Score: {seed_score:.4f}")
    print(f"{'='*60}\n")
    
    # Run GEPA optimization with bandit
    print("Starting optimization...")
    result = optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm,
        # Bandit-based mutation settings
        use_bandit_mutation=args.use_bandit,
        num_prompt_variants=args.num_variants,
        # Other settings
        candidate_selection_strategy="pareto",
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
        logger=logger,
        display_progress_bar=True,
        track_best_outputs=True,
    )
    
    # Display results
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"Best Validation Score: {best_score:.4f}")
    print(f"Best Candidate (Index {result.best_idx}):")
    for k, v in result.best_candidate.items():
        print(f"\nComponent '{k}':\n{v}")
        
    # Evaluate optimized candidate on test set
    print(f"\n{'='*60}")
    print("Evaluating OPTIMIZED candidate on Test Set (Score Check)")
    print(f"{'='*60}")
    best_eval_batch = adapter.evaluate(testset, result.best_candidate)
    best_score_avg = sum(best_eval_batch.scores) / len(best_eval_batch.scores) if best_eval_batch.scores else 0.0
    print(f"Optimized Average Score: {best_score_avg:.4f}")
    print(f"{'='*60}\n")
    
    return result


if __name__ == "__main__":
    main()
