import json
import os
import argparse
from datasets import load_dataset
from dotenv import load_dotenv
import litellm
import time
from concurrent.futures import ThreadPoolExecutor

# Load API keys
load_dotenv()
api_keys_list = [v for k, v in os.environ.items() if k.startswith("OPENROUTER_API_KEY")]

BEST_PROMPT = """# Task Instruction

You are a fact-checking assistant for entertainment and literary facts, as well as general knowledge facts.
Given a claim and a set of evidence passages, determine whether the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO based only on the evidence.

Carefully reason over the evidence before answering. Consider the following:

**General Knowledge:**
* The presence or absence of concrete evidence, such as publication dates, author names, and film/tv titles.
* The accuracy of geographical information, including the size, population, and locations of cities, countries, and regions.
* The presence or absence of awards and their recipients in various domains, such as sports, literature, and entertainment.

**Entertainment and Literary Facts:**
* The accuracy of domain-specific information, including:
	+ The existence and co-founders of production companies (e.g., Seven Arts Productions).
	+ The involvement of authors, directors, and producers in specific projects.
	+ The genres and characteristics of films and TV shows (e.g., black comedy-drama).
* The potential for ambiguity or confusion in the claim or evidence, such as unclear author identities or multiple projects with similar titles.
* The accuracy of baseball player information, including their names, teams, and achievements.

**Domain-Specific Facts:**
* For literary facts, consider the accuracy of book titles, authors, and publication dates.
* For entertainment facts, consider the accuracy of film and TV show titles, actors, directors, release dates, and awards.

When evidence is inconclusive or insufficient, label the claim as NOT ENOUGH INFO.

Output the final label only."""

def get_evidence_text(sample):
    """Extract and format evidence from a HoVer sample."""
    evidence_texts = []
    
    # Handle list of dictionaries
    if isinstance(sample["supporting_facts"], list):
        for fact in sample["supporting_facts"]:
            if isinstance(fact, dict) and "fact" in fact:
                evidence_texts.append(fact["fact"])
            elif isinstance(fact, str):
                evidence_texts.append(fact)
    # Handle string that might be JSON
    elif isinstance(sample["supporting_facts"], str):
        try:
            facts = json.loads(sample["supporting_facts"])
            for fact in facts:
                if isinstance(fact, dict) and "fact" in fact:
                    evidence_texts.append(fact["fact"])
                elif isinstance(fact, str):
                    evidence_texts.append(fact)
        except json.JSONDecodeError:
            evidence_texts.append(sample["supporting_facts"])
            
    # Fallback to contexts if supporting_facts is empty or parsed empty
    if not evidence_texts and "contexts" in sample:
        if isinstance(sample["contexts"], list):
            for ctx in sample["contexts"]:
                if isinstance(ctx, dict) and "paragraph" in ctx:
                    evidence_texts.append(ctx["paragraph"])
                elif isinstance(ctx, str):
                    evidence_texts.append(ctx)
                    
    return "\n".join(evidence_texts) if evidence_texts else "No evidence provided."

def _parse_label(raw: str) -> str:
    """Extract SUPPORTED / NOT_SUPPORTED from LLM response exactly as done in hover_gepa_main.py."""
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

def robust_completion(model, messages, api_keys):
    """Wrapper to handle OpenRouter API key rotation and rate limits."""
    keys = api_keys.copy()
    
    while keys:
        current_key = keys[0]
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                api_key=current_key,
                temperature=1.0,
                max_tokens=50
            )
        except litellm.exceptions.RateLimitError as e:
            if "HTTP status 429" in str(e):
                print(f"Rate limit exceeded for key ending in ...{current_key[-4:]}. Removing key.")
                keys.pop(0)
            else:
                print(f"Other RateLimitError: {e}. Retrying in 5s.")
                time.sleep(5)
        except Exception as e:
            print(f"Unexpected error with key ending in ...{current_key[-4:]}: {e}. Removing key from rotation.")
            keys.pop(0)
            
    # If all keys exhausted
    raise Exception("All API keys exhausted or rate limited.")

def process_sample(sample, prompt, model_name):
    """Process a single sample through the LLM."""
    claim = sample.get("claim", "")
    target = sample.get("label", "")
    
    try:
        response = robust_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Claim: {claim}"}
            ],
            api_keys=api_keys_list
        )
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        prediction = "ERROR"
        
    pred_label = _parse_label(prediction)
    is_correct = pred_label == target
    
    return {
        "uid": sample.get("uid", ""),
        "claim": claim,
        "target": target,
        "prediction": prediction,
        "pred_label": pred_label,
        "is_correct": is_correct
    }

def evaluate_best_prompt(args):
    """Evaluate the best prompt on the HoVer test split."""
    # Using the exact same dataset loading and splitting logic as hover_gepa_main.py
    # to ensure we evaluate on the exact same 15 test samples.
    import random
    ds = load_dataset("vincentkoc/hover-parquet")
    
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
            
            # evaluate_hover_best.py expects the raw sample format to extract evidence
            # We copy what hover_gepa_main.py does to format the sample
            formatted_sample = {
                "claim": x["claim"],
                "label": label_str,
                # In the original, evaluate_hover_best.py uses get_evidence_text on the sample.
                # However vincentkoc/hover-parquet might have a different structure.
                # In hover_gepa_main.py the adapter does:
                # evidence = ""
                # for ctx in data_inst.get('contexts', []):
                # We need to map the parquet fields back so our current get_evidence_text works
            }
            if "supporting_facts" in x:
                formatted_sample["supporting_facts"] = x["supporting_facts"]
            if "contexts" in x:
                formatted_sample["contexts"] = x["contexts"]
                
            examples.append(formatted_sample)

    random.Random(42).shuffle(examples) # same seed=42

    train_size = 30
    val_size = 15
    test_size = 100
    
    test_samples = examples[train_size + val_size : train_size + val_size + test_size]
    args.num_samples = len(test_samples)
    print(f"Loaded {args.num_samples} test samples from vincentkoc/hover-parquet")
    
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    
    results = []
    correct_count = 0
    
    start_time = time.time()
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = []
        for sample in test_samples:
            futures.append(executor.submit(process_sample, sample, BEST_PROMPT, args.model))
            
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
                
            print(f"Sample {i+1}/{args.num_samples}: Target: {result['target']} | Pred: {result['pred_label']} | Correct: {result['is_correct']}")
            
    elapsed = time.time() - start_time
    accuracy = correct_count / args.num_samples
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {args.num_samples}")
    print(f"Correct:       {correct_count}")
    print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Time Elapsed:  {elapsed:.1f}s")
    print("="*50)
    
    # Save detailed results
    output_file = "hover_best_prompt_eval_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_samples": args.num_samples,
            "results": results
        }, f, indent=2)
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate the best HoVer prompt")
    parser.add_argument("--model", type=str, default="openrouter/qwen/qwen3-8b", help="Model to use via litellm")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=5, help="Concurrent workers for evaluation")
    
    args = parser.parse_args()
    evaluate_best_prompt(args)
