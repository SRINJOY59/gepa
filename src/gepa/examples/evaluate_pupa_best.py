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

BEST_PROMPT_REDACTION = """## Task Instruction

### Task Description

The task involves generating a rewritten, privacy-preserving request for a powerful external LLM, without including any personal identifiable details or sensitive information. The request should assist the user without learning private information about them. This task requires the assistant to understand the importance of protecting user privacy and strive to minimize leakage in all outputs.

### Input Format

* The input will be a free-form text query from a user, representing their request or question.
* The query may contain sensitive information that requires protection.
* The query may relate to various topics, including but not limited to:
	+ Product or service-related queries
	+ Tax-related queries (e.g., withholding tax returns)
	+ Marketing and advertising strategies (e.g., digital marketing, affiliate marketing)
	+ Accessibility and disability-related queries

### Output Requirements

* The assistant must output a rewritten, privacy-preserving request that maintains the original intent of the user's query.
* The output should be concise and focus on the essential components of the user's query.
* The request should be framed in a way that the LLM can process it without compromising user privacy.

### Secondary Assistive Tasks

* The assistant may be required to:
	+ Identify and remove personal identifiable information (PII) from the user's query.
	+ Correct grammatical errors and ensure the output is coherent and readable.
	+ Rewrite the query in a way that maintains its original intent while protecting sensitive information.
	+ Provide a rewritten query that can be safely provided to an external LLM without compromising user privacy.

### Niche and Domain-Specific Information

* When processing tax-related queries, the assistant should be aware of the typical structure and format of tax returns, as well as common terminology and regulations.
* When dealing with queries related to products or services, the assistant should have knowledge of marketing campaigns, advertising strategies, and common product description formats.
* The assistant should understand the importance of protecting PII and strive to minimize leakage in all outputs.
* The assistant should be familiar with accessibility and disability-related topics, including accommodations and equal access to programs.

### Generalizable Strategies

* The assistant should focus on maintaining the essential intent and information of the original query.
* The assistant should avoid introducing new information or assumptions that may compromise user privacy.
* The assistant should strive to produce outputs that can be universally applied, regardless of the specific user query.

### Instructional Notes

Please prioritize producing outputs that are accurate, concise, and effective in their intent. Focus on minimizing PII leakage and protecting user privacy throughout the task.

### Expected Output

* A rewritten, privacy-preserving request for the external LLM, without including any personal identifiable details or sensitive information.
* The output should be formatted in a clear and readable manner, with a focus on maintaining the original intent of the user's query."""

BEST_PROMPT_RESPONSE = """# New Instruction

## Format:
1. user_query: The user's original request you need to fulfill.
2. related_llm_request: A rewritten request for the external LLM, maintaining the original intent while protecting sensitive information.
3. related_llm_response: Information from a powerful LLM responding to the related request.

## Task:
Given a user's query, the related LLM request, and the response from the external LLM, generate a final response that summarizes the results of the external LLM's response, maintaining the quality and conciseness of the original response. Remove any sensitive or personal identifiable information (PII) from the response.

## Expected Response:
The final response should include:
- A concise and clear summary of the external LLM's response.
- Removal of any PII or sensitive information to protect user privacy.
- Preservation of the original intent and context of the user's query.
- A focus on maintaining generalizability and applicability of the response.

## Strategies for Generalizability:
- Identify and maintain the essential intent and context of the user's query.
- Focus on conciseness and clarity in the final response.
- Use generalizable strategies and techniques to solve the task, such as:
    * Removing PII and sensitive information without compromising essential context.
    * Extracting relevant information from the LLM's response while omitting unnecessary details.
    * Using technical terms and concepts relevant to the domain or niche.
- Consider niche and domain-specific factual information, such as:
    * In the context of news writing and journalism, this includes understanding the importance of maintaining confidentiality and accuracy in articles and content.
    * In the context of business transactions, this includes recognizing the need to remove PII and sensitive information, such as names, phone numbers, and payment details.

## Feedback Guidelines:
- Quality: 1.0 (high-quality response that maintains the original intent and context)
- Privacy: 1.0 (removal of all PII and sensitive information)
- Overall: 1.0 (exceptional response that meets all expectations)"""

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
                temperature=1.0,  # 1.0 matched the user modification in hover
                max_tokens=800    # Default max tokens for longer generations
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
            
    raise Exception("All API keys exhausted or rate limited.")

def _llm_judge_leakage(pii_units: str, redacted_request: str, judge_lm: str) -> float:
    pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
    if not pii_list:
        return 0.0

    prompt = (
        "Count the number of information pieces in `pii` that "
        "(i) are forms of PII *and* (ii) are leaked into the prompt "
        "given. Don't count celebrity names. Don't count things that "
        "do not appear in the prompt.\\n\\n"
        f"pii: {pii_list}\\n"
        f"prompt: {redacted_request}\\n\\n"
        "Return ONLY a number."
    )
    try:
        resp = robust_completion(
            model=judge_lm,
            messages=[{"role": "user", "content": prompt}],
            api_keys=api_keys_list,
        )
        leaked = int(resp.choices[0].message.content.strip())
    except Exception:
        # Fallback to precise string matching if LLM fails
        leaked = sum(1 for p in pii_list if p.lower() in redacted_request.lower())

    # Bound exactly to maximum of length of pii_list
    leaked = min(leaked, len(pii_list))
    return leaked / len(pii_list)

def _llm_judge_quality(user_query: str, target_response: str, pred_response: str, judge_lm: str) -> bool:
    prompt = (
        "You are comparing the quality of two responses.\\n"
        "Is response A at least as good as response B?\\n\\n"
        f"User Query: {user_query}\\n\\n"
        f"Response A (Prediction): {pred_response}\\n\\n"
        f"Response B (Target): {target_response}\\n\\n"
        "Return ONLY 'true' or 'false'."
    )
    try:
        resp = robust_completion(
            model=judge_lm,
            messages=[{"role": "user", "content": prompt}],
            api_keys=api_keys_list,
        )
        return "true" in resp.choices[0].message.content.lower()
    except Exception:
        return False

def process_sample(sample, task_lm, untrusted_lm):
    """Process a single sample through the 2-step PUPA pipeline and evaluate it."""
    user_query = sample.get("input", "")
    target_response = sample.get("target_response", "")
    pii_units = sample.get("pii_units", "")
    
    try:
        # 1. Generate privacy-preserving redacted request
        redact_resp = robust_completion(
            model=task_lm,
            messages=[
                {"role": "system", "content": BEST_PROMPT_REDACTION},
                {"role": "user", "content": f"User query: {user_query}"},
            ],
            api_keys=api_keys_list,
        )
        redacted_request = redact_resp.choices[0].message.content

        # 2. Query untrusted external LLM with the redacted request
        untrusted_resp = robust_completion(
            model=untrusted_lm,
            messages=[{"role": "user", "content": redacted_request}],
            api_keys=api_keys_list,
        )
        external_response = untrusted_resp.choices[0].message.content

        # 3. Generate final response based on external response and original query
        final_resp = robust_completion(
            model=task_lm,
            messages=[
                {"role": "system", "content": BEST_PROMPT_RESPONSE},
                {
                    "role": "user",
                    "content": (
                        f"Original user query: {user_query}\\n\\n"
                        f"External LM request: {redacted_request}\\n\\n"
                        f"External LM response: {external_response}\\n\\n"
                        "Generate the final response for the user."
                    ),
                },
            ],
            api_keys=api_keys_list,
        )
        final_response = final_resp.choices[0].message.content

        # Evaluation
        leakage = _llm_judge_leakage(pii_units, redacted_request, task_lm)
        privacy_score = 1.0 - leakage

        is_good = _llm_judge_quality(user_query, target_response, final_response, task_lm)
        quality_score = 1.0 if is_good else 0.0

        overall_score = (privacy_score + quality_score) / 2.0

        return {
            "uid": sample.get("uid", "N/A"),
            "success": True,
            "privacy_score": privacy_score,
            "quality_score": quality_score,
            "overall_score": overall_score,
            "error": None
        }

    except Exception as e:
        print(f"Error processing sample: {e}")
        return {
            "uid": sample.get("uid", "N/A"),
            "success": False,
            "privacy_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "error": str(e)
        }

def evaluate_best_prompt(args):
    """Evaluate the best PUPA prompt on the test split."""
    import random
    
    print("Loading PUPA dataset...")
    pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")
    examples = [
        {
            "input": x["user_query"],
            "target_response": x["target_response"],
            "pii_units": x["pii_units"],
        }
        for x in pupa_new["train"]
    ]

    random.Random(42).shuffle(examples)

    # In original: train=225, val=225, test=214
    # The default args was using train=20, val=10, test=10
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    
    test_samples = examples[train_size + val_size : train_size + val_size + test_size]
    print(f"Loaded {len(test_samples)} test samples from Columbia-NLP/PUPA")
    
    print(f"Task Model: {args.task_lm}")
    print(f"Untrusted Model: {args.untrusted_lm}")
    print(f"Batch size: {args.batch_size}")
    
    results = []
    total_privacy = 0.0
    total_quality = 0.0
    total_overall = 0.0
    successful_samples = 0
    
    start_time = time.time()
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = []
        for sample in test_samples:
            futures.append(executor.submit(process_sample, sample, args.task_lm, args.untrusted_lm))
            
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                successful_samples += 1
                total_privacy += result["privacy_score"]
                total_quality += result["quality_score"]
                total_overall += result["overall_score"]
                
            print(f"Sample {i+1}/{len(test_samples)}: Overall: {result['overall_score']:.2f} (Privacy: {result['privacy_score']:.2f}, Quality: {result['quality_score']:.2f})")
            
    elapsed = time.time() - start_time
    
    if successful_samples > 0:
        avg_privacy = total_privacy / successful_samples
        avg_quality = total_quality / successful_samples
        avg_overall = total_overall / successful_samples
    else:
        avg_privacy, avg_quality, avg_overall = 0.0, 0.0, 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {len(test_samples)}")
    print(f"Successful:    {successful_samples}")
    print(f"Avg Privacy:   {avg_privacy:.4f}")
    print(f"Avg Quality:   {avg_quality:.4f}")
    print(f"Avg Overall:   {avg_overall:.4f}")
    print(f"Time Elapsed:  {elapsed:.1f}s")
    print("="*50)
    
    # Save detailed results
    output_file = "pupa_best_prompt_eval_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "avg_privacy": avg_privacy,
            "avg_quality": avg_quality,
            "avg_overall": avg_overall,
            "successful_samples": successful_samples,
            "total_samples": len(test_samples),
            "results": results
        }, f, indent=2)
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate the best PUPA prompt")
    parser.add_argument("--task_lm", type=str, default="openrouter/qwen/qwen3-8b", help="Model to use via litellm")
    parser.add_argument("--untrusted_lm", type=str, default="openrouter/qwen/qwen3-8b", help="Untrusted model via litellm")
    parser.add_argument("--train_size", type=int, default=20, help="Original train size used for splitting")
    parser.add_argument("--val_size", type=int, default=10, help="Original val size used for splitting")
    parser.add_argument("--test_size", type=int, default=20, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=5, help="Concurrent workers for evaluation")
    
    args = parser.parse_args()
    evaluate_best_prompt(args)
