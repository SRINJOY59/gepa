# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Bandit-based strategies for multi-prompt mutation in GEPA.

This module provides:
- MutationStrategy: Enum of different mutation approaches
- ThompsonSamplingBandit: Learns which mutation strategies work best
- MultiPromptGenerator: Generates K prompt variants using a selected strategy
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np


class MutationStrategy(Enum):
    """Types of prompt mutation strategies."""
    
    ADD_EXAMPLES = "add_examples"
    ADD_CONSTRAINTS = "add_constraints"
    SIMPLIFY = "simplify"
    ADD_COT = "add_cot"  # Chain of thought
    RESTRUCTURE = "restructure"
    DOMAIN_INJECT = "domain_inject"


# Strategy-specific meta-prompts for generating variants
STRATEGY_PROMPTS: dict[MutationStrategy, str] = {
    MutationStrategy.ADD_EXAMPLES: """Improve this instruction by adding 2-3 worked examples that demonstrate the expected behavior.

Current instruction:
```
{base_instruction}
```

Based on these task examples and feedback:
{feedback}

Provide the improved instruction with examples within ``` blocks.""",

    MutationStrategy.ADD_CONSTRAINTS: """Improve this instruction by adding explicit constraints, edge cases, and boundary conditions.

Current instruction:
```
{base_instruction}
```

Based on these task examples and feedback:
{feedback}

Provide the improved instruction with constraints within ``` blocks.""",

    MutationStrategy.SIMPLIFY: """Simplify and make this instruction more concise while preserving the core meaning.

Current instruction:
```
{base_instruction}
```

Based on these task examples and feedback:
{feedback}

Provide the simplified instruction within ``` blocks.""",

    MutationStrategy.ADD_COT: """Improve this instruction by adding explicit step-by-step reasoning guidance.

Current instruction:
```
{base_instruction}
```

Based on these task examples and feedback:
{feedback}

Provide the improved instruction with reasoning steps within ``` blocks.""",

    MutationStrategy.RESTRUCTURE: """Restructure this instruction into a clearer format with numbered steps and sections.

Current instruction:
```
{base_instruction}
```

Based on these task examples and feedback:
{feedback}

Provide the restructured instruction within ``` blocks.""",

    MutationStrategy.DOMAIN_INJECT: """Improve this instruction by incorporating domain-specific knowledge, terminology, and patterns from the feedback.

Current instruction:
```
{base_instruction}
```

Based on these task examples and feedback:
{feedback}

Provide the improved instruction with domain knowledge within ``` blocks.""",
}


class LanguageModelProtocol(Protocol):
    """Protocol for language model callable."""
    def __call__(self, prompt: str) -> str: ...


@dataclass
class ThompsonSamplingBandit:
    """
    Thompson Sampling bandit for selecting mutation strategies.
    
    Maintains Beta distribution parameters (alpha, beta) for each strategy.
    Samples from posteriors to balance exploration vs exploitation.
    """
    
    # Beta distribution parameters: Beta(alpha, beta)
    # alpha = successes + 1, beta = failures + 1 (with prior Beta(1,1))
    alpha: dict[MutationStrategy, float] = field(default_factory=lambda: {
        s: 1.0 for s in MutationStrategy
    })
    beta: dict[MutationStrategy, float] = field(default_factory=lambda: {
        s: 1.0 for s in MutationStrategy
    })
    
    # Optional RNG for reproducibility
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    
    def select_strategy(self) -> MutationStrategy:
        """
        Select a mutation strategy using Thompson Sampling.
        
        Samples from the Beta posterior for each strategy and returns
        the strategy with the highest sample.
        """
        samples = {
            strategy: self.rng.beta(self.alpha[strategy], self.beta[strategy])
            for strategy in MutationStrategy
        }
        return max(samples, key=lambda s: samples[s])
    
    def update(self, strategy: MutationStrategy, improved: bool) -> None:
        """
        Update the Beta parameters based on the outcome.
        
        Args:
            strategy: The strategy that was used
            improved: Whether the mutation improved over the parent
        """
        if improved:
            self.alpha[strategy] += 1.0
        else:
            self.beta[strategy] += 1.0
    
    def get_strategy_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for each strategy (for logging/debugging)."""
        stats = {}
        for strategy in MutationStrategy:
            a, b = self.alpha[strategy], self.beta[strategy]
            stats[strategy.value] = {
                "alpha": a,
                "beta": b,
                "mean": a / (a + b),
                "trials": a + b - 2,  # Subtract prior
            }
        return stats
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize bandit state to dictionary."""
        return {
            "alpha": {s.value: v for s, v in self.alpha.items()},
            "beta": {s.value: v for s, v in self.beta.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], rng: np.random.Generator | None = None) -> "ThompsonSamplingBandit":
        """Deserialize bandit state from dictionary."""
        alpha = {MutationStrategy(k): v for k, v in data["alpha"].items()}
        beta = {MutationStrategy(k): v for k, v in data["beta"].items()}
        return cls(
            alpha=alpha,
            beta=beta,
            rng=rng if rng is not None else np.random.default_rng(),
        )


@dataclass
class MultiPromptGenerator:
    """
    Generates multiple prompt variants using a selected mutation strategy.
    """
    
    def generate_variants(
        self,
        base_instruction: str,
        feedback_text: str,
        strategy: MutationStrategy,
        lm: LanguageModelProtocol,
        k: int = 10,
    ) -> list[str]:
        """
        Generate K prompt variants using the given mutation strategy.
        
        Args:
            base_instruction: The current prompt to mutate
            feedback_text: Formatted feedback from reflective dataset
            strategy: The mutation strategy to apply
            lm: Language model callable for generating variants
            k: Number of variants to generate
            
        Returns:
            List of K generated prompt variants
        """
        meta_prompt = STRATEGY_PROMPTS[strategy].format(
            base_instruction=base_instruction,
            feedback=feedback_text,
        )
        
        variants = []
        for _ in range(k):
            try:
                response = lm(meta_prompt)
                # Extract instruction from response (similar to InstructionProposalSignature)
                extracted = self._extract_instruction(response)
                if extracted and extracted not in variants:
                    variants.append(extracted)
                else:
                    # If duplicate or empty, try to get unique variant
                    variants.append(response.strip())
            except Exception:
                # On error, keep the base instruction as fallback
                if base_instruction not in variants:
                    variants.append(base_instruction)
        
        # Ensure we have exactly K variants (pad with base if needed)
        while len(variants) < k:
            variants.append(base_instruction)
        
        return variants[:k]
    
    def _extract_instruction(self, lm_output: str) -> str:
        """Extract instruction text from LM output (handles code blocks)."""
        import re
        
        # Find the first and last backtick positions
        start = lm_output.find("```") + 3
        end = lm_output.rfind("```")
        
        if start >= end or start == 2:  # No valid block found
            return lm_output.strip()
        
        content = lm_output[start:end]
        
        # Skip optional language specifier
        match = re.match(r"^\S*\n", content)
        if match:
            content = content[match.end():]
        
        return content.strip()
    
    def format_feedback(self, reflective_dataset: list[dict[str, Any]]) -> str:
        """Format reflective dataset into feedback text for prompts."""
        if not reflective_dataset:
            return "(No feedback available)"
        
        parts = []
        for i, sample in enumerate(reflective_dataset[:5], 1):  # Limit to 5 examples
            part = f"Example {i}:\n"
            for key, value in sample.items():
                if isinstance(value, str):
                    part += f"  {key}: {value[:500]}...\n" if len(str(value)) > 500 else f"  {key}: {value}\n"
                else:
                    part += f"  {key}: {value}\n"
            parts.append(part)
        
        return "\n".join(parts)
