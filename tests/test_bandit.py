# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import numpy as np
import pytest

from gepa.strategies.bandit import (
    MutationStrategy,
    MultiPromptGenerator,
    STRATEGY_PROMPTS,
    ThompsonSamplingBandit,
)


class TestMutationStrategy:
    """Test MutationStrategy enum."""

    def test_all_strategies_have_prompts(self):
        """Ensure all mutation strategies have corresponding prompts."""
        for strategy in MutationStrategy:
            assert strategy in STRATEGY_PROMPTS
            assert len(STRATEGY_PROMPTS[strategy]) > 0

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert MutationStrategy.ADD_EXAMPLES.value == "add_examples"
        assert MutationStrategy.SIMPLIFY.value == "simplify"
        assert MutationStrategy.ADD_COT.value == "add_cot"


class TestThompsonSamplingBandit:
    """Test ThompsonSamplingBandit class."""

    def test_initialization(self):
        """Test default initialization."""
        bandit = ThompsonSamplingBandit()
        
        # All strategies should have alpha=1, beta=1 (Beta(1,1) prior)
        for strategy in MutationStrategy:
            assert bandit.alpha[strategy] == 1.0
            assert bandit.beta[strategy] == 1.0

    def test_select_strategy_returns_valid_strategy(self):
        """Test that select_strategy returns a valid MutationStrategy."""
        bandit = ThompsonSamplingBandit()
        
        for _ in range(20):
            strategy = bandit.select_strategy()
            assert isinstance(strategy, MutationStrategy)
            assert strategy in MutationStrategy

    def test_seeding_produces_deterministic_sequence(self):
        """Test that same seed produces same sequence."""
        bandit1 = ThompsonSamplingBandit(rng=np.random.default_rng(42))
        bandit2 = ThompsonSamplingBandit(rng=np.random.default_rng(42))
        
        results1 = [bandit1.select_strategy() for _ in range(10)]
        results2 = [bandit2.select_strategy() for _ in range(10)]
        
        assert results1 == results2

    def test_update_success_increases_alpha(self):
        """Test that successful update increases alpha."""
        bandit = ThompsonSamplingBandit()
        strategy = MutationStrategy.ADD_EXAMPLES
        
        initial_alpha = bandit.alpha[strategy]
        initial_beta = bandit.beta[strategy]
        
        bandit.update(strategy, improved=True)
        
        assert bandit.alpha[strategy] == initial_alpha + 1
        assert bandit.beta[strategy] == initial_beta  # beta unchanged

    def test_update_failure_increases_beta(self):
        """Test that failed update increases beta."""
        bandit = ThompsonSamplingBandit()
        strategy = MutationStrategy.SIMPLIFY
        
        initial_alpha = bandit.alpha[strategy]
        initial_beta = bandit.beta[strategy]
        
        bandit.update(strategy, improved=False)
        
        assert bandit.alpha[strategy] == initial_alpha  # alpha unchanged
        assert bandit.beta[strategy] == initial_beta + 1

    def test_learned_bandit_favors_successful_strategies(self):
        """Test that bandit learns to favor strategies with better outcomes."""
        bandit = ThompsonSamplingBandit(rng=np.random.default_rng(42))
        
        # Simulate: ADD_EXAMPLES always succeeds, SIMPLIFY always fails
        for _ in range(50):
            bandit.update(MutationStrategy.ADD_EXAMPLES, improved=True)
            bandit.update(MutationStrategy.SIMPLIFY, improved=False)
        
        # Sample many times and count
        selections = [bandit.select_strategy() for _ in range(100)]
        add_examples_count = sum(1 for s in selections if s == MutationStrategy.ADD_EXAMPLES)
        simplify_count = sum(1 for s in selections if s == MutationStrategy.SIMPLIFY)
        
        # ADD_EXAMPLES should be selected much more often
        assert add_examples_count > simplify_count * 5

    def test_get_strategy_stats(self):
        """Test get_strategy_stats returns correct format."""
        bandit = ThompsonSamplingBandit()
        bandit.update(MutationStrategy.ADD_COT, improved=True)
        bandit.update(MutationStrategy.ADD_COT, improved=False)
        
        stats = bandit.get_strategy_stats()
        
        assert "add_cot" in stats
        assert "alpha" in stats["add_cot"]
        assert "beta" in stats["add_cot"]
        assert "mean" in stats["add_cot"]
        assert "trials" in stats["add_cot"]
        
        # 2 trials, 1 success
        assert stats["add_cot"]["trials"] == 2
        assert stats["add_cot"]["alpha"] == 2.0  # 1 + 1 success
        assert stats["add_cot"]["beta"] == 2.0   # 1 + 1 failure

    def test_serialization(self):
        """Test to_dict and from_dict."""
        bandit = ThompsonSamplingBandit()
        bandit.update(MutationStrategy.DOMAIN_INJECT, improved=True)
        bandit.update(MutationStrategy.RESTRUCTURE, improved=False)
        
        data = bandit.to_dict()
        restored = ThompsonSamplingBandit.from_dict(data)
        
        for strategy in MutationStrategy:
            assert restored.alpha[strategy] == bandit.alpha[strategy]
            assert restored.beta[strategy] == bandit.beta[strategy]


class TestMultiPromptGenerator:
    """Test MultiPromptGenerator class."""

    def test_format_feedback_empty(self):
        """Test formatting empty feedback."""
        generator = MultiPromptGenerator()
        result = generator.format_feedback([])
        assert result == "(No feedback available)"

    def test_format_feedback_with_data(self):
        """Test formatting feedback with data."""
        generator = MultiPromptGenerator()
        feedback = [
            {"input": "test input", "output": "test output", "error": "wrong"},
            {"input": "second", "output": "result"},
        ]
        result = generator.format_feedback(feedback)
        
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "test input" in result
        assert "input:" in result

    def test_format_feedback_limits_examples(self):
        """Test that feedback is limited to 5 examples."""
        generator = MultiPromptGenerator()
        feedback = [{"id": i} for i in range(10)]
        result = generator.format_feedback(feedback)
        
        # Should have at most 5 examples
        assert result.count("Example") <= 5

    def test_extract_instruction_from_code_block(self):
        """Test extracting instruction from markdown code block."""
        generator = MultiPromptGenerator()
        
        lm_output = """Here's the improved instruction:
```
This is the extracted instruction.
```
"""
        result = generator._extract_instruction(lm_output)
        assert result == "This is the extracted instruction."

    def test_extract_instruction_with_language_specifier(self):
        """Test extracting instruction from code block with language."""
        generator = MultiPromptGenerator()
        
        lm_output = """```markdown
Instruction content here.
```"""
        result = generator._extract_instruction(lm_output)
        assert result == "Instruction content here."

    def test_extract_instruction_no_code_block(self):
        """Test extracting instruction when no code block present."""
        generator = MultiPromptGenerator()
        
        lm_output = "Just plain text instruction"
        result = generator._extract_instruction(lm_output)
        assert result == "Just plain text instruction"

    def test_generate_variants_returns_k_variants(self):
        """Test that generate_variants returns K variants."""
        generator = MultiPromptGenerator()
        
        # Mock LM that returns variations
        call_count = [0]
        def mock_lm(prompt: str) -> str:
            call_count[0] += 1
            return f"```\nVariant {call_count[0]}\n```"
        
        variants = generator.generate_variants(
            base_instruction="Original instruction",
            feedback_text="Some feedback",
            strategy=MutationStrategy.ADD_EXAMPLES,
            lm=mock_lm,
            k=10,
        )
        
        assert len(variants) == 10
        # Should have unique variants
        assert "Variant 1" in variants[0]

    def test_generate_variants_pads_with_base_if_needed(self):
        """Test that variants are padded with base instruction if needed."""
        generator = MultiPromptGenerator()
        
        # Mock LM that fails
        def failing_lm(prompt: str) -> str:
            raise Exception("LM failed")
        
        variants = generator.generate_variants(
            base_instruction="Base instruction",
            feedback_text="Feedback",
            strategy=MutationStrategy.SIMPLIFY,
            lm=failing_lm,
            k=5,
        )
        
        assert len(variants) == 5
        # All should be base instruction due to failures
        assert all(v == "Base instruction" for v in variants)
