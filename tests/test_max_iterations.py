from unittest.mock import Mock
from gepa import optimize

def test_max_iterations():
    """Test that optimization stops after max_iterations."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        return "```\nimproved instructions\n```"

    # Set up a situation where it would take many metric calls to stop, 
    # but we limit it by iterations.
    result = optimize(
        seed_candidate={"instructions": "initial instructions"},
        trainset=mock_data,
        task_lm=task_lm,
        reflection_lm=mock_reflection_lm,
        max_iterations=2,
        # Set a very high metric call budget so it doesn't stop because of that
        max_metric_calls=1000,
        reflection_minibatch_size=1,
    )

    # Iteration count in result should be exactly 2
    # In GEPAResult, from_state sets iterations = state.i
    assert result.num_iterations == 2

if __name__ == "__main__":
    test_max_iterations()
    print("Test passed!")
