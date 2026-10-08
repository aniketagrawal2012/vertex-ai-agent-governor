"""
Unit tests for Vertex AI Agent Governor Harness
"""

import unittest
from agent_governor.budget import TokenBudget
from agent_governor.loop_detector import SemanticLoopDetector
from agent_governor.governor import (
    VertexAIAgentGovernorHarness,
    CircuitBreakerState,
    CircuitBreakerOpenException,
)


def test_token_budget_headroom():
    budget = TokenBudget(max_total_tokens=1000, max_turns=3)
    ok, msg = budget.check_headroom()
    assert ok is True

    budget.record_usage(prompt_tokens=400, candidate_tokens=200)
    assert budget.total_consumed_tokens == 600
    assert budget.current_turn == 1

    budget.record_usage(prompt_tokens=300, candidate_tokens=200)
    assert budget.total_consumed_tokens == 1100
    ok, msg = budget.check_headroom()
    assert ok is False
    assert "Hard token budget exhausted" in msg


def test_turn_limit_exhaustion():
    budget = TokenBudget(max_total_tokens=100_000, max_turns=2)
    budget.record_usage(10, 10)
    budget.record_usage(10, 10)
    ok, msg = budget.check_headroom()
    assert ok is False
    assert "Turn limit exceeded" in msg


def test_loop_detector_single_action():
    detector = SemanticLoopDetector(window_size=4, max_identical_actions=2)
    
    # 1st call
    loop, _ = detector.register_and_check("query_db", {"table": "users"})
    assert loop is False

    # 2nd call
    loop, _ = detector.register_and_check("query_db", {"table": "users"})
    assert loop is False

    # 3rd call -> exceeds max_identical_actions (2)
    loop, msg = detector.register_and_check("query_db", {"table": "users"})
    assert loop is True
    assert "Repetitive loop detected" in msg


def test_loop_detector_oscillatory_cycle():
    detector = SemanticLoopDetector(window_size=4, max_identical_actions=3)
    
    detector.register_and_check("tool_a", {"id": 1})
    detector.register_and_check("tool_b", {"id": 2})
    detector.register_and_check("tool_a", {"id": 1})
    loop, msg = detector.register_and_check("tool_b", {"id": 2})
    
    assert loop is True
    assert "Oscillatory 2-cycle detected" in msg


def test_circuit_breaker_harness_integration():
    harness = VertexAIAgentGovernorHarness(
        budget=TokenBudget(max_total_tokens=5000, max_turns=5),
        max_loop_iterations=2
    )

    def dummy_tool(x: int) -> str:
        return f"result_{x}"

    # Step 1
    res = harness.execute_governed_step(1, "dummy_tool", {"x": 1}, dummy_tool)
    assert res["status"] == "SUCCESS"
    assert harness.state == CircuitBreakerState.CLOSED

    # Step 2
    res = harness.execute_governed_step(2, "dummy_tool", {"x": 1}, dummy_tool)
    assert res["status"] == "SUCCESS"

    # Step 3: Repeated identical invocation trips breaker
    tripped = False
    try:
        harness.execute_governed_step(3, "dummy_tool", {"x": 1}, dummy_tool)
    except CircuitBreakerOpenException:
        tripped = True
    assert tripped, "CircuitBreakerOpenException should have been raised!"

    assert harness.state == CircuitBreakerState.OPEN


if __name__ == "__main__":
    test_token_budget_headroom()
    test_turn_limit_exhaustion()
    test_loop_detector_single_action()
    test_loop_detector_oscillatory_cycle()
    test_circuit_breaker_harness_integration()
    print("All unit tests passed successfully!")
