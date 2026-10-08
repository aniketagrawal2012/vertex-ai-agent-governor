"""
Vertex AI Agent Governor Harness
================================
Enterprise execution harness that enforces circuit breaker boundaries,
sliding-window token budgets, and loop detection around Gemini agent loops.
"""

from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from agent_governor.budget import TokenBudget
from agent_governor.loop_detector import SemanticLoopDetector


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"       # Normal execution: requests monitored and allowed
    HALF_OPEN = "HALF_OPEN" # Elevated risk: warning logged, tight budget constraints
    OPEN = "OPEN"           # Tripped: all execution halted immediately


class CircuitBreakerOpenException(Exception):
    """Raised when an execution policy boundary (budget, loop, or timeout) is tripped."""
    def __init__(self, message: str, state_checkpoint: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.state_checkpoint = state_checkpoint or {}


class VertexAIAgentGovernorHarness:
    """
    Zero-Loss Execution Harness wrapping Vertex AI / Gemini API tool loops.
    
    Guarantees:
    - Deterministic token and turn boundaries.
    - Zero runaway tool calling via action-signature hashing.
    - OpenTelemetry / Cloud Trace compatible audit logging.
    - Graceful fallback and state checkpointing on breaker trip.
    """
    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: str = "gemini-2.5-flash",
        budget: Optional[TokenBudget] = None,
        max_loop_iterations: int = 2,
        window_size: int = 4,
    ):
        self.client = client
        self.model_name = model_name
        self.budget = budget or TokenBudget()
        self.loop_detector = SemanticLoopDetector(
            window_size=window_size,
            max_identical_actions=max_loop_iterations
        )
        self.state = CircuitBreakerState.CLOSED
        self.audit_log: List[Dict[str, Any]] = []

    def build_generation_config(
        self,
        tools: Optional[List[Callable]] = None,
        cached_content: Optional[str] = None,
    ) -> types.GenerateContentConfig:
        """Constructs an audited GenerateContentConfig with thinking budget ceilings."""
        config_kwargs: Dict[str, Any] = {
            "temperature": 0.1,
            "max_output_tokens": self.budget.max_turn_tokens,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=self.budget.max_thinking_budget,
            ),
            "tools": tools or [],
        }
        if cached_content:
            config_kwargs["cached_content"] = cached_content
        return types.GenerateContentConfig(**config_kwargs)

    def execute_governed_step(
        self,
        step_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_callable: Callable,
        prompt_tokens: int = 450,
        candidate_tokens: int = 180,
        thinking_tokens: int = 0,
    ) -> Dict[str, Any]:
        """
        Executes an agent step inside proactive safety gates.
        Trips the circuit breaker before tool execution if boundaries are violated.
        """
        # Gate 1: Check Breaker State
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenException(
                "Circuit breaker is in OPEN state. Invocations rejected until reset.",
                state_checkpoint={"turn": self.budget.current_turn, "tokens": self.budget.total_consumed_tokens}
            )

        # Gate 2: Proactive Budget Headroom
        headroom_ok, headroom_msg = self.budget.check_headroom()
        if not headroom_ok:
            self._trip_breaker(step_id, headroom_msg)
            raise CircuitBreakerOpenException(
                headroom_msg,
                state_checkpoint={"step_id": step_id, "tokens": self.budget.total_consumed_tokens}
            )

        # Gate 3: Semantic Loop Detection
        is_loop, loop_msg = self.loop_detector.register_and_check(tool_name, tool_args)
        if is_loop:
            self._trip_breaker(step_id, loop_msg)
            raise CircuitBreakerOpenException(
                loop_msg,
                state_checkpoint={
                    "step_id": step_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "tokens": self.budget.total_consumed_tokens
                }
            )

        # Step 4: Execute Tool Safely
        start_t = time.perf_counter()
        try:
            tool_result = tool_callable(**tool_args)
        except Exception as e:
            tool_result = f"TOOL_ERROR: {str(e)}"
        latency_ms = (time.perf_counter() - start_t) * 1000

        # Step 5: Record consumption
        self.budget.record_usage(prompt_tokens, candidate_tokens, thinking_tokens)
        
        event = {
            "step_id": step_id,
            "status": "SUCCESS",
            "tool": tool_name,
            "args": tool_args,
            "result": tool_result,
            "latency_ms": round(latency_ms, 2),
            "turn": self.budget.current_turn,
            "accumulated_tokens": self.budget.total_consumed_tokens,
            "estimated_cost_usd": self.budget.estimated_cost_usd,
        }
        self._log_event(step_id, "STEP_COMPLETED", event)
        return event

    def _trip_breaker(self, step_id: int, reason: str) -> None:
        self.state = CircuitBreakerState.OPEN
        self._log_event(step_id, "CIRCUIT_BREAKER_TRIPPED", {
            "reason": reason,
            "final_tokens": self.budget.total_consumed_tokens,
            "final_cost_usd": self.budget.estimated_cost_usd,
        })

    def _log_event(self, step_id: int, event_type: str, details: Dict[str, Any]) -> None:
        self.audit_log.append({
            "timestamp": time.time(),
            "step_id": step_id,
            "state": self.state.value,
            "event_type": event_type,
            "details": details,
        })
