"""
Token Budget Governor for Vertex AI Agents
==========================================
Enforces deterministic token ceilings, turn constraints, and cost tracking
for autonomous agent execution loops.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TokenBudget:
    """Enforces mathematical boundaries on agent token consumption."""
    max_total_tokens: int = 16_384
    max_turn_tokens: int = 4_096
    max_turns: int = 8
    max_thinking_budget: int = 2_048
    cost_per_1k_input: float = 0.00015   # Standard Gemini Flash tier pricing
    cost_per_1k_output: float = 0.00060

    # Live telemetry state
    accumulated_prompt_tokens: int = 0
    accumulated_candidate_tokens: int = 0
    accumulated_thinking_tokens: int = 0
    current_turn: int = 0

    @property
    def total_consumed_tokens(self) -> int:
        return self.accumulated_prompt_tokens + self.accumulated_candidate_tokens + self.accumulated_thinking_tokens

    @property
    def estimated_cost_usd(self) -> float:
        input_cost = (self.accumulated_prompt_tokens / 1000.0) * self.cost_per_1k_input
        output_cost = ((self.accumulated_candidate_tokens + self.accumulated_thinking_tokens) / 1000.0) * self.cost_per_1k_output
        return round(input_cost + output_cost, 6)

    def check_headroom(self) -> Tuple[bool, str]:
        """Proactively checks whether execution remains within safe boundaries."""
        if self.current_turn >= self.max_turns:
            return False, f"Turn limit exceeded: {self.current_turn}/{self.max_turns} turns executed."
        if self.total_consumed_tokens >= self.max_total_tokens:
            return False, f"Hard token budget exhausted: {self.total_consumed_tokens}/{self.max_total_tokens} tokens consumed."
        return True, "Headroom verified."

    def record_usage(
        self,
        prompt_tokens: int,
        candidate_tokens: int,
        thinking_tokens: int = 0
    ) -> None:
        """Records token consumption from a completed model turn."""
        self.accumulated_prompt_tokens += prompt_tokens
        self.accumulated_candidate_tokens += candidate_tokens
        self.accumulated_thinking_tokens += thinking_tokens
        self.current_turn += 1
