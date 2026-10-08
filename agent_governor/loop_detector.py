"""
Semantic Loop Detector for Vertex AI Agents
===========================================
Detects cyclic, repetitive, and oscillatory tool executions using
canonical action-signature hashing and sliding-window n-gram analysis.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


class SemanticLoopDetector:
    """
    Detects infinite agent loops before they drain inference budgets.
    
    Catches two primary failure patterns:
    1. Single-action repetition: Calling Tool A with identical arguments >= N times.
    2. Oscillatory cycles: Alternating between Tool A and Tool B in cyclic loops (e.g. A -> B -> A -> B).
    """
    def __init__(self, window_size: int = 4, max_identical_actions: int = 2):
        self.window_size = window_size
        self.max_identical_actions = max_identical_actions
        self.action_history: List[str] = []
        self.raw_invocations: List[Dict[str, Any]] = []

    def compute_signature(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Computes a deterministic SHA-256 fingerprint from tool name and sorted arguments."""
        canonical_payload = {
            "tool": tool_name.strip(),
            "args": args
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def register_and_check(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Registers an intended action and evaluates if it violates loop policies.
        Returns: (is_loop_detected, reason_message)
        """
        sig = self.compute_signature(tool_name, args)
        self.action_history.append(sig)
        self.raw_invocations.append({"tool": tool_name, "args": args, "sig": sig})
        
        # Check 1: Sliding window identical invocation count
        window = self.action_history[-self.window_size:]
        sig_count = window.count(sig)
        if sig_count > self.max_identical_actions:
            return True, f"Repetitive loop detected: Tool '{tool_name}' (sig: {sig}) invoked {sig_count} times in last {len(window)} steps."
        
        # Check 2: Alternating 2-cycle oscillation [A, B, A, B]
        if len(window) >= 4:
            if window[-1] == window[-3] and window[-2] == window[-4] and window[-1] != window[-2]:
                return True, f"Oscillatory 2-cycle detected: Alternating loop between signatures {window[-2]} and {window[-1]}."

        # Check 3: Alternating 3-cycle oscillation [A, B, C, A, B, C]
        if len(self.action_history) >= 6:
            h = self.action_history
            if h[-1] == h[-4] and h[-2] == h[-5] and h[-3] == h[-6] and len({h[-1], h[-2], h[-3]}) == 3:
                return True, f"Oscillatory 3-cycle detected across 6 consecutive steps."

        return False, "Nominal execution."

    def reset(self) -> None:
        """Resets the detector history for a fresh session."""
        self.action_history.clear()
        self.raw_invocations.clear()
