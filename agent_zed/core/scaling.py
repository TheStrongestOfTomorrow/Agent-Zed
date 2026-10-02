"""Inference-Time Compute Scaling Pipeline for Agent-Zed."""

import re
from collections import Counter
from typing import List, Dict, Any, Callable, Optional, Tuple
from agent_zed.core.sandbox import CodeSandbox

class DeterministicVerifier:
    """Verifies candidate code and reasoning outputs using programmatic code rules."""

    @staticmethod
    def verify_syntax_and_structure(code_str: str) -> bool:
        """Rule 1: Check Python AST syntax validity."""
        is_valid, _ = CodeSandbox.verify_syntax(code_str)
        return is_valid

    @staticmethod
    def verify_execution(code_str: str, test_cases: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, str]:
        """Rule 2: Execute in sandbox and test assertions."""
        sandbox_res = CodeSandbox.execute(code_str)
        if not sandbox_res["success"]:
            return False, sandbox_res["stderr"]

        if test_cases:
            for tc in test_cases:
                tc_code = f"{code_str}\n\nresult = {tc.get('call')}\nassert result == {repr(tc.get('expected'))}\n"
                tc_res = CodeSandbox.execute(tc_code)
                if not tc_res["success"]:
                    return False, f"Test failure: {tc_res['stderr']}"

        return True, "Passed all deterministic rule checks"

def best_of_n_scaling(
    generation_fn: Callable[[float], str],
    n_samples: int = 5,
    temperature: float = 0.7,
    test_cases: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, List[str]]:
    valid_candidates = []

    for _ in range(n_samples):
        candidate_code = generation_fn(temperature)
        if DeterministicVerifier.verify_syntax_and_structure(candidate_code):
            is_valid, _ = DeterministicVerifier.verify_execution(candidate_code, test_cases=test_cases)
            if is_valid:
                valid_candidates.append(candidate_code.strip())

    if not valid_candidates:
        fallback = generation_fn(0.2)
        return fallback, [fallback]

    vote_counts = Counter(valid_candidates)
    best_candidate, _ = vote_counts.most_common(1)[0]

    return best_candidate, valid_candidates
