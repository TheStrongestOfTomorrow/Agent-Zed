"""Inference-Time Compute Scaling Pipeline for Agent-Zed."""

import inspect
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

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
                tc_res = CodeSandbox.execute(CodeSandbox.build_test_snippet(code_str, tc))
                if not tc_res["success"]:
                    return False, f"Test failure: {tc_res['stderr']}"

        return True, "Passed all deterministic rule checks"

def _supports_sample_index(generation_fn: Callable) -> bool:
    """Detect whether generation_fn accepts (temperature, sample_index)."""
    try:
        params = inspect.signature(generation_fn).parameters
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind == p.VAR_POSITIONAL for p in params.values())
    return has_var_positional or len(positional) >= 2

def best_of_n_scaling(
    generation_fn: Callable,
    n_samples: int = 5,
    temperature: float = 0.7,
    test_cases: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, List[str]]:
    """Best-of-N inference-time compute scaling with majority voting.

    ``generation_fn`` may accept either ``(temperature)`` or
    ``(temperature, sample_index)``. When it accepts the sample index, each
    sample gets a distinct deterministic seed so candidates can genuinely
    differ — otherwise majority voting over N identical copies is meaningless.

    Candidates are filtered by the DeterministicVerifier (syntax + sandboxed
    execution + test cases); the most common surviving candidate wins the vote.
    """
    two_arg = _supports_sample_index(generation_fn)
    valid_candidates: List[str] = []

    def _generate(temp: float, idx: int) -> str:
        return generation_fn(temp, idx) if two_arg else generation_fn(temp)

    for i in range(n_samples):
        candidate_code = _generate(temperature, i)
        if DeterministicVerifier.verify_syntax_and_structure(candidate_code):
            is_valid, _ = DeterministicVerifier.verify_execution(candidate_code, test_cases=test_cases)
            if is_valid:
                valid_candidates.append(candidate_code.strip())

    if not valid_candidates:
        fallback = _generate(0.2, n_samples).strip()
        return fallback, [fallback]

    vote_counts = Counter(valid_candidates)
    best_candidate, _ = vote_counts.most_common(1)[0]

    return best_candidate, valid_candidates
