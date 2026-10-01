"""Advanced Local Reasoning Engine & Intent Dispatcher for Agent-Zed CEO."""

from typing import Dict, Any, Tuple
from agent_zed.dataset.knowledge import KnowledgeSynthesizer

class ReasoningEngine:
    """Processes user inputs and determines whether to respond conversationally or dispatch to MoA workforce."""

    def __init__(self):
        self.knowledge = KnowledgeSynthesizer()

    def process_input(self, user_text: str) -> Tuple[str, bool]:
        """
        Returns (response_or_problem_description, is_coding_task).
        If is_coding_task is True, the CLI automatically dispatches to the 13 MoA workforce!
        """
        matched = self.knowledge.match_intent(user_text)
        if matched:
            return matched, False

        # If user asks how something works or asks a coding question/task:
        t_lower = user_text.lower()
        coding_keywords = [
            "code", "solve", "build", "write", "function", "algorithm", "class",
            "debug", "fix", "optimize", "create", "how", "make", "implement",
            "sort", "tree", "dp", "queue", "stack", "parser", "html", "python", "js"
        ]

        if any(kw in t_lower for kw in coding_keywords) or len(user_text.split()) > 3:
            return user_text, True

        # Conversational fallback
        return f"Interesting question regarding '{user_text}'. As CEO, I am dispatching our 13-agent MoA workforce to analyze and build a comprehensive solution for you!", True
