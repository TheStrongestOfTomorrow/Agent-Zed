"""Local Knowledge Base & Dataset Synthesizer for Agent-Zed Reasoning Engine."""

INTENT_KNOWLEDGE_BASE = {
    "greetings": [
        "Hello! I am the CEO of Agent-Zed. I lead our 13-agent MoA engineering team. How can I assist your coding journey today?",
        "Greetings! The 13-agent workforce is ready and operating at peak efficiency under a low-memory budget. What shall we design or solve?"
    ],
    "identity": [
        "I am the CEO of Agent-Zed! I lead a 13-agent Mixture-of-Agents (MoA) engineering team featuring a Researcher, Debugger, Logic Specialist, Code Architect, Security Auditor, and 8 other specialized roles. We work 100% locally to solve coding problems.",
        "Agent-Zed is an autonomous MoA framework where 13 AI persona agents collaborate, debate, run code in sandboxes, and synthesize verified solutions."
    ],
    "capabilities": [
        "Here is what our 13-agent MoA team can do:\n1. Solve complex algorithmic problems (Dynamic Programming, Graph theory, AST Parsers, Concurrency).\n2. Write, debug, and run Python, HTML, CSS, JavaScript code.\n3. Perform web research & document scraping via the Researcher agent.\n4. Sandbox execute code and automatically patch stack trace errors via Debugger & Test Engineer.\n5. Optimize O(N) complexity and memory footprint via Performance Optimizer."
    ],
    "python": [
        "Python is a versatile high-level language. For asynchronous I/O, `asyncio` allows non-blocking concurrency. For performance-critical loops, memory profiling or algorithm state compression (e.g. Bitmask DP) can dramatically increase speed."
    ],
    "html_css_js": [
        "Modern frontend design uses semantic HTML5, responsive CSS flexbox/grid layout systems, and asynchronous JavaScript (ES6+ async/await) for interactive UI rendering."
    ]
}

class KnowledgeSynthesizer:
    """Local dataset and heuristic reasoning matcher."""

    @staticmethod
    def match_intent(text: str) -> str:
        t_lower = text.lower().strip()
        if t_lower in ["hi", "hello", "hey", "greetings", "yo"]:
            return INTENT_KNOWLEDGE_BASE["greetings"][0]
        elif "who are you" in t_lower or "what are you" in t_lower or "what is agent-zed" in t_lower:
            return INTENT_KNOWLEDGE_BASE["identity"][0]
        elif "what can you do" in t_lower or "capabilities" in t_lower or "help" in t_lower:
            return INTENT_KNOWLEDGE_BASE["capabilities"][0]
        elif "python" in t_lower or "async" in t_lower:
            return INTENT_KNOWLEDGE_BASE["python"][0]
        elif "html" in t_lower or "css" in t_lower or "javascript" in t_lower or "js" in t_lower:
            return INTENT_KNOWLEDGE_BASE["html_css_js"][0]
        return None
