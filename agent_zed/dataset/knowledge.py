"""Comprehensive Knowledge Base & Software Engineering Dataset for Agent-Zed."""

INTENT_KNOWLEDGE_BASE = {
    "greetings": [
        "Greetings! I am the CEO of Agent-Zed. I direct our 13-agent MoA engineering team. How can we innovate or build today?",
        "Hello! Our 13-agent workforce is operational with Tree-of-Thought search, AST sandboxing, and web intelligence. What coding challenge shall we solve?"
    ],
    "identity": [
        "I am the CEO of Agent-Zed! I lead our 13-agent Mixture-of-Agents (MoA) engineering company featuring a Researcher, Debugger, Logic Specialist, Code Architect, Security Auditor, and 8 other specialized roles. We solve complex software engineering tasks 100% locally."
    ],
    "capabilities": [
        "Agent-Zed Capabilities:\n1. Tree-of-Thought (ToT) & Beam Search algorithmic reasoning.\n2. Dynamic Programming, Graph Optimization, High-Concurrency structures, and AST Compiler evaluators.\n3. Web research & documentation scraping via Researcher agent.\n4. Sandbox AST code execution & error auto-patching via Debugger & Test Engineer.\n5. O(N) Complexity and Memory profiling via Performance Optimizer.\n6. Interactive CEO session & live workforce jobs management."
    ],
    "python": [
        "Python Software Engineering Principles:\n- Asynchronous I/O via `asyncio` for non-blocking network/concurrency.\n- Type hints (`typing` module) for compile-time safety and IDE introspection.\n- Abstract Syntax Tree (`ast` module) for code parsing and metaprogramming.\n- High-performance memory management using state bitmasking and memoization."
    ],
    "html_css_js": [
        "Modern Full-Stack Web Development:\n- HTML5: Semantic elements (`<main>`, `<article>`, `<section>`), accessibility ARIA tags.\n- CSS3: Flexbox, CSS Grid layouts, CSS variables, keyframe animations, responsive design.\n- JavaScript (ES6+): Promises, `async/await`, Event Loop, DOM manipulation, Web Components."
    ],
    "architecture": [
        "Software Architecture Patterns:\n- Clean Architecture / Microservices: Decoupled domain models, interfaces, dependency injection.\n- Mixture-of-Agents (MoA): Multi-agent consensus layers with specialized persona evaluation.\n- Tree-of-Thought Search: Exploring multiple solution branches and selecting optimal heuristic paths."
    ]
}

class KnowledgeSynthesizer:
    """Knowledge graph matcher and query analyzer."""

    @staticmethod
    def match_intent(text: str) -> str:
        t_lower = text.lower().strip()
        if t_lower in ["hi", "hello", "hey", "greetings", "yo"]:
            return INTENT_KNOWLEDGE_BASE["greetings"][0]
        elif "who are you" in t_lower or "what are you" in t_lower or "agent-zed" in t_lower:
            return INTENT_KNOWLEDGE_BASE["identity"][0]
        elif "what can you do" in t_lower or "capabilities" in t_lower or "help" in t_lower:
            return INTENT_KNOWLEDGE_BASE["capabilities"][0]
        elif "python" in t_lower or "async" in t_lower or "ast" in t_lower:
            return INTENT_KNOWLEDGE_BASE["python"][0]
        elif "html" in t_lower or "css" in t_lower or "javascript" in t_lower or "js" in t_lower:
            return INTENT_KNOWLEDGE_BASE["html_css_js"][0]
        elif "architecture" in t_lower or "pattern" in t_lower or "moa" in t_lower:
            return INTENT_KNOWLEDGE_BASE["architecture"][0]
        return None
