# Agent-Zed
A AI Which is Fully trained on Code, Lightweight, Plus Has agentic cli to work with and A MOA Model (mixture of agents ) basically the same mdel but different persnas arugign and doiing things.

## Agent-Zed Architecture & Usage

### Key Features
- **13 Corporate Personas**: CEO (Lead), Researcher (Web Search), Debugger (Sandbox), Logic Specialist, Code Architect, Performance Optimizer, Security Auditor, Test Engineer, Refactoring Specialist, Code Reviewer, UI/UX Designer, Documentation Lead, Spokesperson.
- **zed-cli Jobs System**: CEO manages background workforce jobs, live progress, and status reports.
- **Android / Low-RAM Optimized**: Peak RAM usage < 25MB (ideal for 4GB RAM Android Termux devices).
- **HARD Benchmark Suite**: Passes 100% of Bitmask DP, Thread Concurrency, and AST Compiler evaluation tests.

### How to Run

1. **Start Interactive CEO CLI**:
   ```bash
   python3 -m agent_zed.cli
   ```

2. **Run HARD Coding Benchmark Suite**:
   ```bash
   python3 -m agent_zed.cli benchmark
   ```

3. **Run Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests
   ```
