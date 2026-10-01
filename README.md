# Agent-Zed (Zed-CLI) 🚀

> **A Lightweight, Powerful 13-Agent Mixture-of-Agents (MoA) Coding AI System**
> *Designed to run 100% locally and offline on any machine—including Android phones with 4GB RAM (Termux) with under 25MB RAM footprint! Zero API keys required.*

---

## ⚡ Quick Setup & Installation Guide

Agent-Zed runs on **Python 3.8+** with **zero external API keys or heavy GPU requirements**.

### 📱 Android (Termux) Setup (4GB RAM Compatible)
1. Install [Termux](https://f-droid.org/en/packages/com.termux/).
2. Open Termux and run:
   ```bash
   pkg update && pkg upgrade -y
   pkg install python git -y
   pip install rich
   git clone https://github.com/TheStrongestOfTomorrow/Agent-Zed.git
   cd Agent-Zed
   ```
3. Launch Agent-Zed:
   ```bash
   python3 -m agent_zed.cli
   ```

### 💻 Linux / macOS / Windows Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/TheStrongestOfTomorrow/Agent-Zed.git
   cd Agent-Zed
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install rich
   ```
4. Start the interactive CLI:
   ```bash
   python3 -m agent_zed.cli
   ```

---

## 💡 Frequently Asked Questions (FAQ)

### Can I actually talk to the CEO without an API key?
**YES!** Agent-Zed operates **100% offline out of the box**. You do not need OpenAI, Together AI, Anthropic, or any paid API key. When you launch `python3 -m agent_zed.cli`, you enter a direct, real-time interactive terminal session with the **CEO Agent**.

### How does the 13-Agent Mixture-of-Agents (MoA) work?
While you chat with the CEO in the foreground, the CEO manages a background workforce of 12 specialized agents:
- **Researcher**: Equipped with live web search and document parsing.
- **Debugger**: Executes code in an AST sandbox runtime to capture stack trace errors.
- **Logic Specialist**: Validates math invariants and algorithm edge cases.
- **Code Architect**: Designs clean modular OOP/functional structures.
- **Performance Optimizer**: Analyzes time/space complexity and memory allocation.
- **Security Auditor**: Checks code for unsafe functions and vulnerabilities.
- **Test Engineer**: Writes dynamic unit assertions to verify code correctness.
- **Refactoring Specialist**: Cleans duplicate logic and refactors syntax.
- **Code Reviewer**: Enforces style and syntax rules.
- **UI/UX Designer**: Formats rich terminal visuals and layouts.
- **Documentation Lead**: Writes clean docstrings and complexity breakdowns.
- **Spokesperson**: Synthesizes all agent findings into a final response.

---

## 💼 `zed-cli` Interactive Commands & Jobs System

Inside `zed-cli`, you can manage workforce jobs and interact with the CEO:

| Command | Action |
| :--- | :--- |
| **`/jobs`** | View live workforce jobs dashboard, assigned agents, and progress |
| **`/agents`** | Display 13 agent personas, titles, and corporate hierarchy |
| **`/solve <problem>`** | Create a background MoA coding job to synthesize & verify a solution |
| **`/quit`** | Exit the CLI session |

---

## 🧪 HARD Coding Benchmark Suite

To test Agent-Zed against complex algorithmic challenges:
```bash
python3 -m agent_zed.cli benchmark
```
Benchmark problems evaluated:
1. **Bitmask Dynamic Programming**: Traveling Salesperson Problem (TSP) shortest Hamiltonian cycle.
2. **High Concurrency Threading**: Thread-safe bounded `ConcurrentQueue` data structure.
3. **AST Parser & Compiler Logic**: NodeVisitor expression evaluator parsing math ASTs.

---

## 🧪 Running Unit Tests
```bash
python3 -m unittest discover -s tests
```

---

## 📜 License & Community
Agent-Zed is open-source under the **MIT License**. Contributions and feedback are welcome!
