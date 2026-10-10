# Agent-Zed (Zed-CLI) 🚀

[![CI](https://github.com/TheStrongestOfTomorrow/Agent-Zed/actions/workflows/ci.yml/badge.svg)](https://github.com/TheStrongestOfTomorrow/Agent-Zed/actions/workflows/ci.yml)
[![Train TinyCodeGPT SLM](https://github.com/TheStrongestOfTomorrow/Agent-Zed/actions/workflows/train-slm.yml/badge.svg)](https://github.com/TheStrongestOfTomorrow/Agent-Zed/actions/workflows/train-slm.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

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

### 📦 Install as a Python Package (`zed-cli` command)
From the repo root:
```bash
pip install -e ".[cli]"
```
This registers the `zed-cli` console command:
```bash
zed-cli              # interactive CEO chat session
zed-cli benchmark    # run the HARD coding benchmark suite
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
| **`/solve <problem>`** | Create a background MoA coding job to synthesize & verify a solution |
| **`/jobs`** | View live workforce jobs dashboard, assigned agents, and progress |
| **`/agents`** | Display 13 agent personas, titles, and corporate hierarchy |
| **`/bench`** | Run the HARD coding benchmark suite in-session |
| **`/model`** | Show TinyCodeGPT SLM status + live pure-Python sample generation |
| **`/help`** | Show available commands |
| **`/quit`** | Exit the CLI session |

---

## 🧪 HARD Coding Benchmark Suite

To test Agent-Zed against complex algorithmic challenges:
```bash
python3 -m agent_zed.cli benchmark   # or: zed-cli benchmark, or /bench inside the CLI
```
Every benchmark result is now labeled with its engine verification status
(`engine-verified` vs `engine-UNVERIFIED`) — the suite re-executes each test
case through the same shared sandbox snippet builder the MoA engine uses
internally, so nothing is verified "by magic".

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

## 🧠 TinyCodeGPT: A Real SLM Trained on GitHub Actions

Agent-Zed ships a **real char-level transformer** (TinyCodeGPT, ~116K params) trained on CPU by the
[`train-slm.yml`](../../actions/workflows/train-slm.yml) GitHub Actions workflow — no GPUs, no API keys.

**How it works:**
- **Corpus** (~1.2MB): Agent-Zed's own source + curated pure-Python stdlib modules — collected on the runner itself, zero dataset downloads.
- **Training**: PyTorch (CPU wheels) trains a 2-layer, d=64, 4-head causal transformer with tied embeddings at char level (~1500 steps ≈ a few minutes on a 2-vCPU runner).
- **Inference**: weights export to a compact JSON checkpoint that runs through a **zero-dependency pure-Python runtime** (`agent_zed/slm/purepy.py`, KV-cached, `math`+`random` only) — so the trained model runs even on Termux/4GB Android, keeping the offline promise.
- **Parity is enforced by tests**: torch ↔ pure-Python logits must match (see `tests/test_slm.py`).

**Try it:**
```bash
zed-cli            # then type: /model
```

**Retrain it:**
```bash
# locally (needs the CPU torch build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m agent_zed.slm.train_github --preset tiny --steps 1500

# or on GitHub Actions:
#   Actions → "Train TinyCodeGPT SLM" → Run workflow
#   (preset tiny/small, custom steps, optional commit-back of refreshed weights)
# A scheduled run also refreshes weights every Sunday 03:00 UTC (artifact only).
```

Each run publishes a training report (loss curve, val perplexity, sample generations) in the job's
**Step Summary**, plus a downloadable weights artifact.

> **Honesty note:** at this scale the model produces *code-shaped* text (learned indentation,
> keywords, identifiers) — it is a genuine trained SLM, not a substitute for the deterministic
> template+verification path that `/solve` uses for *verified* deliverables. The plumbing
> (seeded sampling, checkpoint format, parity tests) is built for the model to grow into.

---

## 📜 License & Community
Agent-Zed is open-source under the **MIT License**. Contributions and feedback are welcome!
