"""Corpus builder for TinyCodeGPT training.

Collects real Python source code from two dependency-free sources:
1. This repository's own source (self-hosted bootstrapping).
2. A curated list of pure-Python standard library modules, which exist on
   every CPython installation (including GitHub Actions runners) - no large
   dataset downloads required.

Standard library code is (c) Python Software Foundation, PSF License; used
here as a small training corpus for a toy-scale model.
"""

import importlib.util
import os
from typing import Dict, List, Optional, Tuple

DEFAULT_STDLIB_MODULES = [
    "abc", "argparse", "ast", "bisect", "calendar", "cmd", "collections",
    "contextlib", "copy", "dataclasses", "decimal", "enum", "fractions",
    "functools", "gettext", "heapq", "inspect", "keyword", "linecache",
    "numbers", "pprint", "queue", "random", "reprlib", "shlex", "statistics",
    "string", "textwrap", "token", "tokenize", "traceback", "types",
    "typing", "warnings", "weakref",
]

MAX_BYTES_PER_FILE = 120_000

def _repo_python_files(repo_root: str) -> List[str]:
    files: List[str] = []
    for base in ("agent_zed", "tests"):
        base_path = os.path.join(repo_root, base)
        for dirpath, dirnames, filenames in os.walk(base_path):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "weights")]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    files.append(os.path.join(dirpath, fn))
    return sorted(files)

def _stdlib_module_path(module_name: str) -> Optional[str]:
    """Resolve a stdlib module to its .py source file without importing it."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return spec.origin

def build_corpus(
    repo_root: Optional[str] = None,
    include_repo: bool = True,
    stdlib_modules: Optional[List[str]] = None,
    max_bytes: int = 1_200_000,
) -> Tuple[str, Dict[str, int]]:
    """Build the training corpus. Returns (text, stats).

    Deterministic: sources are visited in sorted order and the result is
    truncated to ``max_bytes``.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if stdlib_modules is None:
        stdlib_modules = DEFAULT_STDLIB_MODULES

    chunks: List[str] = []
    n_repo_files = 0
    n_stdlib_files = 0

    if include_repo:
        for path in _repo_python_files(repo_root):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()[:MAX_BYTES_PER_FILE]
            except OSError:
                continue
            rel = os.path.relpath(path, repo_root)
            chunks.append(f"\n\n# ==== source: {rel} ====\n\n{content}")
            n_repo_files += 1

    for mod in sorted(set(stdlib_modules)):
        path = _stdlib_module_path(mod)
        if path is None:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()[:MAX_BYTES_PER_FILE]
        except OSError:
            continue
        chunks.append(f"\n\n# ==== source: stdlib/{mod}.py ====\n\n{content}")
        n_stdlib_files += 1

    text = "".join(chunks)[:max_bytes]
    stats = {
        "chars": len(text),
        "repo_files": n_repo_files,
        "stdlib_files": n_stdlib_files,
    }
    return text, stats

if __name__ == "__main__":
    corpus, corpus_stats = build_corpus()
    print("Corpus stats:", corpus_stats)
    print("First 300 chars:\n", corpus[:300])
