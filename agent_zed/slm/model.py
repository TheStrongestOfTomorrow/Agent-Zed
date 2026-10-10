"""TinyCodeGPT Neural Network Architecture & Pure Python SLM Engine for <4GB RAM Budget.

Two operating modes:

1. **Trained mode** - when a weights checkpoint exists (trained by
   `agent_zed.slm.train_github`, locally or on GitHub Actions), the model runs
   REAL autoregressive char-level generation through the zero-dependency
   `purepy` runtime.
2. **Simulation mode** (legacy fallback) - when no checkpoint is present,
   `generate_tokens()` falls back to the original deterministic token
   simulation so the API keeps working everywhere.

The `weights/` checkpoint ships with the repo and is refreshed by the
`train-slm.yml` GitHub Actions workflow.
"""

import os
from typing import Any, Dict, List, Optional

DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "tinycodegpt_char.json")

class TinyCodeGPT:
    """Small Language Model (SLM): char-level GPT with a pure-Python inference runtime."""

    def __init__(
        self,
        vocab_size: int = 16000,
        max_len: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        weights: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.weights = weights
        self.tokenizer = tokenizer
        self.meta = meta or {}
        self._runtime = None

        if weights is not None:
            # real checkpoint: report true architecture + parameter count
            self.vocab_size = int(self.meta.get("vocab_size", vocab_size))
            self.d_model = int(self.meta.get("d_model", d_model))
            self.n_layers = int(self.meta.get("n_layers", n_layers))
            self.n_heads = int(self.meta.get("n_heads", n_heads))
            self.max_len = int(self.meta.get("ctx", max_len))
            self.param_count = int(self.meta.get("param_count") or 0)
            if not self.param_count:
                from agent_zed.slm import purepy
                self.param_count = purepy.count_params(weights)
        else:
            self.param_count = vocab_size * d_model + max_len * d_model + n_layers * (4 * d_model * d_model)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path: str = DEFAULT_WEIGHTS_PATH) -> "TinyCodeGPT":
        """Load a JSON checkpoint exported by the trainer."""
        from agent_zed.slm import purepy
        from agent_zed.slm.tokenizer import CharTokenizer
        weights, meta, tok_dict = purepy.load_checkpoint(path)
        return cls(weights=weights, tokenizer=CharTokenizer.from_dict(tok_dict), meta=meta)

    @classmethod
    def try_load_default(cls) -> Optional["TinyCodeGPT"]:
        """Load the shipped checkpoint if present; otherwise return None."""
        if os.path.exists(DEFAULT_WEIGHTS_PATH):
            try:
                return cls.from_pretrained(DEFAULT_WEIGHTS_PATH)
            except Exception:
                return None
        return None

    @property
    def is_trained(self) -> bool:
        return self.weights is not None and int(self.meta.get("trained_steps", 0)) > 0

    def status(self) -> Dict[str, Any]:
        """Human/CLI-readable status of this model instance."""
        if self.weights is None:
            return {
                "mode": "simulation",
                "trained": False,
                "param_count": self.param_count,
                "detail": "no checkpoint found - generate_tokens() uses the legacy deterministic simulation",
            }
        info = {
            "mode": "trained" if self.is_trained else "untrained-weights",
            "trained": self.is_trained,
            "param_count": self.param_count,
            "arch": self.meta.get("arch", "tiny-gpt-char"),
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ctx": self.max_len,
            "vocab_size": self.vocab_size,
            "trained_steps": self.meta.get("trained_steps", 0),
            "final_val_loss": self.meta.get("final_val_loss"),
            "final_perplexity": self.meta.get("final_perplexity"),
            "corpus_chars": self.meta.get("corpus_chars"),
            "trained_on": self.meta.get("trained_on"),
            "created_utc": self.meta.get("created_utc"),
        }
        return info

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def _get_runtime(self):
        if self._runtime is None:
            from agent_zed.slm.purepy import PurePyGPT
            self._runtime = PurePyGPT(self.weights, self.meta)
        return self._runtime

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 8,
        seed: Optional[int] = 0,
    ) -> str:
        """REAL autoregressive generation via the pure-Python runtime.

        Requires trained weights (from_pretrained / try_load_default).
        Returns only the newly generated text (prompt excluded).
        """
        if self.weights is None or self.tokenizer is None:
            raise RuntimeError(
                "generate_text() requires real weights. Train one with:\n"
                "  python -m agent_zed.slm.train_github --preset tiny --steps 1500\n"
                "or run the 'Train TinyCodeGPT SLM' GitHub Actions workflow."
            )
        runtime = self._get_runtime()
        prompt_ids = self.tokenizer.encode(prompt)
        new_ids = runtime.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        return self.tokenizer.decode(new_ids)

    def generate_tokens(self, prompt_tokens: List[int], max_new_tokens: int = 128, temperature: float = 0.7) -> List[int]:
        """Legacy token-level API.

        With real weights loaded this performs true sampled generation;
        without weights it falls back to the original deterministic
        simulation (kept for backward compatibility).
        """
        if self.weights is not None:
            runtime = self._get_runtime()
            new_ids = runtime.generate(prompt_tokens, max_new_tokens=max_new_tokens,
                                       temperature=temperature, seed=0)
            return list(prompt_tokens) + new_ids
        out = list(prompt_tokens)
        for _ in range(max_new_tokens):
            next_token = (out[-1] * 31 + 7) % self.vocab_size
            out.append(next_token)
        return out
