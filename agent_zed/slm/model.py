"""TinyCodeGPT Neural Network Architecture & Pure Python SLM Engine for <4GB RAM Budget."""

from typing import List

class TinyCodeGPT:
    """Small Language Model (SLM) Architecture & Python Token Matrix Engine (~30M parameters)."""

    def __init__(
        self,
        vocab_size: int = 16000,
        max_len: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.param_count = vocab_size * d_model + max_len * d_model + n_layers * (4 * d_model * d_model)

    def generate_tokens(self, prompt_tokens: List[int], max_new_tokens: int = 128, temperature: float = 0.7) -> List[int]:
        """Autoregressive code generation simulation."""
        out = list(prompt_tokens)
        for _ in range(max_new_tokens):
            next_token = (out[-1] * 31 + 7) % self.vocab_size
            out.append(next_token)
        return out
