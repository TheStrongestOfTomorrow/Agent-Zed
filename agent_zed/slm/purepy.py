"""Dependency-free inference runtime for TinyCodeGPT.

Implements the exact same architecture as the PyTorch trainer
(`agent_zed.slm.torch_trainer`) using only the standard library
(`math` + `random`), so trained weights can run anywhere Agent-Zed runs -
including Termux on a 4GB Android phone - with zero pip dependencies.

Architecture (must stay in lockstep with torch_trainer.TinyGPT):
    x = wte[token] + wpe[pos]
    per block:
        h   = LayerNorm(x, ln1_w, ln1_b)                # eps 1e-5
        qkv = h @ Wqkv^T                                # Wqkv rows: [q; k; v]
        causal multi-head attention, scale 1/sqrt(head_dim)
        x   = x + attn @ Wo^T
        h2  = LayerNorm(x, ln2_w, ln2_b)
        f   = GELU_erf(h2 @ fc^T)                       # fc: (4d, d)
        x   = x + f @ proj^T                            # proj: (d, 4d)
    logits = LayerNorm(x, lnf_w, lnf_b) @ wte^T         # tied embedding head

All linear layers are bias-free; only LayerNorms carry weight+bias.
"""

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

LN_EPS = 1e-5

def _mat_vec(w: List[List[float]], x: List[float]) -> List[float]:
    """y = W @ x, where W is stored as a list of output-dimension rows."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def _layer_norm(x: List[float], w: List[float], b: List[float]) -> List[float]:
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + LN_EPS)
    return [(v - mean) * inv * wi + bi for v, wi, bi in zip(x, w, b)]

def _softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(v - m) for v in xs]
    s = sum(exps)
    return [e / s for e in exps]

def _gelu(x: float) -> float:
    """Exact (erf-based) GELU - matches torch.nn.functional.gelu default."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

class PurePyGPT:
    """Pure-Python forward pass + KV-cached autoregressive generation."""

    def __init__(self, weights: Dict[str, Any], meta: Dict[str, Any]):
        self.w = weights
        self.meta = meta
        self.d: int = int(meta["d_model"])
        self.n_layers: int = int(meta["n_layers"])
        self.n_heads: int = int(meta["n_heads"])
        self.ctx: int = int(meta["ctx"])
        self.vocab_size: int = int(meta["vocab_size"])
        if self.d % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d // self.n_heads
        self.wte: List[List[float]] = weights["wte"]      # (V, d)
        self.wpe: List[List[float]] = weights["wpe"]      # (ctx, d)
        self.blocks: List[Dict[str, Any]] = weights["blocks"]
        self.lnf_w: List[float] = weights["ln_f"]["w"]
        self.lnf_b: List[float] = weights["ln_f"]["b"]
        if len(self.blocks) != self.n_layers:
            raise ValueError("weights/meta disagree on n_layers")

    # ------------------------------------------------------------------
    # core step: one token through the stack, using per-layer KV caches
    # ------------------------------------------------------------------
    def _step(
        self,
        token: int,
        pos: int,
        cache_k: List[List[List[float]]],
        cache_v: List[List[List[float]]],
    ) -> List[float]:
        d = self.d
        pos = min(pos, self.ctx - 1)
        wte_row = self.wte[token]
        wpe_row = self.wpe[pos]
        x = [t + p for t, p in zip(wte_row, wpe_row)]

        scale = 1.0 / math.sqrt(self.head_dim)

        for li, blk in enumerate(self.blocks):
            h = _layer_norm(x, blk["ln1_w"], blk["ln1_b"])
            qkv = _mat_vec(blk["Wqkv"], h)
            q = qkv[0:d]
            k = qkv[d:2 * d]
            v = qkv[2 * d:3 * d]
            cache_k[li].append(k)
            cache_v[li].append(v)

            attn = [0.0] * d
            for hi in range(self.n_heads):
                s = hi * self.head_dim
                e = s + self.head_dim
                qh = q[s:e]
                scores = [
                    sum(a * b for a, b in zip(qh, kj[s:e])) * scale
                    for kj in cache_k[li]
                ]
                probs = _softmax(scores)
                for j, p in enumerate(probs):
                    vj = cache_v[li][j]
                    for ii in range(s, e):
                        attn[ii] += p * vj[ii]

            ao = _mat_vec(blk["Wo"], attn)
            x = [xi + ai for xi, ai in zip(x, ao)]

            h2 = _layer_norm(x, blk["ln2_w"], blk["ln2_b"])
            f = [_gelu(v) for v in _mat_vec(blk["fc"], h2)]
            pr = _mat_vec(blk["proj"], f)
            x = [xi + pi for xi, pi in zip(x, pr)]

        xf = _layer_norm(x, self.lnf_w, self.lnf_b)
        return _mat_vec(self.wte, xf)  # tied head: logits = wte @ xf

    def _new_cache(self) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
        return [[] for _ in range(self.n_layers)], [[] for _ in range(self.n_layers)]

    # ------------------------------------------------------------------
    # full forward (no cache) - used for tests / logits inspection
    # ------------------------------------------------------------------
    def forward_full(self, tokens: List[int]) -> List[List[float]]:
        """Return logits for every position (list of T vectors of size V)."""
        cache_k, cache_v = self._new_cache()
        all_logits = []
        for pos, t in enumerate(tokens[: self.ctx]):
            all_logits.append(self._step(t, pos, cache_k, cache_v))
        return all_logits

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _argmax(logits: List[float]) -> int:
        best_i = 0
        best_v = logits[0]
        for i in range(1, len(logits)):
            if logits[i] > best_v:
                best_v = logits[i]
                best_i = i
        return best_i

    def _sample(self, logits: List[float], temperature: float, top_k: int, rng: random.Random) -> int:
        if temperature <= 0:
            return self._argmax(logits)
        k = max(1, min(top_k, len(logits)))
        order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]
        scaled = [logits[i] / temperature for i in order]
        probs = _softmax(scaled)
        r = rng.random()
        acc = 0.0
        for idx, p in zip(order, probs):
            acc += p
            if r <= acc:
                return idx
        return order[-1]

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 8,
        seed: Optional[int] = 0,
    ) -> List[int]:
        """Autoregressive generation with a KV cache. Returns NEW token ids."""
        rng = random.Random(seed)
        ids = list(prompt_ids)
        if len(ids) > self.ctx:
            ids = ids[-self.ctx:]
        if not ids:
            ids = [0]

        cache_k, cache_v = self._new_cache()
        logits: List[float] = []
        for pos, t in enumerate(ids):
            logits = self._step(t, pos, cache_k, cache_v)

        generated: List[int] = []
        total_len = len(ids)
        while len(generated) < max_new_tokens and total_len < self.ctx:
            nxt = self._sample(logits, temperature, top_k, rng)
            generated.append(nxt)
            logits = self._step(nxt, total_len, cache_k, cache_v)
            total_len += 1
        return generated

# ----------------------------------------------------------------------
# weight construction / serialization helpers (pure python, no torch)
# ----------------------------------------------------------------------
def random_weights(
    vocab_size: int,
    ctx: int = 96,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Create randomly-initialized weights (std 0.02, like the torch trainer)."""
    rng = random.Random(seed)

    def mat(rows: int, cols: int) -> List[List[float]]:
        return [[rng.gauss(0.0, 0.02) for _ in range(cols)] for _ in range(rows)]

    blocks = []
    for _ in range(n_layers):
        blocks.append({
            "ln1_w": [1.0] * d_model,
            "ln1_b": [0.0] * d_model,
            "Wqkv": mat(3 * d_model, d_model),
            "Wo": mat(d_model, d_model),
            "ln2_w": [1.0] * d_model,
            "ln2_b": [0.0] * d_model,
            "fc": mat(4 * d_model, d_model),
            "proj": mat(d_model, 4 * d_model),
        })

    return {
        "wte": mat(vocab_size, d_model),
        "wpe": mat(ctx, d_model),
        "blocks": blocks,
        "ln_f": {"w": [1.0] * d_model, "b": [0.0] * d_model},
    }

def count_params(weights: Dict[str, Any]) -> int:
    def numel(obj: Any) -> int:
        if isinstance(obj, list):
            if not obj:
                return 0
            if isinstance(obj[0], list):
                return sum(numel(r) for r in obj)
            return len(obj)
        return 0

    total = numel(weights["wte"]) + numel(weights["wpe"]) + numel(weights["ln_f"]["w"]) + numel(weights["ln_f"]["b"])
    for blk in weights["blocks"]:
        for key in ("ln1_w", "ln1_b", "Wqkv", "Wo", "ln2_w", "ln2_b", "fc", "proj"):
            total += numel(blk[key])
    return total

def _round_nested(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, list):
        return [_round_nested(v, ndigits) for v in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj

def save_checkpoint(
    path: str,
    weights: Dict[str, Any],
    meta: Dict[str, Any],
    tokenizer_dict: Dict[str, Any],
) -> None:
    """Serialize weights+meta+tokenizer to a compact JSON checkpoint."""
    payload = {
        "format": "agent-zed-tinygpt-char-v1",
        "meta": meta,
        "tokenizer": tokenizer_dict,
        "weights": _round_nested(weights),
    }
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

def load_checkpoint(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load (weights, meta, tokenizer_dict) from a JSON checkpoint."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "agent-zed-tinygpt-char-v1":
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}")
    return payload["weights"], payload["meta"], payload["tokenizer"]
