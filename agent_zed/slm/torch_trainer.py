"""PyTorch trainer for TinyCodeGPT - a training-time-only dependency.

The architecture here MUST stay in lockstep with `agent_zed.slm.purepy`
(same layer shapes, same order, bias-free linears, erf-GELU, tied embedding
head, LayerNorm eps 1e-5), so exported weights are bit-for-bit compatible
with the zero-dependency runtime. Parity is enforced by tests.

Designed for CPU-only environments, including GitHub Actions runners.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime path must survive without torch
    TORCH_AVAILABLE = False

from agent_zed.slm.tokenizer import CharTokenizer

def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for training. Install the CPU build with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

class TinyGPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.Wqkv = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Parameter(torch.empty(4 * d_model, d_model))
        self.proj = nn.Parameter(torch.empty(d_model, 4 * d_model))
        for p in (self.Wqkv, self.Wo, self.fc, self.proj):
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        h = self.ln1(x)
        qkv = F.linear(h, self.Wqkv)  # (B, T, 3d); row order [q; k; v]
        q, k, v = torch.split(qkv, d, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (qh @ kh.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ vh).transpose(1, 2).contiguous().view(B, T, d)
        x = x + F.linear(out, self.Wo)

        h2 = self.ln2(x)
        f = F.gelu(F.linear(h2, self.fc))  # erf-based GELU, matches purepy
        x = x + F.linear(f, self.proj)
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, ctx: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(ctx, d_model)
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(TinyGPTBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = F.linear(self.ln_f(x), self.wte.weight)  # tied head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: List[int], max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 8, seed: Optional[int] = 0) -> List[int]:
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        idx = torch.tensor([ids[-self.ctx:]], dtype=torch.long)
        out = list(ids)
        for _ in range(max_new_tokens):
            if idx.shape[1] >= self.ctx:
                break
            logits, _ = self.forward(idx)
            next_logits = logits[0, -1]
            if temperature <= 0:
                nxt = int(next_logits.argmax())
            else:
                k = max(1, min(top_k, next_logits.numel()))
                vals, idxs = torch.topk(next_logits, k)
                probs = F.softmax(vals / temperature, dim=-1)
                choice = int(torch.multinomial(probs, 1, generator=gen).item())
                nxt = int(idxs[choice])
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        return out[len(ids):]

def export_weights(model: TinyGPT) -> Dict[str, Any]:
    """Export to the purepy-compatible nested-list weight format."""
    weights: Dict[str, Any] = {
        "wte": model.wte.weight.detach().tolist(),
        "wpe": model.wpe.weight.detach().tolist(),
        "blocks": [],
        "ln_f": {
            "w": model.ln_f.weight.detach().tolist(),
            "b": model.ln_f.bias.detach().tolist(),
        },
    }
    for blk in model.blocks:
        weights["blocks"].append({
            "ln1_w": blk.ln1.weight.detach().tolist(),
            "ln1_b": blk.ln1.bias.detach().tolist(),
            "Wqkv": blk.Wqkv.detach().tolist(),
            "Wo": blk.Wo.detach().tolist(),
            "ln2_w": blk.ln2.weight.detach().tolist(),
            "ln2_b": blk.ln2.bias.detach().tolist(),
            "fc": blk.fc.detach().tolist(),
            "proj": blk.proj.detach().tolist(),
        })
    return weights

def train_tiny_gpt(
    corpus_text: str,
    steps: int = 1500,
    batch_size: int = 24,
    seq_len: int = 96,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    lr: float = 5e-4,
    seed: int = 1337,
    eval_interval: int = 250,
    val_fraction: float = 0.1,
    log_every: int = 50,
    on_log: Optional[Any] = None,
) -> Tuple[TinyGPT, CharTokenizer, Dict[str, Any]]:
    """Train TinyGPT on char-level corpus text. CPU-friendly.

    Returns (model, tokenizer, history) where history contains loss curve,
    final val loss/perplexity, and config metadata.
    """
    _require_torch()
    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = CharTokenizer.build(corpus_text)
    data = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    n_val = max(1, int(len(data) * val_fraction))
    train_data, val_data = data[:-n_val], data[-n_val:]

    model = TinyGPT(tokenizer.vocab_size, ctx=seq_len, d_model=d_model,
                    n_heads=n_heads, n_layers=n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(split: str):
        d = train_data if split == "train" else val_data
        max_start = len(d) - seq_len - 1
        if max_start <= 0:
            raise ValueError("corpus too small for the requested seq_len")
        ix = torch.randint(0, max_start, (batch_size,), generator=torch.Generator().manual_seed(random.randrange(2**31)))
        x = torch.stack([d[i:i + seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + seq_len] for i in ix])
        return x, y

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = []
        for _ in range(8):
            x, y = get_batch("val")
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    history: Dict[str, Any] = {
        "config": {
            "steps": steps, "batch_size": batch_size, "seq_len": seq_len,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "lr": lr, "seed": seed,
        },
        "vocab_size": tokenizer.vocab_size,
        "corpus_chars": len(corpus_text),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
        "loss_curve": [],
        "val_curve": [],
        "samples": [],
        "param_count": sum(p.numel() for p in model.parameters()),
    }

    warmup = max(1, int(steps * 0.05))
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        # linear warmup + cosine decay
        if step <= warmup:
            cur_lr = lr * step / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * prog)) * 0.9 + lr * 0.1
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running = loss.item() if running == 0.0 else 0.9 * running + 0.1 * loss.item()
        if step % log_every == 0 or step == steps:
            history["loss_curve"].append({"step": step, "loss": round(running, 4), "lr": round(cur_lr, 6)})
            if on_log:
                on_log(step, steps, running, cur_lr)

        if step % eval_interval == 0 or step == steps:
            val_loss = evaluate()
            perplexity = math.exp(min(20.0, val_loss))
            new_tokens = model.generate(tokenizer.encode("\n\ndef "), max_new_tokens=48,
                                        temperature=0.8, top_k=8, seed=seed + step)
            sample = tokenizer.decode(new_tokens)
            history["val_curve"].append({"step": step, "val_loss": round(val_loss, 4), "perplexity": round(perplexity, 2)})
            history["samples"].append({"step": step, "sample": sample})
            if on_log:
                on_log(step, steps, running, cur_lr, val_loss=val_loss, sample=sample)

    history["elapsed_seconds"] = round(time.time() - t0, 1)
    history["final_val_loss"] = history["val_curve"][-1]["val_loss"] if history["val_curve"] else None
    history["final_perplexity"] = history["val_curve"][-1]["perplexity"] if history["val_curve"] else None
    return model, tokenizer, history
