"""Train TinyCodeGPT on CPU - designed for GitHub Actions runners.

Usage (local or CI):
    python -m agent_zed.slm.train_github --preset tiny --steps 1500
    python -m agent_zed.slm.train_github --self-check

Writes a pure-Python-loadable JSON checkpoint (no torch needed at inference),
prints a report, and appends a summary to $GITHUB_STEP_SUMMARY when running
inside GitHub Actions.
"""

import argparse
import datetime
import os
import platform
import sys
from typing import Any, Dict

from agent_zed.slm import purepy
from agent_zed.slm.corpus import build_corpus
from agent_zed.slm.tokenizer import CharTokenizer

DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "tinycodegpt_char.json")

PRESETS: Dict[str, Dict[str, Any]] = {
    # ~115K params - trains in minutes on a 2-vCPU GitHub Actions runner
    "tiny":  {"d_model": 64,  "n_layers": 2, "n_heads": 4, "seq_len": 96,  "batch_size": 24},
    # ~620K params - still CPU-feasible, better samples, longer runs
    "small": {"d_model": 128, "n_layers": 3, "n_heads": 4, "seq_len": 128, "batch_size": 32},
}

def _in_github_actions() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS"))

def _write_step_summary(report: Dict[str, Any], samples_text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    cfg = report["config"]
    lines = [
        "## 🧠 TinyCodeGPT SLM Training Report",
        "",
        f"- **Preset**: `{report.get('preset')}` | **Steps**: {cfg['steps']} | **Params**: {report['param_count']:,}",
        f"- **Corpus**: {report['corpus_chars']:,} chars ({report['corpus_stats']['repo_files']} repo files + {report['corpus_stats']['stdlib_files']} stdlib files)",
        f"- **Vocab**: {report['vocab_size']} chars | **Seq len**: {cfg['seq_len']} | **Batch**: {cfg['batch_size']}",
        f"- **Final train loss**: {report['loss_curve'][-1]['loss'] if report['loss_curve'] else 'n/a'}",
        f"- **Final val loss**: {report.get('final_val_loss')} | **Perplexity**: {report.get('final_perplexity')}",
        f"- **Wall time**: {report.get('elapsed_seconds')}s on {report.get('trained_on')}",
        "",
        "### Loss curve (train)",
        "",
        "| Step | Loss | LR |",
        "| --- | --- | --- |",
    ]
    for point in report.get("loss_curve", []):
        lines.append(f"| {point['step']} | {point['loss']} | {point['lr']} |")
    lines += ["", "### Val loss / perplexity", "", "| Step | Val loss | Perplexity |", "| --- | --- | --- |"]
    for point in report.get("val_curve", []):
        lines.append(f"| {point['step']} | {point['val_loss']} | {point['perplexity']} |")
    lines += ["", "### Sample generations", "", "```python", samples_text.strip(), "```", ""]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

def self_check(weights_path: str = DEFAULT_WEIGHTS_PATH) -> int:
    """Load a checkpoint (or random weights) and run pure-Python generation."""
    if os.path.exists(weights_path):
        weights, meta, tok_dict = purepy.load_checkpoint(weights_path)
        tokenizer = CharTokenizer.from_dict(tok_dict)
        source = weights_path
    else:
        print(f"[self-check] no checkpoint at {weights_path}; using random weights")
        tokenizer = CharTokenizer.build("abcdefghijklmnopqrstuvwxyz0123456789_ \n()[]{}:,.=+-*/<>#!\"'")
        meta = {"d_model": 32, "n_layers": 1, "n_heads": 2, "ctx": 32,
                "vocab_size": tokenizer.vocab_size, "trained_steps": 0}
        weights = purepy.random_weights(tokenizer.vocab_size, ctx=32, d_model=32,
                                        n_heads=2, n_layers=1, seed=7)
        source = "random"
    model = purepy.PurePyGPT(weights, meta)
    new_ids = model.generate(tokenizer.encode("def "), max_new_tokens=8, temperature=0.0, seed=0)
    text = tokenizer.decode(new_ids)
    print(f"[self-check] OK: loaded {source}, {purepy.count_params(weights):,} params, "
          f"pure-Python greedy sample: {text!r}")
    return 0

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train TinyCodeGPT (CPU) and export pure-Python weights")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--max-corpus-bytes", type=int, default=1_200_000)
    parser.add_argument("--out", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--self-check", action="store_true", help="verify pure-Python inference on a checkpoint and exit")
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(args.out)

    from agent_zed.slm.torch_trainer import TORCH_AVAILABLE, export_weights, train_tiny_gpt
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed. On CI runners use:\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
        return 2

    cfg = dict(PRESETS[args.preset])
    print(f"=== TinyCodeGPT training | preset={args.preset} steps={args.steps} cfg={cfg} ===")

    corpus_text, corpus_stats = build_corpus(max_bytes=args.max_corpus_bytes)
    print(f"Corpus: {corpus_stats['chars']:,} chars "
          f"({corpus_stats['repo_files']} repo files, {corpus_stats['stdlib_files']} stdlib files)")

    def on_log(step, steps, loss, lr, val_loss=None, sample=None):
        if val_loss is not None:
            print(f"  step {step:>5}/{steps} | train {loss:.4f} | val {val_loss:.4f} | ppl {__import__('math').exp(min(20, val_loss)):.2f} | lr {lr:.2e}")
            print(f"  sample: {sample[:120]!r}")
        else:
            print(f"  step {step:>5}/{steps} | train {loss:.4f} | lr {lr:.2e}")

    model, tokenizer, history = train_tiny_gpt(
        corpus_text,
        steps=args.steps,
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        lr=args.lr,
        seed=args.seed,
        eval_interval=args.eval_interval,
        on_log=on_log,
    )

    trained_on = "github-actions" if _in_github_actions() else f"local:{platform.node()}"
    meta = {
        "arch": "tiny-gpt-char",
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "ctx": cfg["seq_len"],
        "vocab_size": tokenizer.vocab_size,
        "trained_steps": args.steps,
        "final_val_loss": history.get("final_val_loss"),
        "final_perplexity": history.get("final_perplexity"),
        "corpus_chars": corpus_stats["chars"],
        "param_count": history["param_count"],
        "trained_on": trained_on,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": __import__("torch").__version__,
        "seed": args.seed,
    }

    purepy.save_checkpoint(args.out, export_weights(model), meta, tokenizer.to_dict())
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"\nCheckpoint written: {args.out} ({size_kb:.1f} KB, {meta['param_count']:,} params)")
    print(f"Final val loss: {meta['final_val_loss']} | perplexity: {meta['final_perplexity']} | "
          f"time: {history['elapsed_seconds']}s")

    samples_text = "\n\n".join(f"# step {s['step']}\n{s['sample']}" for s in history["samples"][-3:])

    report = dict(history)
    report.update({
        "preset": args.preset, "vocab_size": tokenizer.vocab_size,
        "corpus_chars": corpus_stats["chars"], "corpus_stats": corpus_stats,
        "trained_on": trained_on, "final_val_loss": meta["final_val_loss"],
        "final_perplexity": meta["final_perplexity"],
    })
    _write_step_summary(report, samples_text)

    # Also verify the exported checkpoint loads in the pure-Python runtime
    return self_check(args.out)

if __name__ == "__main__":
    sys.exit(main())
