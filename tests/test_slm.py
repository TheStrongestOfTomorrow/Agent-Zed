"""Tests for the TinyCodeGPT SLM subsystem: tokenizer, corpus, pure-Python
runtime, checkpoint I/O, engine integration, and (when torch is available)
torch<->purepy architectural parity + a training smoke test."""

import json
import os
import tempfile
import unittest

from agent_zed.slm import purepy
from agent_zed.slm.tokenizer import CharTokenizer
from agent_zed.slm.corpus import build_corpus
from agent_zed.slm.model import TinyCodeGPT

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TINY_CFG = {"vocab_size": 40, "ctx": 16, "d_model": 16, "n_heads": 2, "n_layers": 1}

class TestCharTokenizer(unittest.TestCase):

    def test_roundtrip(self):
        text = "def foo():\n    return 42\n"
        tok = CharTokenizer.build(text)
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_unknown_chars_map_to_unk(self):
        tok = CharTokenizer.build("abc")
        ids = tok.encode("a\u2603b")
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[1], tok.unk_id)

    def test_dict_roundtrip(self):
        tok = CharTokenizer.build("xyz\n")
        tok2 = CharTokenizer.from_dict(tok.to_dict())
        self.assertEqual(tok.vocab, tok2.vocab)
        self.assertEqual(tok.encode("zyx"), tok2.encode("zyx"))

class TestCorpusBuilder(unittest.TestCase):

    def test_builds_real_python_text(self):
        text, stats = build_corpus(stdlib_modules=["statistics", "bisect"], max_bytes=200_000)
        self.assertGreater(stats["chars"], 5_000)
        self.assertGreaterEqual(stats["repo_files"], 5)
        self.assertIn("def ", text)
        self.assertLessEqual(len(text), 200_000)

class TestPurePyRuntime(unittest.TestCase):

    def setUp(self):
        self.weights = purepy.random_weights(seed=3, **TINY_CFG)
        self.meta = {**TINY_CFG, "trained_steps": 0}
        self.model = purepy.PurePyGPT(self.weights, self.meta)

    def test_forward_full_shapes(self):
        tokens = [1, 2, 3, 4, 5]
        logits = self.model.forward_full(tokens)
        self.assertEqual(len(logits), 5)
        self.assertEqual(len(logits[0]), TINY_CFG["vocab_size"])

    def test_kv_cache_matches_full_forward(self):
        """Greedy decoding via the KV cache must equal step-by-step full forward."""
        tokens = [2, 5, 1, 9]
        # cached generation
        cached_ids = self.model.generate(tokens, max_new_tokens=4, temperature=0.0, seed=0)
        # reference: recompute full forward each step
        ref_ids = []
        ctx = list(tokens)
        for _ in range(4):
            logits = self.model.forward_full(ctx)
            nxt = purepy.PurePyGPT._argmax(logits[-1])
            ref_ids.append(nxt)
            ctx.append(nxt)
        self.assertEqual(cached_ids, ref_ids)

    def test_generate_is_seeded_deterministic(self):
        a = self.model.generate([1, 2, 3], max_new_tokens=6, temperature=0.9, seed=42)
        b = self.model.generate([1, 2, 3], max_new_tokens=6, temperature=0.9, seed=42)
        self.assertEqual(a, b)

    def test_generate_respects_context_window(self):
        out = self.model.generate(list(range(TINY_CFG["ctx"])), max_new_tokens=10, temperature=0.0, seed=0)
        self.assertIsInstance(out, list)

    def test_param_count(self):
        n = purepy.count_params(self.weights)
        self.assertGreater(n, 1000)

class TestCheckpointIO(unittest.TestCase):

    def test_save_load_roundtrip(self):
        weights = purepy.random_weights(seed=5, **TINY_CFG)
        meta = {**TINY_CFG, "trained_steps": 10}
        tok = CharTokenizer.build("abc\n")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ckpt.json")
            purepy.save_checkpoint(path, weights, meta, tok.to_dict())
            w2, m2, t2 = purepy.load_checkpoint(path)
        self.assertEqual(m2, meta)
        self.assertEqual(t2["vocab"], tok.vocab)
        model_a = purepy.PurePyGPT(weights, meta)
        model_b = purepy.PurePyGPT(w2, m2)
        ids_a = model_a.generate([0, 1], max_new_tokens=4, temperature=0.0, seed=0)
        ids_b = model_b.generate([0, 1], max_new_tokens=4, temperature=0.0, seed=0)
        self.assertEqual(ids_a, ids_b)  # 6-decimal rounding must not change greedy path

    def test_bad_format_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.json")
            with open(path, "w") as f:
                json.dump({"format": "nope"}, f)
            with self.assertRaises(ValueError):
                purepy.load_checkpoint(path)

class TestTinyCodeGPTWrapper(unittest.TestCase):

    def test_simulation_mode_backwards_compatible(self):
        m = TinyCodeGPT(vocab_size=100, max_len=32, d_model=16)
        out = m.generate_tokens([1, 2], max_new_tokens=3)
        self.assertEqual(len(out), 5)
        self.assertFalse(m.is_trained)
        self.assertEqual(m.status()["mode"], "simulation")
        with self.assertRaises(RuntimeError):
            m.generate_text("def ")

    def test_from_pretrained_real_generation(self):
        weights = purepy.random_weights(seed=11, **{**TINY_CFG, "vocab_size": 30})
        meta = {**TINY_CFG, "vocab_size": 30, "trained_steps": 100}
        tok = CharTokenizer.build("abcdefghijklmnopqrstuvwxyz0123\n ")
        meta["vocab_size"] = tok.vocab_size
        weights = purepy.random_weights(seed=11, vocab_size=tok.vocab_size, ctx=16, d_model=16, n_heads=2, n_layers=1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "w.json")
            purepy.save_checkpoint(path, weights, {**meta, "vocab_size": tok.vocab_size}, tok.to_dict())
            m = TinyCodeGPT.from_pretrained(path)
        self.assertTrue(m.is_trained)
        self.assertEqual(m.status()["mode"], "trained")
        s1 = m.generate_text("def ", max_new_tokens=8, temperature=0.9, seed=1)
        s2 = m.generate_text("def ", max_new_tokens=8, temperature=0.9, seed=1)
        self.assertEqual(s1, s2)
        self.assertIsInstance(s1, str)

    def test_try_load_default_never_raises(self):
        result = TinyCodeGPT.try_load_default()
        self.assertTrue(result is None or isinstance(result, TinyCodeGPT))

class TestEngineIntegration(unittest.TestCase):

    def test_engine_exposes_slm_status(self):
        from agent_zed.core.engine import MoAEngine
        engine = MoAEngine()
        status = engine.slm_status()
        self.assertIn(status["mode"], ("trained", "simulation", "untrained-weights"))
        self.assertIn("param_count", status)

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestTorchParityAndTraining(unittest.TestCase):

    def test_exported_weights_match_torch_forward(self):
        from agent_zed.slm.torch_trainer import TinyGPT, export_weights
        torch.manual_seed(0)
        V, ctx, d, H, L = 40, 16, 16, 2, 2
        model = TinyGPT(V, ctx, d, H, L)
        tokens = [3, 1, 4, 1, 5, 9, 2, 6]
        with torch.no_grad():
            logits_t, _ = model(torch.tensor([tokens]))
        meta = {"d_model": d, "n_layers": L, "n_heads": H, "ctx": ctx, "vocab_size": V}
        pp = purepy.PurePyGPT(export_weights(model), meta)
        logits_p = pp.forward_full(tokens)
        for row_t, row_p in zip(logits_t[0].tolist(), logits_p):
            for a, b in zip(row_t, row_p):
                self.assertAlmostEqual(a, b, delta=1e-3)

    def test_training_smoke_loss_decreases(self):
        from agent_zed.slm.torch_trainer import train_tiny_gpt
        # Highly repetitive synthetic corpus - loss must clearly decrease
        corpus = ("def add(a, b):\n    return a + b\n\n" * 120) + ("def mul(a, b):\n    return a * b\n\n" * 120)
        model, tok, hist = train_tiny_gpt(
            corpus, steps=120, batch_size=8, seq_len=32,
            d_model=32, n_heads=2, n_layers=1, lr=3e-3,
            eval_interval=60, log_every=60,
        )
        losses = [p["loss"] for p in hist["loss_curve"]]
        self.assertGreater(losses[0], losses[-1], f"loss did not decrease: {losses}")
        self.assertIsNotNone(hist["final_val_loss"])

    def test_train_export_purepy_load_pipeline(self):
        """End-to-end: train a few steps, export, reload in pure Python, generate."""
        from agent_zed.slm.torch_trainer import train_tiny_gpt, export_weights
        corpus = "x = 1\ny = 2\nprint(x + y)\n" * 200
        model, tok, hist = train_tiny_gpt(
            corpus, steps=40, batch_size=8, seq_len=24,
            d_model=24, n_heads=2, n_layers=1, eval_interval=40, log_every=40,
        )
        meta = {"arch": "tiny-gpt-char", "d_model": 24, "n_layers": 1, "n_heads": 2,
                "ctx": 24, "vocab_size": tok.vocab_size, "trained_steps": 40,
                "param_count": hist["param_count"]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "e2e.json")
            purepy.save_checkpoint(path, export_weights(model), meta, tok.to_dict())
            m = TinyCodeGPT.from_pretrained(path)
            self.assertTrue(m.is_trained)
            text = m.generate_text("x =", max_new_tokens=12, temperature=0.0, seed=0)
            self.assertIsInstance(text, str)

if __name__ == "__main__":
    unittest.main()
