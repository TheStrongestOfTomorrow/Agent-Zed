"""Comprehensive Unit Tests for Agent-Zed Mixture-of-Agents Architecture."""

import unittest
import asyncio
from agent_zed.core.sandbox import CodeSandbox
from agent_zed.core.jobs import JobManager, JobPriority, JobStatus
from agent_zed.core.engine import MoAEngine, EngineOptimizations
from agent_zed.core.reasoning import TreeOfThoughtEngine
from agent_zed.core.scaling import best_of_n_scaling
from agent_zed.agents.definitions import get_13_agent_personas, WebSearchTool
from agent_zed.benchmark.runner import run_benchmark_suite

class TestAgentZed(unittest.TestCase):

    def test_code_sandbox_execution(self):
        code = "a = 5\nb = 10\nprint(a * b)"
        res = CodeSandbox.execute(code)
        self.assertTrue(res["success"])
        self.assertIn("50", res["stdout"])

    def test_sandbox_syntax_error(self):
        code = "def bad_func(: pass"
        res = CodeSandbox.execute(code)
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "SyntaxError")

    def test_job_manager_creation_and_update(self):
        jm = JobManager()
        job = jm.create_job("Test Job", "Test Description", ["Researcher"], priority=JobPriority.HIGH)
        self.assertEqual(job.status, JobStatus.PENDING)
        jm.update_job_progress(job.id, 50.0, "Running tests...")
        self.assertEqual(job.progress, 50.0)
        jm.mark_completed(job.id, {"result": "ok"})
        self.assertEqual(job.status, JobStatus.COMPLETED)

    def test_13_agent_personas(self):
        personas = get_13_agent_personas()
        self.assertEqual(len(personas), 13)
        self.assertIn("CEO", personas)
        self.assertIn("Researcher", personas)
        self.assertIn("Debugger", personas)
        self.assertIn("Spokesperson", personas)

    def test_engine_context_compression(self):
        opts = EngineOptimizations(max_context_chars=100)
        long_text = "A" * 200
        compressed = opts.compress_context(long_text)
        self.assertIn("Context compressed", compressed)

    def test_benchmark_suite_execution(self):
        pass_rate, total = run_benchmark_suite()
        self.assertEqual(pass_rate, 100.0)
        self.assertEqual(total, 3)


class TestSandboxRobustness(unittest.TestCase):
    """The sandbox must enforce its timeout and return a consistent payload."""

    def test_timeout_is_enforced(self):
        res = CodeSandbox.execute("import time\ntime.sleep(10)", timeout_seconds=0.3)
        self.assertFalse(res["success"])
        self.assertTrue(res["timed_out"])
        self.assertIn("TimeoutError", res["error"])

    def test_fast_code_does_not_report_timeout(self):
        res = CodeSandbox.execute("x = 1 + 1", timeout_seconds=5.0)
        self.assertTrue(res["success"])
        self.assertFalse(res["timed_out"])
        self.assertIsNone(res["error"])

    def test_runtime_exception_captured(self):
        res = CodeSandbox.execute("raise ValueError('boom')")
        self.assertFalse(res["success"])
        self.assertFalse(res["timed_out"])
        self.assertIn("ValueError", res["stderr"])

    def test_result_keys_consistent_across_paths(self):
        codes = [
            "x = 1",                    # success path
            "def f(: pass",             # syntax error path
            "raise ValueError('x')",    # runtime error path
        ]
        for code in codes:
            res = CodeSandbox.execute(code, timeout_seconds=2.0)
            for key in CodeSandbox.RESULT_KEYS:
                self.assertIn(key, res, msg=f"missing key {key!r} for code {code!r}")

    def test_build_test_snippet_includes_setup_and_assert(self):
        snippet = CodeSandbox.build_test_snippet(
            "def f():\n    return 2",
            {"setup": "def helper():\n    return f() * 21", "call": "helper()", "expected": 42},
        )
        res = CodeSandbox.execute(snippet)
        self.assertTrue(res["success"], res["stderr"])

    def test_build_test_snippet_assertion_fails_loudly(self):
        snippet = CodeSandbox.build_test_snippet(
            "def f():\n    return 1",
            {"call": "f()", "expected": 42},
        )
        res = CodeSandbox.execute(snippet)
        self.assertFalse(res["success"])
        self.assertIn("AssertionError", res["stderr"])


class TestTreeOfThoughtSampling(unittest.TestCase):
    """ToT sampling must be reproducible per seed and honor temperature=0 greediness."""

    TASK = "Implement bitmask dp memo TSP solver"
    SEED_CODE = "def solve(n, dist):\n    memo = {}\n    return 0\n"

    def test_sampling_is_deterministic_for_same_seed(self):
        eng = TreeOfThoughtEngine()
        a = eng.sample_thought(self.TASK, self.SEED_CODE, temperature=0.7, seed=42)
        b = eng.sample_thought(self.TASK, self.SEED_CODE, temperature=0.7, seed=42)
        self.assertEqual(a, b)

    def test_greedy_search_returns_valid_code(self):
        eng = TreeOfThoughtEngine()
        content, score = eng.search_best_thought(self.TASK, self.SEED_CODE)
        self.assertIn("def solve", content)
        self.assertGreater(score, 0.0)

    def test_heuristic_penalizes_invalid_syntax(self):
        from agent_zed.core.reasoning import ThoughtNode
        good = ThoughtNode("def f():\n    return 1").evaluate_heuristic("write a function")
        bad = ThoughtNode("def f(:\n    return 1").evaluate_heuristic("write a function")
        self.assertGreater(good, bad)


class TestBestOfNScaling(unittest.TestCase):
    """Best-of-N must pass sample indices, filter invalid code, and vote by majority."""

    def test_two_arg_generator_receives_sample_index(self):
        seen_indices = []

        def gen(temp, idx):
            seen_indices.append(idx)
            return "x = 1"

        best, valid = best_of_n_scaling(gen, n_samples=4, temperature=0.7)
        self.assertEqual(seen_indices, [0, 1, 2, 3])
        self.assertEqual(best, "x = 1")
        self.assertEqual(len(valid), 4)

    def test_single_arg_generator_still_supported(self):
        best, valid = best_of_n_scaling(lambda t: "y = 2", n_samples=3)
        self.assertEqual(best, "y = 2")
        self.assertEqual(len(valid), 3)

    def test_majority_vote_selects_most_common_valid_candidate(self):
        def gen(temp, idx):
            # 2 of 5 samples are invalid syntax; of the valid ones "x = 1" wins 2-1
            return ["x = 1", "def (:", "x = 1", "x = 2", ")bad("][idx]

        best, valid = best_of_n_scaling(gen, n_samples=5, temperature=0.7)
        self.assertEqual(best, "x = 1")
        self.assertEqual(len(valid), 3)

    def test_fallback_when_nothing_valid(self):
        best, valid = best_of_n_scaling(lambda t, i: "def (:", n_samples=2)
        self.assertEqual(best, "def (:")
        self.assertEqual(valid, ["def (:"])

    def test_test_cases_filter_candidates(self):
        def gen(temp, idx):
            return "def f():\n    return 1" if idx % 2 == 0 else "def f():\n    return 42"

        best, valid = best_of_n_scaling(
            gen, n_samples=4, temperature=0.5,
            test_cases=[{"call": "f()", "expected": 42}]
        )
        self.assertEqual(best, "def f():\n    return 42")
        self.assertEqual(len(valid), 2)


class TestEngineRepairHonesty(unittest.TestCase):
    """The engine must not fake verification or burn rounds on no-op repairs."""

    def test_verified_solution_reports_verified_true(self):
        engine = MoAEngine()
        res = asyncio.run(engine.solve_coding_problem(
            "Implement solve(n, dist) to find the minimum Hamiltonian cycle cost using Bitmask Dynamic Programming.",
            test_cases=[{"call": "solve(3, [[0, 10, 15], [10, 0, 35], [15, 35, 0]])", "expected": 60}],
        ))
        self.assertTrue(res["verified"])
        self.assertFalse(res["repair_stalled"])
        self.assertEqual(res["rounds_used"], 1)

    def test_repair_stall_breaks_early_and_reports_honestly(self):
        engine = MoAEngine()
        res = asyncio.run(engine.solve_coding_problem(
            "compute the fibby wibby number thing",  # matches no template keyword
            test_cases=[{"call": "solve(5)", "expected": 999}],
            max_rounds=3,
        ))
        self.assertFalse(res["verified"])
        self.assertTrue(res["repair_stalled"])
        self.assertEqual(res["rounds_used"], 1, "stalled repair must not burn remaining rounds")

    def test_result_payload_contract(self):
        engine = MoAEngine()
        res = asyncio.run(engine.solve_coding_problem("write a thread-safe concurrent queue with push and pop"))
        for key in ("code", "verified", "tests_supplied", "rounds_used", "repair_stalled", "elapsed_seconds", "job_id"):
            self.assertIn(key, res)
        self.assertFalse(res["tests_supplied"])
        job = engine.job_manager.get_job(res["job_id"])
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JobStatus.COMPLETED)


class TestWebSearchToolFallback(unittest.TestCase):
    """The Researcher's web tool must degrade gracefully when offline."""

    def test_search_always_returns_structured_results(self):
        results = WebSearchTool.search_duckduckgo("python asyncio tutorial", max_results=2)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        for r in results[:2]:
            self.assertIn("title", r)
            self.assertIn("snippet", r)


if __name__ == "__main__":
    unittest.main()
