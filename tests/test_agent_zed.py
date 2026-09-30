"""Comprehensive Unit Tests for Agent-Zed Mixture-of-Agents Architecture."""

import unittest
import asyncio
from agent_zed.core.sandbox import CodeSandbox
from agent_zed.core.jobs import JobManager, JobPriority, JobStatus
from agent_zed.core.engine import MoAEngine, EngineOptimizations
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

if __name__ == "__main__":
    unittest.main()
