"""Sandboxed Python code execution and AST verification engine for Agent-Zed."""

import sys
import io
import ast
import threading
import traceback
from typing import Dict, Any, Optional, Tuple

class CodeSandbox:
    """Executes generated Python code safely in isolated namespace and captures stdout/stderr/exceptions.

    Execution runs on a daemon worker thread so that a wall-clock timeout can be
    enforced without blocking the CLI event loop. Note: a timed-out thread cannot
    be force-killed in CPython; it is left running as a daemon and its output is
    discarded. This is acceptable for the lightweight local sandbox budget.
    """

    # Every execute() result contains exactly these keys, on all code paths.
    RESULT_KEYS = ("success", "stdout", "stderr", "error", "timed_out", "globals")

    @staticmethod
    def verify_syntax(code: str) -> Tuple[bool, str]:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"SyntaxError on line {e.lineno}: {e.msg}\nCode context: {e.text}"

    @staticmethod
    def execute(
        code: str,
        timeout_seconds: float = 5.0,
        globals_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run code with an enforced wall-clock timeout; capture outputs and errors.

        Returns a dict with the keys in ``CodeSandbox.RESULT_KEYS``:
        ``success``, ``stdout``, ``stderr``, ``error``, ``timed_out``, ``globals``.
        """
        is_valid, syntax_msg = CodeSandbox.verify_syntax(code)
        if not is_valid:
            return {
                "success": False,
                "stdout": "",
                "stderr": syntax_msg,
                "error": "SyntaxError",
                "timed_out": False,
                "globals": {},
            }

        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()

        exec_globals: Dict[str, Any] = {"__builtins__": __builtins__}
        if globals_dict:
            exec_globals.update(globals_dict)

        error_holder: Dict[str, Any] = {}

        def _run_body():
            try:
                exec(compile(ast.parse(code), "<sandbox>", "exec"), exec_globals)
            except BaseException as e:  # noqa: BLE001 - sandbox must capture everything
                error_holder["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr

        worker = threading.Thread(target=_run_body, daemon=True, name="zed-sandbox")
        try:
            worker.start()
            worker.join(timeout_seconds)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        timed_out = worker.is_alive()

        if timed_out:
            error_msg = f"TimeoutError: execution exceeded {timeout_seconds}s (terminated)"
            success = False
        else:
            error_msg = error_holder.get("error")
            success = error_msg is None

        stderr_value = redirected_stderr.getvalue()
        if error_msg:
            stderr_value = f"{stderr_value}\n{error_msg}" if stderr_value else error_msg

        return {
            "success": success,
            "stdout": redirected_stdout.getvalue(),
            "stderr": stderr_value,
            "error": error_msg,
            "timed_out": timed_out,
            "globals": exec_globals,
        }

    @staticmethod
    def build_test_snippet(code: str, test_case: Dict[str, Any]) -> str:
        """Build a single runnable test source from solution code + test case dict.

        A test case may contain:
        - ``setup``: extra source (helpers/fixtures) appended after the solution
        - ``call``:  expression to evaluate
        - ``expected``: value the expression must equal

        Shared by the MoA engine loop, the DeterministicVerifier, and the
        benchmark runner so all three verify against the exact same source.
        """
        parts = [code.strip()]
        setup = test_case.get("setup")
        if setup:
            parts.append(str(setup).strip())
        func_call = test_case.get("call")
        if func_call is not None:
            expected = test_case.get("expected")
            parts.append(f"# Test execution\nresult = {func_call}")
            parts.append(
                f"assert result == {repr(expected)}, f'Expected {repr(expected)}, got {{result}}'"
            )
        return "\n\n".join(parts) + "\n"
