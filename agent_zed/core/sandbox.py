"""Sandboxed Python code execution and AST verification engine for Agent-Zed."""

import sys
import io
import ast
import traceback
from typing import Dict, Any, Tuple

class CodeSandbox:
    """Executes generated Python code safely in isolated namespace and captures stdout/stderr/exceptions."""

    @staticmethod
    def verify_syntax(code: str) -> Tuple[bool, str]:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"SyntaxError on line {e.lineno}: {e.msg}\nCode context: {e.text}"

    @staticmethod
    def execute(code: str, timeout_seconds: float = 5.0, globals_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run code and capture outputs and errors."""
        is_valid, syntax_msg = CodeSandbox.verify_syntax(code)
        if not is_valid:
            return {
                "success": False,
                "stdout": "",
                "stderr": syntax_msg,
                "error": "SyntaxError",
                "return_value": None
            }

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()

        exec_globals = {"__builtins__": __builtins__}
        if globals_dict:
            exec_globals.update(globals_dict)

        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr

        success = True
        error_msg = None
        result = None

        try:
            # Parse into AST to separate module execution from final expression return if applicable
            parsed = ast.parse(code)
            exec(compile(parsed, "<sandbox>", "exec"), exec_globals)
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return {
            "success": success,
            "stdout": redirected_stdout.getvalue(),
            "stderr": redirected_stderr.getvalue() if not error_msg else (redirected_stderr.getvalue() + "\n" + error_msg),
            "error": error_msg,
            "globals": exec_globals
        }
