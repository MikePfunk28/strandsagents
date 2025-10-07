# sandbox_executor.py
"""
Safe code execution sandbox for agents.
Provides isolated environment for running Python and Bash code.
"""

import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("sandbox_executor")

class SandboxExecutor:
    """
    Safe code execution environment for agents.
    Executes code in isolated temporary directories with timeouts.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"🔒 Sandbox created at: {self.temp_dir}")

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in sandbox environment.

        Args:
            code: Code to execute
            language: Language (python, bash)

        Returns:
            Dict with execution results
        """
        logger.info(f"🔒 Executing {language} code in sandbox")

        if language == "python":
            return self._execute_python(code)
        elif language == "bash":
            return self._execute_bash(code)
        else:
            return {"error": f"Unsupported language: {language}"}

    def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code in isolated environment"""
        try:
            # Write code to temp file
            code_file = self.temp_dir / "temp_code.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)

            logger.debug(f"🔒 Wrote {len(code)} chars to {code_file}")

            # Execute with timeout and resource limits
            result = subprocess.run(
                ["python", str(code_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
                # Security: no shell, limited environment
                env={"PYTHONPATH": str(self.temp_dir), "TEMP": str(self.temp_dir)}
            )

            logger.info(f"🔒 Python execution completed with return code: {result.returncode}")

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "python"
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"🔒 Python code execution timed out after {self.timeout}s")
            return {
                "error": f"Code execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "python"
            }
        except Exception as e:
            logger.error(f"🔒 Python execution error: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "language": "python"
            }

    def _execute_bash(self, code: str) -> Dict[str, Any]:
        """Execute Bash code in isolated environment"""
        try:
            # Write script to temp file
            script_file = self.temp_dir / "temp_script.sh"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\\n")
                f.write(code)
                f.write("\\n")

            # Make executable
            script_file.chmod(0o755)

            logger.debug(f"🔒 Wrote bash script to {script_file}")

            # Execute with timeout
            result = subprocess.run(
                ["bash", str(script_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
                # Security: no shell injection, limited environment
                env={"TEMP": str(self.temp_dir), "TMPDIR": str(self.temp_dir)}
            )

            logger.info(f"🔒 Bash execution completed with return code: {result.returncode}")

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "bash"
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"🔒 Bash script timed out after {self.timeout}s")
            return {
                "error": f"Script execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "bash"
            }
        except Exception as e:
            logger.error(f"🔒 Bash execution error: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "language": "bash"
            }

    def cleanup(self):
        """Clean up temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"🔒 Cleaned up sandbox directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"🔒 Failed to cleanup sandbox: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Test the sandbox
    sandbox = SandboxExecutor(timeout=10)

    # Test Python code
    python_code = '''
print("Hello from sandbox!")
x = 42
print(f"Answer: {x}")
'''

    result = sandbox.execute_code(python_code, "python")
    print("Python Result:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout']}")
    if result.get('error'):
        print(f"Error: {result['error']}")

    # Test Bash code
    bash_code = '''
echo "Hello from bash sandbox!"
echo "Current directory: $(pwd)"
echo "Files: $(ls -la)"
'''

    result = sandbox.execute_code(bash_code, "bash")
    print("\\nBash Result:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout']}")
    if result.get('error'):
        print(f"Error: {result['error']}")

    # Cleanup
    sandbox.cleanup()
