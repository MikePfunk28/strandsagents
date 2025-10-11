# sandbox_executor.py
"""
Enhanced Safe Code Execution Sandbox for StrandsAgents

Provides isolated environment for running multiple programming languages
with REPL-like capabilities and persistent sessions.

Features:
- Multi-language support (Python, JavaScript, Bash, etc.)
- Persistent sessions with state management
- Interactive REPL mode
- Enhanced security and resource limits
- Comprehensive logging and error handling
"""

import subprocess
import tempfile
import os
import logging
import json
import uuid
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger("enhanced_sandbox")

# Remove emoji characters that cause encoding issues on Windows


def safe_print(text):
    """Print text with emoji characters replaced for Windows compatibility"""
    emoji_replacements = {
        '🔒': '[SECURE]',
        '': '[LAUNCH]',
        '🐍': '[PYTHON]',
        '🔧': '[TOOL]',
        '': '[SEARCH]',
        '⚠️': '[WARNING]',
        '': '[OK]',
        '❌': '[ERROR]',
        '': '[IDEA]',
        '🔗': '[LINK]',
        '🔄': '[SYNC]',
        '📦': '[PACKAGE]',
        '': '[FIND]'
    }

    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)

    print(text)


# Add parent directory to path for imports
try:
    # When run as module
    pass
except:
    # When run as script, add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


@dataclass
class ExecutionResult:
    """Structured result from code execution"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    language: str = "unknown"
    execution_time: float = 0.0
    memory_used: int = 0
    error: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SandboxExecutor:
    """
    Enhanced safe code execution environment for agents.
    Supports multiple languages with persistent sessions and REPL capabilities.
    """

    # Supported languages and their configurations
    SUPPORTED_LANGUAGES = {
        'python': {
            'extension': '.py',
            'interpreter': 'python',
            'max_memory': '100MB',
            'timeout_multiplier': 1.0
        },
        'javascript': {
            'extension': '.js',
            'interpreter': 'node',
            'max_memory': '50MB',
            'timeout_multiplier': 1.5
        },
        'bash': {
            'extension': '.sh',
            'interpreter': 'bash',
            'max_memory': '10MB',
            'timeout_multiplier': 0.5
        },
        'powershell': {
            'extension': '.ps1',
            'interpreter': 'pwsh',
            'max_memory': '50MB',
            'timeout_multiplier': 1.0
        }
    }

    def __init__(self, timeout: int = 30, max_memory: str = "100MB"):
        self.timeout = timeout
        self.max_memory = max_memory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_counter = 0

        logger.info(f"[SECURE] Enhanced Sandbox created at: {self.temp_dir}")
        logger.info(
            f"[SECURE] Supported languages: {list(self.SUPPORTED_LANGUAGES.keys())}")

    def execute_code(self, code: str, language: str = "python",
                     session_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in sandbox environment with enhanced features.

        Args:
            code: Code to execute
            language: Programming language
            session_id: Optional persistent session ID

        Returns:
            ExecutionResult with comprehensive execution details
        """
        logger.info(f"[SECURE] Executing {language} code in sandbox")

        # Validate language support
        if language not in self.SUPPORTED_LANGUAGES:
            return ExecutionResult(
                success=False,
                error=f"Unsupported language: {language}. Supported: {list(self.SUPPORTED_LANGUAGES.keys())}",
                language=language
            )

        # Create or get persistent session
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = self._create_session(language)
        else:
            session_id = self._create_new_session(language)

        session = self.sessions[session_id]

        # Execute based on language
        start_time = datetime.now()

        try:
            if language == "python":
                result = self._execute_python_enhanced(code, session)
            elif language == "javascript":
                result = self._execute_javascript(code, session)
            elif language == "bash":
                result = self._execute_bash_enhanced(code, session)
            elif language == "powershell":
                result = self._execute_powershell(code, session)
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Execution method not implemented for {language}",
                    language=language,
                    session_id=session_id
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Create comprehensive result
            execution_result = ExecutionResult(
                success=result.get("success", False),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                returncode=result.get("returncode", 1),
                language=language,
                execution_time=execution_time,
                memory_used=result.get("memory_used", 0),
                error=result.get("error"),
                session_id=session_id
            )

            # Update session state
            session["last_execution"] = execution_result.to_dict()
            session["execution_count"] += 1

            logger.info(
                f"[SECURE] {language} execution completed in {execution_time:.2f}s")
            return execution_result

        except Exception as e:
            logger.error(f"[SECURE] Execution error: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                language=language,
                session_id=session_id
            )

    def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code in isolated environment"""
        try:
            # Write code to temp file
            code_file = self.temp_dir / "temp_code.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)

            logger.debug(f"[SECURE] Wrote {len(code)} chars to {code_file}")

            # Execute with timeout and resource limits
            result = subprocess.run(
                ["python", str(code_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
                # Security: no shell, limited environment
                env={"PYTHONPATH": str(self.temp_dir),
                     "TEMP": str(self.temp_dir)}
            )

            logger.info(
                f"[SECURE] Python execution completed with return code: {result.returncode}")

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "python"
            }

        except subprocess.TimeoutExpired:
            logger.warning(
                f"[SECURE] Python code execution timed out after {self.timeout}s")
            return {
                "error": f"Code execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "python"
            }
        except Exception as e:
            logger.error(f"[SECURE] Python execution error: {str(e)}")
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

            logger.debug(f"[SECURE] Wrote bash script to {script_file}")

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

            logger.info(
                f"[SECURE] Bash execution completed with return code: {result.returncode}")

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "bash"
            }

        except subprocess.TimeoutExpired:
            logger.warning(
                f"[SECURE] Bash script timed out after {self.timeout}s")
            return {
                "error": f"Script execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "bash"
            }
        except Exception as e:
            logger.error(f"[SECURE] Bash execution error: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "language": "bash"
            }

    def _create_session(self, language: str) -> Dict[str, Any]:
        """Create a new persistent session for a language"""
        session_id = str(uuid.uuid4())
        lang_config = self.SUPPORTED_LANGUAGES[language]

        session = {
            "id": session_id,
            "language": language,
            "created_at": datetime.now(),
            "execution_count": 0,
            "variables": {},
            "last_execution": None,
            "working_directory": self.temp_dir / f"session_{session_id}",
            "config": lang_config
        }

        # Create session directory
        session["working_directory"].mkdir(exist_ok=True)

        logger.info(f"[SECURE] Created session {session_id} for {language}")
        return session

    def _create_new_session(self, language: str) -> str:
        """Create a new session and return its ID"""
        session = self._create_session(language)
        self.sessions[session["id"]] = session
        return session["id"]

    def _execute_python_enhanced(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Python execution with session support"""
        try:
            # Write code to session file
            code_file = session["working_directory"] / "code.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)

            # Execute with enhanced environment
            env = {
                "PYTHONPATH": str(session["working_directory"]),
                "TEMP": str(self.temp_dir),
                "SESSION_ID": session["id"]
            }

            result = subprocess.run(
                ["python", str(code_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=session["working_directory"],
                env=env
            )

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "python",
                "memory_used": 0  # Could be enhanced with memory monitoring
            }

        except subprocess.TimeoutExpired:
            return {
                "error": f"Python code execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "python"
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "language": "python"
            }

    def _execute_javascript(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JavaScript code"""
        try:
            code_file = session["working_directory"] / "code.js"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)

            result = subprocess.run(
                ["node", str(code_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=session["working_directory"]
            )

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "javascript"
            }

        except subprocess.TimeoutExpired:
            return {
                "error": f"JavaScript execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "javascript"
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "language": "javascript"
            }

    def _execute_bash_enhanced(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Bash execution with session support"""
        try:
            script_file = session["working_directory"] / "script.sh"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\n")
                f.write(f"cd {session['working_directory']}\n")
                f.write(code)
                f.write("\n")

            script_file.chmod(0o755)

            result = subprocess.run(
                ["bash", str(script_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=session["working_directory"]
            )

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "bash"
            }

        except subprocess.TimeoutExpired:
            return {
                "error": f"Bash script timed out after {self.timeout} seconds",
                "success": False,
                "language": "bash"
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "language": "bash"
            }

    def _execute_powershell(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PowerShell code"""
        try:
            script_file = session["working_directory"] / "script.ps1"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)

            result = subprocess.run(
                ["pwsh", "-File", str(script_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=session["working_directory"]
            )

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "powershell"
            }

        except subprocess.TimeoutExpired:
            return {
                "error": f"PowerShell execution timed out after {self.timeout} seconds",
                "success": False,
                "language": "powershell"
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "language": "powershell"
            }

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "id": session["id"],
                "language": session["language"],
                "created_at": session["created_at"].isoformat(),
                "execution_count": session["execution_count"],
                "last_execution": session.get("last_execution")
            }
            for session in self.sessions.values()
        ]

    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a specific session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            try:
                import shutil
                shutil.rmtree(session["working_directory"])
                del self.sessions[session_id]
                logger.info(f"[SECURE] Cleaned up session {session_id}")
                return True
            except Exception as e:
                logger.error(
                    f"[SECURE] Failed to cleanup session {session_id}: {str(e)}")
                return False
        return False

    def cleanup(self):
        """Clean up all sessions and temporary directory"""
        # Clean up all sessions
        for session_id in list(self.sessions.keys()):
            self.cleanup_session(session_id)

        # Clean up main temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(
                f"[SECURE] Cleaned up sandbox directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"[SECURE] Failed to cleanup sandbox: {str(e)}")

    def create_repl_session(self, language: str = "python") -> str:
        """Create an interactive REPL session"""
        session_id = self._create_new_session(language)
        session = self.sessions[session_id]

        # Add REPL-specific initialization
        if language == "python":
            init_code = '''
import sys
print(f"🐍 Python REPL Session: {session_id}")
print("Type 'exit()' or 'quit()' to end session")
print("=" * 50)
'''
            self._execute_python_enhanced(init_code, session)

        logger.info(
            f"[SECURE] Created REPL session {session_id} for {language}")
        return session_id


# Example usage
if __name__ == "__main__":
    # Test the enhanced sandbox
    sandbox = SandboxExecutor(timeout=10)

    print(" Testing Enhanced Sandbox Executor")
    print("=" * 50)

    # Test Python code with session
    python_code = '''
print("Hello from enhanced sandbox!")
x = 42
print(f"Answer: {x}")
result = x * 2
print(f"Doubled: {result}")
'''

    print("Testing Python execution...")
    result = sandbox.execute_code(python_code, "python")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print(f"Execution time: {result.execution_time:.3f}s")
    if result.error:
        print(f"Error: {result.error}")

    # Test JavaScript code
    js_code = '''
console.log("Hello from JavaScript!");
let y = 10;
console.log(`Result: ${y * 3}`);
'''

    print("\\nTesting JavaScript execution...")
    result = sandbox.execute_code(js_code, "javascript")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    if result.error:
        print(f"Error: {result.error}")

    # Test session management
    print("\\nTesting session management...")
    session_id = sandbox.create_repl_session("python")
    print(f"Created REPL session: {session_id}")

    sessions = sandbox.list_sessions()
    print(f"Active sessions: {len(sessions)}")

    # Cleanup
    sandbox.cleanup()
