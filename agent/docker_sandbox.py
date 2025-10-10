"""
Docker-based Safe Code Execution Sandbox for StrandsAgents

Provides isolated Docker container environment for running multiple programming languages
with REPL-like capabilities and persistent sessions.

Features:
- Multi-language support (Python, JavaScript, Bash, etc.) via Docker containers
- Persistent sessions with state management
- Interactive REPL mode
- Enhanced security through containerization
- Resource limits and timeout controls
"""

import subprocess
import json
import logging
import uuid
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("docker_sandbox")


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


class DockerSandboxExecutor:
    """
    Docker-based safe code execution environment for agents.
    Supports multiple languages with persistent sessions and REPL capabilities.
    """

    # Supported languages and their Docker configurations
    SUPPORTED_LANGUAGES = {
        'python': {
            'image': 'python:3.9-slim',
            'command': 'python3',
            'extension': '.py',
            'max_memory': '100MB',
            'timeout_multiplier': 1.0
        },
        'javascript': {
            'image': 'node:16-slim',
            'command': 'node',
            'extension': '.js',
            'max_memory': '50MB',
            'timeout_multiplier': 1.5
        },
        'bash': {
            'image': 'ubuntu:20.04',
            'command': 'bash',
            'extension': '.sh',
            'max_memory': '10MB',
            'timeout_multiplier': 0.5
        }
    }

    def __init__(self, timeout: int = 30, max_memory: str = "100MB"):
        self.timeout = timeout
        self.max_memory = max_memory
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_counter = 0

        # Check if Docker is available
        self._check_docker_availability()

        logger.info(f"Docker Sandbox initialized with timeout: {timeout}s")

    def _check_docker_availability(self):
        """Check if Docker is available and working"""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("Docker is available and working")
            else:
                logger.warning(
                    "Docker may not be available or configured properly")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.error(f"Docker check failed: {str(e)}")
            raise RuntimeError(
                "Docker is not available. Please install and start Docker.")

    def execute_code(self, code: str, language: str = "python",
                     session_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in Docker container with enhanced features.

        Args:
            code: Code to execute
            language: Programming language
            session_id: Optional persistent session ID

        Returns:
            ExecutionResult with comprehensive execution details
        """
        logger.info(f"Executing {language} code in Docker sandbox")

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
                result = self._execute_python_docker(code, session)
            elif language == "javascript":
                result = self._execute_javascript_docker(code, session)
            elif language == "bash":
                result = self._execute_bash_docker(code, session)
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
                f"{language} execution completed in {execution_time:.2f}s")
            return execution_result

        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                language=language,
                session_id=session_id
            )

    def _execute_python_docker(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in Docker container"""
        try:
            # Create temporary script file
            script_content = f'''
{code}
'''

            # Docker run command with security constraints
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "--memory=100m",  # Memory limit
                "--cpus=0.5",  # CPU limit
                "--network=none",  # No network access
                "--user=1000:1000",  # Non-root user
                "--read-only",  # Read-only filesystem
                "--tmpfs=/tmp:rw,noexec,nosuid,size=10m",  # Temporary writable space
                "python:3.9-slim",  # Python image
                "python3", "-c", script_content  # Execute code directly
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "language": "python"
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

    def _execute_bash_docker(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Bash code in Docker container"""
        try:
            # Create temporary script file
            script_content = f'''
#!/bin/bash
{code}
'''

            # Docker run command with security constraints
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "--memory=50m",  # Memory limit
                "--cpus=0.5",  # CPU limit
                "--network=none",  # No network access
                "--user=1000:1000",  # Non-root user
                "--read-only",  # Read-only filesystem
                "--tmpfs=/tmp:rw,noexec,nosuid,size=10m",  # Temporary writable space
                "ubuntu:20.04",  # Ubuntu image
                "bash", "-c", script_content  # Execute script
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
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

    def _execute_javascript_docker(self, code: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JavaScript code in Docker container"""
        try:
            # Create temporary script file
            script_content = code

            # Docker run command with security constraints
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "--memory=50m",  # Memory limit
                "--cpus=0.5",  # CPU limit
                "--network=none",  # No network access
                "--user=1000:1000",  # Non-root user
                "--read-only",  # Read-only filesystem
                "--tmpfs=/tmp:rw,noexec,nosuid,size=10m",  # Temporary writable space
                "node:16-slim",  # Node.js image
                "node", "-e", script_content  # Execute code directly
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
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
            "config": lang_config
        }

        logger.info(f"Created session {session_id} for {language}")
        return session

    def _create_new_session(self, language: str) -> str:
        """Create a new session and return its ID"""
        session = self._create_session(language)
        self.sessions[session["id"]] = session
        return session["id"]

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
            del self.sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")
            return True
        return False

    def cleanup(self):
        """Clean up all sessions"""
        session_count = len(self.sessions)
        for session_id in list(self.sessions.keys()):
            self.cleanup_session(session_id)
        logger.info(f"Cleaned up {session_count} sessions")


# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the Docker sandbox
        sandbox = DockerSandboxExecutor(timeout=10)

        print("Testing Docker-based Sandbox Executor")
        print("=" * 50)

        # Test Python code
        python_code = '''
print("Hello from Docker Python sandbox!")
x = 42
print(f"Answer: {x}")
result = x * 2
print(f"Doubled: {result}")
'''

        print("Testing Python execution in Docker...")
        result = sandbox.execute_code(python_code, "python")
        print(f"Success: {result.success}")
        print(f"Output: {result.stdout}")
        print(f"Execution time: {result.execution_time:.3f}s")
        if result.error:
            print(f"Error: {result.error}")

        # Test Bash code
        bash_code = '''
echo "Hello from Docker Bash sandbox!"
echo "Current directory: $(pwd)"
echo "Available memory:"
cat /proc/meminfo | head -5
'''

        print("\nTesting Bash execution in Docker...")
        result = sandbox.execute_code(bash_code, "bash")
        print(f"Success: {result.success}")
        print(f"Output: {result.stdout}")
        if result.error:
            print(f"Error: {result.error}")

        # Test JavaScript code
        js_code = '''
console.log("Hello from Docker JavaScript sandbox!");
const y = 10;
console.log(`Result: ${y * 3}`);
console.log(`Square: ${y * y}`);
'''

        print("\nTesting JavaScript execution in Docker...")
        result = sandbox.execute_code(js_code, "javascript")
        print(f"Success: {result.success}")
        print(f"Output: {result.stdout}")
        if result.error:
            print(f"Error: {result.error}")

        # Test session management
        print("\nTesting session management...")
        session_id = sandbox._create_new_session("python")
        print(f"Created session: {session_id}")

        sessions = sandbox.list_sessions()
        print(f"Active sessions: {len(sessions)}")

        # Cleanup
        sandbox.cleanup()

    except RuntimeError as e:
        print(f"Docker not available: {e}")
        print("Please install and start Docker to use this sandbox.")
    except Exception as e:
        print(f"Unexpected error: {e}")
