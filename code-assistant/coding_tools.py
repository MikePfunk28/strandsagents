"""Coding tools for the assistant using @tool decorator and meta-tooling."""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from strands import tool

logger = logging.getLogger(__name__)


@tool
def python_repl(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code in a REPL environment."""
    try:
        # Compile the code first to check for syntax errors
        compiled = compile(code, '<string>', 'exec')

        # Capture stdout and stderr
        import io
        import contextlib

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None

        try:
            # Redirect stdout and stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute the code
            exec(compiled)

            # Get the output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            return {
                "success": True,
                "stdout": stdout_output,
                "stderr": stderr_output,
                "result": str(result) if result is not None else None,
            }

        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax Error: {e}",
            "error_type": "SyntaxError",
            "line": e.lineno,
            "column": e.offset,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def file_read(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read a file and return its contents."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
            }

        content = path.read_text(encoding=encoding)

        return {
            "success": True,
            "content": content,
            "file_path": str(path.absolute()),
            "size": len(content),
            "lines": content.count('\n') + 1,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def file_write(file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True) -> Dict[str, Any]:
    """Write content to a file."""
    try:
        path = Path(file_path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding=encoding)

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "size": len(content),
            "lines": content.count('\n') + 1,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def file_append(file_path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Append content to a file."""
    try:
        path = Path(file_path)

        with open(path, 'a', encoding=encoding) as f:
            f.write(content)

        new_size = path.stat().st_size

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "appended_size": len(content),
            "total_size": new_size,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
    """List files in a directory."""
    try:
        path = Path(directory)
        if not path.exists():
            return {
                "success": False,
                "error": f"Directory does not exist: {directory}",
            }

        if not path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {directory}",
            }

        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        file_info = []
        for file_path in files:
            info = {
                "name": file_path.name,
                "path": str(file_path.absolute()),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
            }

            if file_path.is_file():
                stat = file_path.stat()
                info.update({
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })

            file_info.append(info)

        return {
            "success": True,
            "directory": str(path.absolute()),
            "files": file_info,
            "count": len(file_info),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def shell_execute(command: str, working_dir: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Execute a shell command."""
    try:
        cwd = Path(working_dir) if working_dir else None

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command,
            "working_dir": str(cwd.absolute()) if cwd else os.getcwd(),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
            "error_type": "TimeoutExpired",
            "command": command,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "command": command,
        }


@tool
def code_analyze(file_path: str) -> Dict[str, Any]:
    """Analyze Python code file for structure and complexity."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
            }

        content = path.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Syntax error in file: {e}",
                "error_type": "SyntaxError",
            }

        analysis = {
            "file_path": str(path.absolute()),
            "lines": content.count('\n') + 1,
            "characters": len(content),
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_score": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": len(node.args.args),
                    "decorators": len(node.decorator_list),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
                analysis["functions"].append(func_info)

            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": sum(1 for n in node.body if isinstance(n, ast.FunctionDef)),
                    "decorators": len(node.decorator_list),
                }
                analysis["classes"].append(class_info)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append({
                            "module": alias.name,
                            "alias": alias.asname,
                            "type": "import",
                        })
                else:  # ImportFrom
                    for alias in node.names:
                        analysis["imports"].append({
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "type": "from_import",
                        })

        # Simple complexity calculation
        analysis["complexity_score"] = (
            len(analysis["functions"]) * 2 +
            len(analysis["classes"]) * 3 +
            len(analysis["imports"])
        )

        return {
            "success": True,
            "analysis": analysis,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def code_format(file_path: str, formatter: str = "black") -> Dict[str, Any]:
    """Format Python code using a code formatter."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
            }

        original_content = path.read_text()

        if formatter.lower() == "black":
            try:
                import black
                formatted_content = black.format_str(original_content, mode=black.FileMode())
            except ImportError:
                return {
                    "success": False,
                    "error": "Black formatter not available. Install with: pip install black",
                }
        else:
            return {
                "success": False,
                "error": f"Unsupported formatter: {formatter}",
            }

        # Check if formatting changed anything
        if original_content == formatted_content:
            return {
                "success": True,
                "formatted": False,
                "message": "Code was already formatted correctly",
            }

        # Write formatted content back
        path.write_text(formatted_content)

        return {
            "success": True,
            "formatted": True,
            "file_path": str(path.absolute()),
            "original_lines": original_content.count('\n') + 1,
            "formatted_lines": formatted_content.count('\n') + 1,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def code_test(file_path: str, test_framework: str = "pytest") -> Dict[str, Any]:
    """Run tests for a Python file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
            }

        if test_framework.lower() == "pytest":
            command = f"python -m pytest {path} -v"
        elif test_framework.lower() == "unittest":
            command = f"python -m unittest {path.stem}"
        else:
            return {
                "success": False,
                "error": f"Unsupported test framework: {test_framework}",
            }

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=path.parent,
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command,
            "test_framework": test_framework,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Tests timed out after 60 seconds",
            "error_type": "TimeoutExpired",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def git_status(repository_path: str = ".") -> Dict[str, Any]:
    """Get git status for a repository."""
    try:
        path = Path(repository_path)

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=path,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
                "error_type": "GitError",
            }

        # Parse git status output
        files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                status = line[:2]
                filepath = line[3:]
                files.append({
                    "status": status,
                    "file": filepath,
                    "staged": status[0] != ' ',
                    "modified": status[1] != ' ',
                })

        return {
            "success": True,
            "repository": str(path.absolute()),
            "files": files,
            "clean": len(files) == 0,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@tool
def search_code(directory: str, pattern: str, file_pattern: str = "*.py", case_sensitive: bool = False) -> Dict[str, Any]:
    """Search for code patterns in files."""
    try:
        import re

        path = Path(directory)
        if not path.exists():
            return {
                "success": False,
                "error": f"Directory does not exist: {directory}",
            }

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        matches = []
        for file_path in path.rglob(file_pattern):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    lines = content.split('\n')

                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(file_path.relative_to(path)),
                                "line_number": line_num,
                                "line_content": line.strip(),
                                "match": regex.search(line).group(),
                            })

                except Exception:
                    # Skip files that can't be read
                    continue

        return {
            "success": True,
            "pattern": pattern,
            "directory": str(path.absolute()),
            "matches": matches,
            "count": len(matches),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Editor tool that integrates with the meta-tooling system
@tool
def editor(action: str, file_path: Optional[str] = None, content: Optional[str] = None, line_number: Optional[int] = None) -> Dict[str, Any]:
    """Advanced editor tool for code manipulation."""
    try:
        if action == "create":
            if not file_path or content is None:
                return {"success": False, "error": "create action requires file_path and content"}
            return file_write(file_path, content)

        elif action == "read":
            if not file_path:
                return {"success": False, "error": "read action requires file_path"}
            return file_read(file_path)

        elif action == "append":
            if not file_path or content is None:
                return {"success": False, "error": "append action requires file_path and content"}
            return file_append(file_path, content)

        elif action == "insert_line":
            if not file_path or content is None or line_number is None:
                return {"success": False, "error": "insert_line action requires file_path, content, and line_number"}

            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}

            lines = path.read_text().split('\n')
            lines.insert(line_number - 1, content)
            path.write_text('\n'.join(lines))

            return {
                "success": True,
                "action": "insert_line",
                "file_path": str(path.absolute()),
                "line_number": line_number,
                "content": content,
            }

        elif action == "replace_line":
            if not file_path or content is None or line_number is None:
                return {"success": False, "error": "replace_line action requires file_path, content, and line_number"}

            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}

            lines = path.read_text().split('\n')
            if line_number < 1 or line_number > len(lines):
                return {"success": False, "error": f"Line number {line_number} out of range"}

            old_content = lines[line_number - 1]
            lines[line_number - 1] = content
            path.write_text('\n'.join(lines))

            return {
                "success": True,
                "action": "replace_line",
                "file_path": str(path.absolute()),
                "line_number": line_number,
                "old_content": old_content,
                "new_content": content,
            }

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Tool loader for meta-tooling
@tool
def load_tool(tool_name: str, tool_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a tool dynamically for meta-tooling."""
    try:
        if tool_path:
            # Load from file
            import importlib.util
            spec = importlib.util.spec_from_file_location(tool_name, tool_path)
            if spec is None or spec.loader is None:
                return {"success": False, "error": f"Could not load tool from {tool_path}"}

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return {
                "success": True,
                "tool_name": tool_name,
                "tool_path": tool_path,
                "attributes": dir(module),
            }
        else:
            # Load from current module
            current_module = sys.modules[__name__]
            if hasattr(current_module, tool_name):
                tool_func = getattr(current_module, tool_name)
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "tool_function": str(tool_func),
                    "docstring": tool_func.__doc__,
                }
            else:
                return {"success": False, "error": f"Tool {tool_name} not found"}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }