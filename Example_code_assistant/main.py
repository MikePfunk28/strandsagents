from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from logging_config import init_logging
from memory_assistant import MemoryAssistant
from meta_tool_assistant import MetaToolAssistant
from strands import Agent
from strands.models import OllamaModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands_tools import editor, file_read, file_write, python_repl, shell

from utils.prompt import CODE_ASSISTANT_PROMPT
from utils.tools import (
    code_execute,
    code_generator,
    code_reviewer,
    code_writer_agent,
    project_reader,
)

init_logging()
logger = logging.getLogger(__name__)


ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2",
)


def _python_runner(code: str) -> str:
    agent = Agent(tools=[python_repl])
    return agent.tool.python_repl(code=code)


def _shell_runner(command: str) -> str:
    agent = Agent(tools=[shell])
    return agent.tool.shell(command=command)


def _file_reader(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    agent = Agent(tools=[file_read])
    return agent.tool.file_read(path=path, mode="view", encoding=encoding)


def _file_writer(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> Dict[str, Any]:
    agent = Agent(tools=[file_write])
    return agent.tool.file_write(path=path, content=content, mode=mode, encoding=encoding)


def _editor_update(path: str, instructions: str) -> str:
    agent = Agent(tools=[editor])
    return agent.tool.editor(path=path, instructions=instructions)


CODING_TOOLBOX: Dict[str, Dict[str, Any]] = {
    "project_reader": {
        "description": "Reads files from a project directory to build context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_directory": {"type": "string"},
            },
            "required": ["project_directory"],
        },
        "handler": project_reader,
    },
    "code_generator": {
        "description": "Generates code snippets for a requested task.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
        "handler": code_generator,
    },
    "code_reviewer": {
        "description": "Reviews code and suggests improvements.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        "handler": code_reviewer,
    },
    "code_writer": {
        "description": "Writes code artifacts to disk using editor/file tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "project_name": {"type": "string"},
            },
            "required": ["code", "project_name"],
        },
        "handler": code_writer_agent,
    },
    "code_executor": {
        "description": "Executes Python code in an isolated REPL.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        "handler": code_execute,
    },
    "python_repl": {
        "description": "Runs raw Python code via python_repl tool.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        "handler": _python_runner,
    },
    "shell": {
        "description": "Executes shell commands in the project workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        "handler": _shell_runner,
    },
    "file_read": {
        "description": "Reads file contents using the file_read tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "required": ["path"],
        },
        "handler": _file_reader,
    },
    "file_write": {
        "description": "Writes text to disk using the file_write tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        "handler": _file_writer,
    },
    "editor": {
        "description": "Updates files using the editor tool with natural language instructions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "instructions": {"type": "string"},
            },
            "required": ["path", "instructions"],
        },
        "handler": _editor_update,
    },
}


class SimpleCodingAssistant:
    """Multi-agent coding assistant orchestrated by the MetaToolAssistant."""

    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps
        available_tools = {
            name: {
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for name, spec in CODING_TOOLBOX.items()
        }
        self.meta = MetaToolAssistant(
            model=ollama_model,
            available_tools=available_tools,
            system_prompt="You orchestrate specialised coding tools. Always request parameters needed to run the tool.",
        )
        self.memory = MemoryAssistant(user_id="coding_assistant")
        self.base_agent = Agent(
            system_prompt=CODE_ASSISTANT_PROMPT,
            model=ollama_model,
            tools=[],
            conversation_manager=SlidingWindowConversationManager(window_size=8),
        )

    def _execute(self, agent_name: str, parameters: Dict[str, Any]) -> Any:
        handler = CODING_TOOLBOX.get(agent_name, {}).get("handler")
        if handler is None:
            raise ValueError(f"Unknown agent selected: {agent_name}")
        logger.debug("Executing %s with parameters %s", agent_name, parameters)
        return handler(**parameters)

    def run(
        self,
        user_query: str,
        project_directory: Optional[str] = None,
        history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger.info("Starting coding assistant workflow for query: %s", user_query)
        history = history or []
        current_results: Dict[str, Any] = {}
        workflow_steps: List[Dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            meta_decision = self.meta.select_next_action(
                user_query=user_query,
                context=project_directory or "",
                history=history,
                thinking_results={},
                current_results=current_results,
            )
            agent_name = meta_decision.get("next_agent", "code_generator")
            parameters = meta_decision.get("parameters") or {}

            if agent_name == "project_reader" and project_directory and "project_directory" not in parameters:
                parameters["project_directory"] = project_directory
            if agent_name in {"code_generator", "code_reviewer"} and "task" not in parameters and "code" not in parameters:
                parameters.setdefault("task", user_query)
                if agent_name == "code_reviewer" and "code" not in parameters:
                    generated = current_results.get("code_generator")
                    if isinstance(generated, str):
                        parameters["code"] = generated
            if agent_name in {"code_writer", "code_executor", "python_repl"} and "code" not in parameters:
                generated = current_results.get("code_reviewer") or current_results.get("code_generator")
                if isinstance(generated, str):
                    parameters["code"] = generated
            if agent_name == "code_writer" and "project_name" not in parameters:
                parameters["project_name"] = project_directory or "code-assistant-session"

            try:
                result = self._execute(agent_name, parameters)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Error executing %s: %s", agent_name, exc)
                result = {"status": "error", "message": str(exc)}

            current_results[agent_name] = result
            workflow_steps.append(
                {
                    "step": step,
                    "agent": agent_name,
                    "parameters": parameters,
                    "result": result,
                    "reasoning": meta_decision.get("reasoning"),
                    "confidence": meta_decision.get("confidence"),
                }
            )

            self.memory.store_memory(
                json.dumps(
                    {
                        "agent": agent_name,
                        "parameters": parameters,
                        "result": result,
                    }
                ),
                metadata={"type": "coding_step", "step": step, "query": user_query},
            )

            if meta_decision.get("intent") == "create_tool":
                logger.info("Meta tool requested tool creation – not supported in this workflow")
                break

            if agent_name in {"code_writer", "code_executor", "python_repl"}:
                logger.info("Terminating workflow after execution step %s", agent_name)
                break

        summary = self.base_agent(
            "Summarize the coding workflow results:\n" + json.dumps(workflow_steps, indent=2)
        )

        return {
            "summary": str(summary),
            "steps": workflow_steps,
            "final_agent": workflow_steps[-1]["agent"] if workflow_steps else None,
        }


if __name__ == "__main__":
    assistant = SimpleCodingAssistant()
    print("\n?? Simple Coding Assistant ??")
    print("Type your coding request (or 'exit' to quit). Optionally prefix with 'dir:' to provide a project path.")
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            project_dir = None
            query = user_input
            if user_input.startswith("dir:"):
                payload = user_input[4:]
                if " " in payload:
                    project_dir, query = payload.split(" ", 1)
                else:
                    project_dir = payload.strip() or None
                    query = input("Describe the coding task: ")
            response = assistant.run(query, project_directory=project_dir)
            print(json.dumps(response, indent=2))
        except KeyboardInterrupt:
            print("\nInterrupted – exiting")
            break
        except Exception as exc:  # pragma: no cover - interactive loop safeguard
            logger.exception("Unexpected error in CLI loop: %s", exc)
            print(f"Error: {exc}")