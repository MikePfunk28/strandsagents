# scaffolder.py
"""
Deterministic agent scaffolding utilities.

The scaffolder collects a structured questionnaire, validates responses, and
generates fully configured agent modules (plus metadata) using the @agent
decorator. It is designed so that an AI assistant can run it after gathering the
required answers from a user, guaranteeing consistent output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agent_scaffolder")

DEFAULT_AGENT_DIR = Path("assistants/generated")
DEFAULT_PROMPT_DIR = Path("assistants/generated/prompts")
DEFAULT_METADATA_DIR = Path("assistants/generated/metadata")
DEFAULT_EMBEDDINGS_DIR = Path("assistants/generated/embeddings")

try:
    from agent.model_selector import ModelSelector  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ModelSelector = None


# ---------------------------------------------------------------------------
# Model catalog helpers
# ---------------------------------------------------------------------------

_SUPPORTED_PROVIDERS = {"ollama", "bedrock"}


def _fallback_model_catalog() -> Tuple[Dict[str, str], Dict[str, str]]:
    chat_models = {
        "qwen3:8b": "ollama",
        "qwen3:4b": "ollama",
        "llama3.2": "ollama",
        "gemma3:12b": "ollama",
        "gemma3:4b": "ollama",
        "gemma3:270m": "ollama",
        "qwen3:1.7b": "ollama",
        "gemma3:1b": "ollama",
        "qwen3-coder:latest": "ollama",
        "qwen3:0.6b": "ollama",
        "qwen3:14b": "ollama",
        "phi4-reasoning:latest": "ollama",
        "phi4-mini:latest": "ollama",
        "phi4-mini-reasoning:latest": "ollama",
        "anthropic.claude-3-sonnet-20240229-v1:0": "bedrock",
        "anthropic.claude-3-haiku-20240307-v1:0": "bedrock",
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "bedrock",
        "meta.llama3-1-8b-instruct-v1:0": "bedrock",
        "meta.llama3-1-70b-instruct-v1:0": "bedrock",
    }

    embedding_models = {
        "qwen3-embedding:8b": "ollama",
        "qwen3-embedding:4b": "ollama",
        "qwen3-embedding:0.6b": "ollama",
        "embeddinggemma:300m": "ollama",
    }

    return chat_models, embedding_models


def _build_model_catalog() -> Tuple[Dict[str, str], Dict[str, str]]:
    if ModelSelector is None:  # pragma: no cover - fallback path
        return _fallback_model_catalog()

    chat: Dict[str, str] = {}
    embedding: Dict[str, str] = {}

    for name, info in ModelSelector.MODEL_CAPABILITIES.items():
        provider = (info.provider or "ollama").lower()
        if provider not in _SUPPORTED_PROVIDERS:
            continue

        capabilities = {cap.lower() for cap in info.capabilities}
        if "embedding" in capabilities:
            embedding[name] = provider
        else:
            chat[name] = provider

    if not chat:
        chat, _ = _fallback_model_catalog()
        chat = {model: provider for model, provider in chat.items() if provider in _SUPPORTED_PROVIDERS}

    if not embedding:
        _, embedding = _fallback_model_catalog()

    return dict(sorted(chat.items())), dict(sorted(embedding.items()))


CHAT_MODEL_PROVIDER_MAP, EMBEDDING_MODEL_PROVIDER_MAP = _build_model_catalog()
AVAILABLE_MODELS = list(CHAT_MODEL_PROVIDER_MAP.keys())
AVAILABLE_EMBED_MODELS = list(EMBEDDING_MODEL_PROVIDER_MAP.keys())
DEFAULT_PROVIDER = "ollama"


def _default_model_for_provider(provider: str) -> str:
    provider = provider.lower()
    for model, model_provider in CHAT_MODEL_PROVIDER_MAP.items():
        if model_provider == provider:
            return model
    return AVAILABLE_MODELS[0]


def _default_embedding_for_provider(provider: str) -> Optional[str]:
    provider = provider.lower()
    for model, model_provider in EMBEDDING_MODEL_PROVIDER_MAP.items():
        if model_provider == provider:
            return model
    return None

AVAILABLE_MODELS = [
    "qwen3:8b",
    "qwen3:4b",
    "llama3.2",
    "gemma3:270m",
    "qwen3:1.7b",
    "gemma3:1b",
    "gemma3:4b",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
    "qwen3-embedding:8b",
    "embeddinggemma:300m",
    "qwen3-coder:latest",
    "qwen3:0.6b",
    "qwen3:14b",
    "phi4-reasoning:latest",
    "phi4-mini:latest",
    "phi4-mini-reasoning:latest"

]

TOOL_LIBRARY: Dict[str, Dict[str, str]] = {
    "sandbox_test_code": {
        "import": "from agent.sandbox_tool import sandbox_test_code",
        "reference": "sandbox_test_code",
        "description": "Execute code snippets safely inside the sandbox."
    },
    "http_request": {
        "import": "from strands_tools import http_request",
        "reference": "http_request",
        "description": "Perform HTTP requests for retrieving remote data."
    },
    "retrieve": {
        "import": "from strands_tools import retrieve",
        "reference": "retrieve",
        "description": "Retrieve web content using the strands retrieval helper."
    },
    "file_read": {
        "import": "from strands_tools import file_read",
        "reference": "file_read",
        "description": "Read files from the workspace."
    },
    "file_write": {
        "import": "from strands_tools import file_write",
        "reference": "file_write",
        "description": "Write files to the workspace."
    }
}

def register_tool(
    name: str,
    import_stmt: str,
    reference: str,
    description: str = "",
) -> None:
    """
    Register a tool so that generated agents can import it.

    This lets higher-level builders add project-specific tools before calling
    ``generate_agent``.  If a tool already exists it will be overwritten with
    the new definition.
    """
    TOOL_LIBRARY[name] = {
        "import": import_stmt,
        "reference": reference,
        "description": description or "Custom tool",
    }


def list_registered_tools() -> Dict[str, Dict[str, str]]:
    """Return a copy of the tool registry."""
    return dict(TOOL_LIBRARY)

MEMORY_PROFILE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "none": {
        "chunk_size": None,
        "chunk_overlap": None,
        "levels": [],
        "description": "No persistent memory; skips chunking and embedding.",
    },
    "light": {
        "chunk_size": 900,
        "chunk_overlap": 180,
        "levels": ["chunks", "sections"],
        "description": "Default profile balancing chunk granularity and storage.",
    },
    "full": {
        "chunk_size": 700,
        "chunk_overlap": 120,
        "levels": ["document", "summary", "sections", "chunks", "sentences"],
        "description": "Rich hierarchy for agents requiring deep retrieval.",
    },
}


@dataclass
class ScaffoldingResult:
    """Structured result returned to the UI/backend."""

    ok: bool
    message: str
    artifacts: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    """Represents a single scaffolding question."""

    id: str
    prompt: str
    help_text: str
    required: bool = True
    default: Optional[Any] = None
    qtype: str = "text"  # text, choice, multi_choice, boolean, integer, list
    choices: Optional[List[Any]] = None


QUESTIONNAIRE: List[Question] = [
    Question(
        id="agent_name",
        prompt="Agent function name (snake_case)",
        help_text="Used for the Python function and filename. Stick to lowercase letters, numbers, and underscores.",
    ),
    Question(
        id="display_name",
        prompt="Human-friendly display name",
        help_text="Shown in documentation/metadata. Title case recommended.",
        required=False,
        default=""
    ),
    Question(
        id="description",
        prompt="Short agent description",
        help_text="One sentence summary of what the agent does.",
    ),
    Question(
        id="prompt_source",
        prompt="Prompt input mode (inline or file)",
        help_text="Choose 'inline' to type the system prompt or 'file' to load it from disk.",
        qtype="choice",
        choices=["inline", "file"],
        default="inline",
    ),
    Question(
        id="system_prompt",
        prompt="System prompt (multiline allowed)",
        help_text="Detailed instructions for the LLM (used when prompt mode is inline).",
        required=False,
    ),
    Question(
        id="prompt_path",
        prompt="System prompt file path",
        help_text="Absolute or relative path to a prompt file (used when prompt mode is file).",
        required=False,
    ),
    Question(
        id="model_id",
        prompt="Model identifier",
        help_text="Select the Ollama model the agent should use.",
        qtype="choice",
        choices=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS[0]
    ),
    Question(
        id="enable_code_execution",
        prompt="Enable sandboxed code execution? (true/false)",
        help_text="If true, the agent can run code through the sandbox.",
        qtype="boolean",
        default=True
    ),
    Question(
        id="sandbox_timeout",
        prompt="Sandbox timeout in seconds",
        help_text="Timeout applied when executing code in the sandbox.",
        qtype="integer",
        default=45
    ),
    Question(
        id="tool_selection",
        prompt="Comma-separated list of tool identifiers",
        help_text="Pick from the TOOL_LIBRARY keys. Include sandbox_test_code if you want code execution.",
        required=False,
        default="sandbox_test_code"
    ),
    Question(
        id="memory_profile",
        prompt="Memory profile (none|light|full)",
        help_text="Controls how chunking/embedding is applied for this agent.",
        qtype="choice",
        choices=["none", "light", "full"],
        default="light"
    ),
    Question(
        id="embedding_namespace",
        prompt="Embedding namespace (optional)",
        help_text="Namespace for embedding storage. Leave blank to auto-generate.",
        required=False,
        default=""
    ),
    Question(
        id="context_documents",
        prompt="Paths (comma separated) to supporting documents for chunking",
        help_text="Relative paths to context files that should be chunked & embedded.",
        required=False,
        default=""
    ),
    Question(
        id="output_dir",
        prompt=f"Output directory (default {DEFAULT_AGENT_DIR})",
        help_text="Relative path where the generated agent module should live.",
        required=False,
        default=str(DEFAULT_AGENT_DIR)
    )
]


@dataclass
class AgentConfig:
    """Structured configuration for an agent."""

    agent_name: str
    description: str
    system_prompt: str
    prompt_source: str = "inline"
    prompt_path: Optional[str] = None
    model_id: str = AVAILABLE_MODELS[0]
    enable_code_execution: bool = True
    sandbox_timeout: int = 45
    tool_selection: List[str] = field(
        default_factory=lambda: ["sandbox_test_code"])
    memory_profile: str = "light"
    embedding_namespace: Optional[str] = None
    context_documents: List[str] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: DEFAULT_AGENT_DIR)
    display_name: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_name = self._sanitize_name(self.agent_name)
        if not self.agent_name:
            raise ValueError("agent_name cannot be empty after sanitization")

        if self.display_name is None or not self.display_name.strip():
            self.display_name = self.agent_name.replace("_", " ").title()

        if not self.embedding_namespace:
            self.embedding_namespace = f"{self.agent_name}_namespace"

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        self.prompt_source = (self.prompt_source or "inline").lower()
        if self.prompt_source not in {"inline", "file"}:
            raise ValueError("prompt_source must be either 'inline' or 'file'")

        prompt_path_obj: Optional[Path] = None
        if self.prompt_path:
            prompt_path_obj = Path(self.prompt_path).expanduser()
            if not prompt_path_obj.is_absolute():
                prompt_path_obj = (Path.cwd() / prompt_path_obj).resolve()
        self.prompt_path = prompt_path_obj

        if self.prompt_source == "file":
            if self.prompt_path is None or not self.prompt_path.exists():
                raise ValueError("Prompt file path is invalid or does not exist.")
            self.system_prompt = self.prompt_path.read_text(encoding="utf-8")
        else:
            self.system_prompt = self.system_prompt or ""

        self.system_prompt = self.system_prompt.strip()
        if not self.system_prompt:
            raise ValueError("System prompt content cannot be empty.")

        # Normalise tool selection
        unique_tools = []
        for tool in self.tool_selection:
            cleaned = tool.strip()
            if cleaned and cleaned not in unique_tools:
                unique_tools.append(cleaned)
        self.tool_selection = unique_tools

        # Normalise document paths
        cleaned_docs = []
        for doc in self.context_documents:
            doc_path = doc.strip()
            if doc_path:
                cleaned_docs.append(doc_path)
        self.context_documents = cleaned_docs

        self.metadata.update(
            {
                "display_name": self.display_name,
                "memory_profile": self.memory_profile,
                "embedding_namespace": self.embedding_namespace,
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "tools": self.tool_selection,
                "model_id": self.model_id,
                "sandbox_timeout": self.sandbox_timeout,
                "enable_code_execution": self.enable_code_execution,
                "prompt_source": self.prompt_source,
                "prompt_file_source": str(self.prompt_path) if self.prompt_path else None,
                "system_prompt_length": len(self.system_prompt),
            }
        )

    @staticmethod
    def _sanitize_name(raw_name: str) -> str:
        """Convert arbitrary user input into snake_case."""
        if not raw_name:
            return ""
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_name)
        cleaned = re.sub(r"_{2,}", "_", cleaned)
        cleaned = cleaned.strip("_").lower()
        return cleaned

    @classmethod
    def from_answers(cls, answers: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from questionnaire answers."""
        tool_selection = [
            item.strip()
            for item in answers.get("tool_selection", "").split(",")
            if item.strip()
        ]

        raw_docs = answers.get("context_documents", "")
        context_docs = [
            item.strip()
            for item in re.split(r"[\n,;]+", raw_docs)
            if item.strip()
        ]

        return cls(
            agent_name=answers["agent_name"],
            display_name=answers.get("display_name", ""),
            description=answers["description"],
            system_prompt=answers.get("system_prompt", ""),
            prompt_source=(answers.get("prompt_source") or "inline"),
            prompt_path=answers.get("prompt_path") or None,
            model_id=answers.get("model_id") or AVAILABLE_MODELS[0],
            enable_code_execution=_normalise_boolean(
                answers.get("enable_code_execution", True),
                default=True
            ),
            sandbox_timeout=int(answers.get("sandbox_timeout", 45)),
            tool_selection=tool_selection or ["sandbox_test_code"],
            memory_profile=answers.get("memory_profile", "light"),
            embedding_namespace=answers.get("embedding_namespace") or None,
            context_documents=context_docs,
            output_dir=answers.get("output_dir") or str(DEFAULT_AGENT_DIR),
        )


def validate_answers(answers: Dict[str, Any]) -> None:
    """Raise descriptive errors for missing/invalid answers."""
    for question in QUESTIONNAIRE:
        value = answers.get(question.id)
        if question.required and (value is None or str(value).strip() == ""):
            raise ValueError(f"Missing required answer: {question.id}")

        if question.id == "agent_name" and value:
            sanitised = value.replace("-", "_")
            if not re.fullmatch(r"[a-zA-Z0-9_]+", sanitised):
                raise ValueError(
                    "agent_name must contain only letters, numbers, underscores."
                )

        if question.qtype == "choice" and question.choices and value:
            if value not in question.choices:
                raise ValueError(
                    f"{question.id} must be one of {question.choices}")

        if question.qtype == "integer" and value not in (None, ""):
            try:
                int(value)
            except ValueError as exc:
                raise ValueError(f"{question.id} must be an integer") from exc

        if question.qtype == "boolean" and value not in (None, ""):
            if isinstance(value, str):
                lowered = value.lower()
                if lowered not in ("true", "false", "yes", "no", "1", "0"):
                    raise ValueError(
                        f"{question.id} must be a boolean-like value")

    prompt_source = (answers.get("prompt_source") or "inline").lower()
    if prompt_source not in {"inline", "file"}:
        raise ValueError("prompt_source must be either 'inline' or 'file'")

    if prompt_source == "inline":
        prompt_text = answers.get("system_prompt", "")
        if not str(prompt_text).strip():
            raise ValueError(
                "system_prompt is required when prompt_source is 'inline'")
    else:
        prompt_path = answers.get("prompt_path")
        if not prompt_path or not str(prompt_path).strip():
            raise ValueError(
                "prompt_path is required when prompt_source is 'file'")
        resolved = Path(str(prompt_path)).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if not resolved.exists():
            raise ValueError(f"Prompt file not found at {resolved}")


def _normalise_boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "y"):
            return True
        if lowered in ("false", "no", "0", "n"):
            return False
    return default


def collect_answers_interactively() -> Dict[str, Any]:
    """Prompt via stdin/stdout. Useful when running the script manually."""
    answers: Dict[str, Any] = {}
    for question in QUESTIONNAIRE:
        default_hint = f" [{question.default}]" if question.default not in (
            None, "") else ""
        prompt = f"{question.prompt}{default_hint}: "
        response = input(prompt).strip()

        if not response and question.default not in (None, ""):
            response = str(question.default)

        if question.qtype == "boolean":
            answers[question.id] = _normalise_boolean(
                response, bool(question.default))
        else:
            answers[question.id] = response
    return answers


def materialise_agent_module(config: AgentConfig) -> Dict[str, Path]:
    """Generate agent module and associated files."""
    module_path = config.output_dir / f"{config.agent_name}.py"
    prompt_dir = DEFAULT_PROMPT_DIR
    metadata_dir = DEFAULT_METADATA_DIR

    module_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompt_dir / f"{config.agent_name}.prompt"
    prompt_text = config.system_prompt.rstrip() + "\n"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    module_content = render_agent_module(config, module_path, prompt_path)
    module_path.write_text(module_content, encoding="utf-8")

    metadata_path = metadata_dir / f"{config.agent_name}.json"
    metadata_payload = config.metadata.copy()
    metadata_payload.update(
        {
            "agent_name": config.agent_name,
            "description": config.description,
            "system_prompt_file": str(prompt_path),
            "module_file": str(module_path),
            "context_documents": config.context_documents,
            "memory_profile_settings": MEMORY_PROFILE_SETTINGS.get(
                config.memory_profile, MEMORY_PROFILE_SETTINGS["none"]
            ),
            "tool_details": {
                tool: TOOL_LIBRARY.get(tool, {}).get(
                    "description", "custom tool")
                for tool in config.tool_selection
            },
        }
    )

    embedding_artifacts = process_context_documents(
        config, metadata_path.parent)
    if embedding_artifacts:
        metadata_payload["embedding_artifacts"] = embedding_artifacts

    metadata_path.write_text(json.dumps(
        metadata_payload, indent=2), encoding="utf-8")
    config.metadata = metadata_payload.copy()

    logger.info("Generated agent module at %s", module_path)

    paths = {
        "module": module_path,
        "prompt": prompt_path,
        "metadata": metadata_path,
    }

    if embedding_artifacts:
        paths["embeddings"] = Path(embedding_artifacts["embedding_file"])

    return paths


def generate_agent(answers: Dict[str, Any]) -> ScaffoldingResult:
    """Programmatic entry point for the UI. Returns structured artifacts."""
    try:
        validate_answers(answers)
        config = AgentConfig.from_answers(answers)
        paths = materialise_agent_module(config)
        artifacts = {key: str(value) for key, value in paths.items()}

        message = f"Agent '{config.agent_name}' generated successfully."
        return ScaffoldingResult(
            ok=True,
            message=message,
            artifacts=artifacts,
            metadata=config.metadata,
        )

    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Agent scaffolding failed: %s", exc)
        return ScaffoldingResult(
            ok=False,
            message=str(exc),
            artifacts={},
            metadata={},
        )


def render_agent_module(config: AgentConfig, module_path: Path, prompt_path: Path) -> str:
    """Render Python module for the generated agent."""
    timestamp = datetime.utcnow().isoformat() + "Z"

    tool_imports: List[str] = []
    tool_references: List[str] = []

    for tool_key in config.tool_selection:
        if tool_key not in TOOL_LIBRARY:
            raise ValueError(
                f"Unknown tool '{tool_key}'. Available: {list(TOOL_LIBRARY)}")
        entry = TOOL_LIBRARY[tool_key]
        tool_imports.append(entry["import"])
        tool_references.append(entry["reference"])

    tool_imports = sorted(set(tool_imports))
    tool_references = ", ".join(tool_references)

    if not tool_references:
        tool_references = ""

    prompt_relative = prompt_path.relative_to(module_path.parent)
    prompt_relative_str = "/".join(prompt_relative.parts)

    header = dedent(
        f"""\
        \"\"\"Auto-generated agent module for {config.display_name}.\"\"\"

        # Generated by agent.scaffolder on {timestamp}
        """
    ).strip()

    imports = ["from pathlib import Path", "from agent import agent"] + tool_imports
    imports_block = "\n".join(dict.fromkeys(imports))  # Preserve order, remove duplicates

    tools_block = (
        f"TOOLS = [{tool_references}]\n" if tool_references else "TOOLS: list = []\n"
    )

    function_body = dedent(
        f"""
        PROMPT_PATH = Path(__file__).parent / Path(r"{prompt_relative_str}")
        SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


        @agent(
            model_id="{config.model_id}",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
            enable_code_execution={str(config.enable_code_execution)},
            sandbox_timeout={config.sandbox_timeout}
        )
        def {config.agent_name}(query: str) -> str:
            \"\"\"{config.description}\"\"\"
            return query
        """
    ).strip()

    return "\n\n".join([header, imports_block, "", tools_block, "", function_body]) + "\n"


def process_context_documents(config: AgentConfig, metadata_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Chunk and embed context documents according to the selected memory profile.

    Args:
        config: Agent configuration containing document paths and profile.
        metadata_dir: Directory where metadata/embedding artefacts should be stored.

    Returns:
        Summary dictionary describing generated artefacts or None if skipped.
    """
    profile = MEMORY_PROFILE_SETTINGS.get(
        config.memory_profile, MEMORY_PROFILE_SETTINGS["none"])
    if not config.context_documents or not profile["levels"]:
        logger.info("No context documents to process for profile '%s'",
                    config.memory_profile)
        return None

    try:
        from assistants.chunking_assistant import ChunkingAssistant
        from assistants.embedding_assistant import EmbeddingAssistant
    except ImportError as exc:
        logger.error("Failed to import chunking/embedding assistants: %s", exc)
        return None

    chunk_size = profile["chunk_size"]
    chunk_overlap = profile["chunk_overlap"]
    if chunk_size is None or chunk_overlap is None:
        logger.info("Memory profile '%s' disabled chunking.",
                    config.memory_profile)
        return None

    documents: Dict[str, str] = {}
    for raw_path in config.context_documents:
        path = Path(raw_path)
        if not path.exists():
            logger.warning(
                "Context document %s not found; skipping.", raw_path)
            continue
        try:
            documents[str(path)] = path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem specific
            logger.error("Failed to read document %s: %s", raw_path, exc)

    if not documents:
        logger.info("No readable context documents for agent %s",
                    config.agent_name)
        return None

    chunker = ChunkingAssistant(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_metadata = {
        "agent_name": config.agent_name,
        "namespace": config.embedding_namespace,
        "profile": config.memory_profile,
    }
    chunk_results = chunker.chunk_documents(
        documents, base_metadata=chunk_metadata)

    embedder = EmbeddingAssistant()
    embedding_payload: Dict[str, List[Dict[str, Any]]] = {}
    chunk_totals = 0
    for doc_path, chunks in chunk_results.items():
        if not chunks:
            continue
        texts = [chunk["text"] for chunk in chunks]
        metadata = [chunk.get("metadata", {}) for chunk in chunks]
        embeddings = embedder.embed_texts(texts, metadata=metadata)
        embedding_payload[doc_path] = [item.to_dict() for item in embeddings]
        chunk_totals += len(embeddings)

    if not embedding_payload:
        logger.warning("Embedding payload empty for agent %s",
                       config.agent_name)
        return None

    embedding_file = metadata_dir / f"{config.agent_name}_embeddings.json"
    embedding_file.write_text(
        json.dumps(
            {
                "namespace": config.embedding_namespace,
                "profile": config.memory_profile,
                "documents": embedding_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Generated %d embeddings across %d documents for agent %s",
        chunk_totals,
        len(embedding_payload),
        config.agent_name,
    )

    return {
        "embedding_file": str(embedding_file),
        "document_count": len(embedding_payload),
        "total_chunks": chunk_totals,
        "profile": config.memory_profile,
        "namespace": config.embedding_namespace,
    }


def run_cli() -> None:
    """Entry point for manual execution."""
    print("StrandsAgents deterministic scaffolder\n")
    answers = collect_answers_interactively()
    result = generate_agent(answers)
    if not result.ok:
        print(f"\nGeneration failed: {result.message}")
        return

    print("\nGeneration complete. Artifacts:")
    for label, path in result.artifacts.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    run_cli()
