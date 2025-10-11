# file: meta_agent.py
"""
A production-grade @agent decorator for Strands-style agent creation.

Goals
- Feel like: from strands import Agent, tool ... but add a higher-level @agent that
  configures model, system prompt, tools, safety, and providers (Bedrock | Ollama)
- Register Strands @tool and MCP tools in one place with allowlists and guards
- Lazy-initialize the underlying Agent once per function (fast cold start, reuse later)
- Provide a simple call signature for the decorated function: func(ctx, **kwargs)
- Keep security-first defaults (timeouts, tool allowlist, no shell unless explicit)

Requires
- strands-agents (Agent + @tool)
- Optionally strands-agents-tools (for built-ins) and MCP client if you load MCP tools
- Optional: Bedrock AgentCore or boto3 if calling Bedrock directly (out of scope here)

"""
from __future__ import annotations
import os
import functools
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Dict, Any, Union

try:
    # Strands core primitives
    from strands import Agent, tool  # type: ignore
except Exception as e:
    raise RuntimeError(
        "strands-agents is required. Install: `pip install strands-agents strands-agents-tools`"
    ) from e

# ---- Optional imports (loaded lazily in decorator) ----
# - MCP client / FastMCP for bringing external tools
# - Ollama client, Bedrock runtime, etc.


# ---------- Provider Model Spec ----------

@dataclass(frozen=True)
class ModelSpec:
    """
    Standardized model spec so callers don't need to remember provider quirks.

    Examples:
      Bedrock Claude Sonnet 4.5:
        ModelSpec(
          provider="bedrock",
          model_id=os.getenv("BEDROCK_CLAUDE_SONNET_45_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
          region=os.getenv("AWS_REGION", "us-east-1")
        )

      Ollama (local):
        ModelSpec(provider="ollama", model_id="qwen3:4b")
    """
    provider: str                          # "bedrock" | "ollama" | "openai" | etc.
    model_id: str                          # provider-specific model identifier
    region: Optional[str] = None           # used by Bedrock
    # temperature, top_p, etc.
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------- Tool Loader Spec ----------

@dataclass
class ToolSpec:
    """
    Register tools in a structured, safe way.

    kind: "strands" (native @tool callables) | "mcp" (remote tools from MCP servers)
    ref:  Python callable for "strands", or MCP descriptor dict for "mcp"
    name_override: rename tool exposed to model (optional)
    """
    kind: str
    ref: Any
    name_override: Optional[str] = None
    allow_destructive: bool = False  # guard flag for risky tools


# ---------- Agent Decorator ----------

def agent(
    *,
    name: str,
    description: str = "",
    system_prompt: str = "",
    model: Optional[ModelSpec] = None,
    tools: Optional[Iterable[ToolSpec]] = None,
    tool_allowlist: Optional[Iterable[str]] = None,
    max_tokens: int = 2048,
    request_timeout_s: int = 60,
    # for models supporting extended thinking modes
    enable_thinking: bool = False,
    # surface logs/traces if underlying SDK supports it
    telemetry: bool = True,
    default_stream: bool = False,
    strict_mode: bool = True,              # fail-fast on unknown tools/params
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Usage:
      @agent(
        name="researcher",
        description="Plans and executes web research with citations",
        system_prompt="You are a careful, source-grounded researcher...",
        model=ModelSpec(
            provider="bedrock",
            model_id=os.getenv("BEDROCK_CLAUDE_SONNET_45_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
            region=os.getenv("AWS_REGION", "us-east-1"),
            extras={"temperature": 0.2}
        ),
        tools=[
            ToolSpec(kind="strands", ref=my_local_strands_tool),
            ToolSpec(kind="mcp", ref={"server": "context7", "args": {"sources": ["aws","python"]}})
        ],
        tool_allowlist=["search", "http_request", "retrieve", "context7.query"],
        max_tokens=4096,
        request_timeout_s=90,
      )
      async def my_agent(ctx, query: str) -> str:
          # ctx.agent is a ready-to-use Strands Agent instance
          # Return the final string (or structured) result
          response = await ctx.agent.run(query)
          return response

    The decorator:
    - Lazily builds a Strands Agent with the selected provider+model
    - Registers allowed tools (native Strands @tool and MCP-provided)
    - Injects an execution context `ctx` into your function with `ctx.agent`
    """
    def decorator(user_func: Callable[..., Any]) -> Callable[..., Any]:
        agent_holder: Dict[str, Agent] = {}  # cache per-decorated function

        # Inner helper: build model backend for Strands Agent
        async def _build_agent_once() -> Agent:
            if "instance" in agent_holder:
                return agent_holder["instance"]

            # Provider selection & construction
            prov = (model.provider if model else "").lower() if model else ""
            mdl = (model.model_id if model else None)

            if not mdl:
                raise ValueError("ModelSpec.model_id is required")

            # Instantiate Strands Agent (Strands handles the provider adapters)
            # NOTE: Strands docs show Agent(model=..., tools=[...]) style.
            # We keep tools empty for now and register after creation.
            # Ref: Quickstart & API reference.
            ag = Agent(
                name=name,
                description=description,
                model=mdl,
                system_prompt=system_prompt,
                provider=prov or None,
                provider_options=(model.extras if model else None),
                region=(model.region if model else None),
                max_tokens=max_tokens,
                request_timeout_s=request_timeout_s,
                enable_thinking=enable_thinking,
                telemetry=telemetry,
                stream=default_stream,
                strict_mode=strict_mode,
            )

            # Register tools with security guardrails
            if tools:
                await _register_tools(ag, tools, tool_allowlist or [])

            agent_holder["instance"] = ag
            return ag

        # Tool registration with allowlist enforcement
        async def _register_tools(ag: Agent, specs: Iterable[ToolSpec], allowlist: Iterable[str]):
            allow = set(t.lower() for t in allowlist) if allowlist else set()
            for spec in specs:
                if spec.kind == "strands":
                    # Native Strands tool callable (already decorated with @tool or plain callable)
                    func = spec.ref
                    exposed_name = (spec.name_override or getattr(
                        func, "__name__", "tool")).lower()
                    if allow and exposed_name.lower() not in allow:
                        if strict_mode:
                            raise PermissionError(
                                f"Tool '{exposed_name}' not in allowlist")
                        else:
                            continue
                    # If callable already wrapped by @tool, Strands accepts it; else we annotate
                    # We rely on Strands to introspect type hints and docstring for schema.
                    ag.register_tool(func, name=exposed_name,
                                     allow_destructive=spec.allow_destructive)

                elif spec.kind == "mcp":
                    # Load MCP server tools and expose selected ones.
                    # This is a placeholder for your MCP client wiring; different clients exist.
                    # We demonstrate the intent and security flow while keeping vendor-agnostic.
                    mcp_desc: Dict[str, Any] = spec.ref
                    server_name = mcp_desc.get("server")
                    server_args = mcp_desc.get("args", {})
                    if not server_name:
                        raise ValueError(
                            "MCP ToolSpec.ref must include 'server'")

                    # Example: with FastMCP client you would:
                    # from mcp.client import connect_server, list_tools
                    # tools = await connect_server(server_name, **server_args).list_tools()
                    # for t in tools: if in allowlist -> ag.register_tool(MCPProxy(t))
                    # For safety, refuse shell-like tools unless explicitly allowed.
                    discovered = await _discover_mcp_tools(server_name, server_args)
                    for tname, proxy_callable in discovered.items():
                        lname = tname.lower()
                        if allow and lname not in allow:
                            continue
                        if ("shell" in lname or "exec" in lname) and lname not in allow:
                            continue
                        ag.register_tool(
                            proxy_callable, name=lname, allow_destructive=False)
                else:
                    raise ValueError(f"Unknown ToolSpec.kind '{spec.kind}'")

        # Placeholder MCP discovery (replace with your actual client)
        async def _discover_mcp_tools(server: str, args: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
            # Implement with python-sdk FastMCP or your preferred MCP client
            # Return dict: { "tool_name": async callable(...)->Any }
            return {}

        @functools.wraps(user_func)
        def sync_wrapper(*args, **kwargs):
            # Allow user function to be sync; we’ll lift to async for agent calls as needed
            async def _runner():
                ag = await _build_agent_once()
                ctx = type("AgentContext", (), {"agent": ag})
                if asyncio.iscoroutinefunction(user_func):
                    return await user_func(ctx, *args, **kwargs)
                return user_func(ctx, *args, **kwargs)

            try:
                loop = asyncio.get_running_loop()
                return asyncio.ensure_future(_runner())  # caller can await
            except RuntimeError:
                return asyncio.run(_runner())

        return sync_wrapper
    return decorator
