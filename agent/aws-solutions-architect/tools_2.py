# tools.py
# Strands meta-tooling: define concrete tools with schemas and guardrails.
# Replace `mcp_call` with your actual MCP client adapter.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import time

try:
    # Strands tool decorator (adjust import if your package differs)
    from strands.tools import tool
except ImportError:
    # Fallback no-op decorator for local tests
    def tool(name: str = None, description: str = "", schema: Dict[str, Any] = None):
        def deco(fn):
            fn._tool_name = name or fn.__name__
            fn._tool_desc = description
            fn._tool_schema = schema or {}
            return fn
        return deco

# -----------------------------
# Simple MCP adapter (replace!)
# -----------------------------


def mcp_call(server: str, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic call into an MCP server/tool. Replace this with your real client logic.
    server: MCP server/tool identifier, e.g., "aws_docs.search"
    method: operation/method inside that server (if applicable)
    payload: arguments for the method
    """
    raise NotImplementedError("Wire this to your MCP client (stdio/ws).")


# ---------------------------------
# Small in-memory cache (hash keys)
# ---------------------------------
_cache: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SEC = 600  # 10 minutes


def _cache_get(key: str) -> Optional[Any]:
    ent = _cache.get(key)
    if not ent:
        return None
    ts, val = ent
    if time.time() - ts > CACHE_TTL_SEC:
        _cache.pop(key, None)
        return None
    return val


def _cache_put(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)


def _hash_payload(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------------------------------
# 1) AWS Docs Search
# ---------------------------------


@tool(
    name="aws_docs.search",
    description="Search official AWS documentation. Returns top hits with title, url, snippet.",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6},
            "filters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "topic": {"type": "string"}
                }
            }
        },
        "required": ["query"]
    }
)
def aws_docs_search(query: str, max_results: int = 6, filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    payload = {"query": query, "max_results": max_results,
               "filters": filters or {}}
    key = "aws_docs.search:" + _hash_payload(payload)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    # MCP server name and method may be identical; adapt as needed
    res = mcp_call(server="aws_docs.search", method="search", payload=payload)
    _cache_put(key, res)
    return res

# ---------------------------------
# 2) AWS Solutions Library (reference architectures)
# ---------------------------------


@tool(
    name="aws_solutions_library.find",
    description="Find AWS reference architectures/patterns in the Solutions Library.",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "category": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6}
        },
        "required": ["query"]
    }
)
def aws_solutions_library_find(query: str, category: Optional[str] = None, max_results: int = 6) -> Dict[str, Any]:
    payload = {"query": query, "category": category,
               "max_results": max_results}
    key = "aws_solutions_library.find:" + _hash_payload(payload)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    res = mcp_call(server="aws_solutions_library.find",
                   method="find", payload=payload)
    _cache_put(key, res)
    return res

# ---------------------------------
# 3) AWS Well-Architected check (6 pillars checklist)
# ---------------------------------


@tool(
    name="aws_well_architected.check",
    description="Validate a design against the 6 pillars. Input: short design summary; Output: risks, questions, mitigations.",
    schema={
        "type": "object",
        "properties": {
            "design_summary": {"type": "string", "minLength": 10},
            "pillars": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["operationalExcellence", "security", "reliability", "performanceEfficiency", "costOptimization", "sustainability"]
            }
        },
        "required": ["design_summary"]
    }
)
def aws_well_architected_check(design_summary: str, pillars: Optional[List[str]] = None) -> Dict[str, Any]:
    payload = {"design_summary": design_summary, "pillars": pillars or [
        "operationalExcellence", "security", "reliability", "performanceEfficiency", "costOptimization", "sustainability"
    ]}
    # No cache; design summaries are unique and we want fresh guidance
    return mcp_call(server="aws_well_architected.check", method="check", payload=payload)

# ---------------------------------
# 4) AWS Pricing estimate (rough order of magnitude)
# ---------------------------------


@tool(
    name="aws_pricing.estimate",
    description="Estimate monthly cost for selected AWS services under provided assumptions.",
    schema={
        "type": "object",
        "properties": {
            "services": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "region": {"type": "string"},
            # e.g., {"lambda_requests": 5e6, "s3_gb": 500, ...}
            "assumptions": {"type": "object"}
        },
        "required": ["services"]
    }
)
def aws_pricing_estimate(services: List[str], region: Optional[str] = None, assumptions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {"services": services, "region": region,
               "assumptions": assumptions or {}}
    # cache because pricing calls can be slow/paid
    key = "aws_pricing.estimate:" + _hash_payload(payload)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    res = mcp_call(server="aws_pricing.estimate",
                   method="estimate", payload=payload)
    _cache_put(key, res)
    return res

# ---------------------------------
# 5) KB vector search (internal playbooks/designs)
# ---------------------------------


@tool(
    name="kb_vector.search",
    description="Semantic search over internal KB. Returns doc ids + snippets; keep k small.",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
        },
        "required": ["query"]
    }
)
def kb_vector_search(query: str, k: int = 5) -> Dict[str, Any]:
    payload = {"query": query, "k": k}
    key = "kb_vector.search:" + _hash_payload(payload)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    res = mcp_call(server="kb_vector.search", method="search", payload=payload)
    _cache_put(key, res)
    return res

# ---------------------------------
# 6) Diagram export (Mermaid)
# ---------------------------------


@tool(
    name="diagram.mermaid_export",
    description="Render Mermaid text to an SVG/PNG and persist it to a file path.",
    schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "minLength": 5},
            "format": {"type": "string", "enum": ["svg", "png"], "default": "svg"},
            "out_path": {"type": "string"}
        },
        "required": ["code"]
    }
)
def diagram_mermaid_export(code: str, format: str = "svg", out_path: Optional[str] = None) -> Dict[str, Any]:
    payload = {"code": code, "format": format, "out_path": out_path}
    # no cache — output is a file
    return mcp_call(server="diagram.mermaid_export", method="render", payload=payload)

# ---------------------------------
# NOTE on `graph` and `a2a`:
# In most Strands runtimes these are built-in tools. If yours needs explicit wrappers,
# you can add thin @tool shims that proxy to the runtime’s graph/a2a endpoints.
# ---------------------------------
