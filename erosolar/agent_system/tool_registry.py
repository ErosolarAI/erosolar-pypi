"""MCP-compatible tool registry and caching helpers."""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""


@dataclass
class MCPTool:
    """Model Context Protocol compatible tool descriptor."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_mcp_descriptor(self) -> Dict[str, Any]:
        """Return a serializable MCP compatible descriptor."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowStep:
    tool_name: str
    optional: bool = False
    depends_on: Optional[str] = None


@dataclass
class WorkflowDefinition:
    name: str
    triggers: Sequence[str]
    description: str
    steps: Sequence[WorkflowStep]


class ToolCache:
    """Simple LRU cache with TTL tracking for tool calls."""

    def __init__(self, max_entries: int = 512, ttl_seconds: int = 600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _purge(self) -> None:
        now = time.time()
        keys_to_delete = [
            key
            for key, payload in self._store.items()
            if payload["expires_at"] <= now
        ]
        for key in keys_to_delete:
            self._store.pop(key, None)

        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._store.get(key)
            if not payload:
                self.misses += 1
                return None
            if payload["expires_at"] <= time.time():
                self._store.pop(key, None)
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return payload["value"]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = {
                "value": value,
                "expires_at": time.time() + self.ttl_seconds,
            }
            self._store.move_to_end(key)
            self._purge()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


class ToolRegistry:
    """Central MCP tool store with embedding support."""

    def __init__(self, cache: Optional[ToolCache] = None):
        self._tools: Dict[str, MCPTool] = {}
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._lock = threading.Lock()
        self._embeddings: Dict[str, List[float]] = {}
        self._embedding_model = "text-embedding-3-large"
        self._cache = cache or ToolCache()
        self._embedding_client = None

    def register_tool(self, tool: MCPTool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        with self._lock:
            self._workflows[workflow.name] = workflow

    def list_tools(self) -> List[MCPTool]:
        with self._lock:
            return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        with self._lock:
            return self._tools.get(name)

    def get_workflows(self) -> List[WorkflowDefinition]:
        with self._lock:
            return list(self._workflows.values())

    def describe_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_mcp_descriptor() for tool in self.list_tools()]

    def set_embedding_client(self, client: Any, model: Optional[str] = None) -> None:
        self._embedding_client = client
        if model:
            self._embedding_model = model

    def ensure_embeddings(self) -> None:
        if not self._embedding_client:
            return
        missing = [tool for tool in self.list_tools() if tool.name not in self._embeddings]
        if not missing:
            return
        for tool in missing:
            text = json.dumps(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "metadata": tool.metadata,
                    "input_schema": tool.input_schema,
                },
                default=str,
            )
            response = self._embedding_client.embeddings.create(
                model=self._embedding_model,
                input=text,
            )
            vector = response.data[0].embedding
            self._embeddings[tool.name] = vector

    def semantic_search(
        self, query_embedding: List[float], top_k: int = 3
    ) -> List[MCPTool]:
        if not self._embeddings:
            return []
        scored: List[tuple[float, MCPTool]] = []
        for tool in self.list_tools():
            vector = self._embeddings.get(tool.name)
            if not vector:
                continue
            score = cosine_similarity(query_embedding, vector)
            scored.append((score, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored[:top_k]]

    def execute_tool(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if not tool:
            raise ToolExecutionError(f"Unknown tool: {name}")
        cache_key = f"{name}:{json.dumps(payload, sort_keys=True)}"
        cached = self._cache.get(cache_key)
        if cached:
            return {"cached": True, **cached}
        result = tool.handler(payload)
        self._cache.set(cache_key, result)
        return {"cached": False, **result}

    def cache_hit_rate(self) -> float:
        return self._cache.hit_rate

    def metadata_snapshot(self) -> Dict[str, Any]:
        return {
            "tool_count": len(self._tools),
            "workflow_count": len(self._workflows),
            "cache_hit_rate": round(self.cache_hit_rate(), 3),
            "embedding_model": self._embedding_model,
        }


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    from math import sqrt

    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = sqrt(sum(x * x for x in a_list))
    norm_b = sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
