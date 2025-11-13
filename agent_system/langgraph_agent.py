"""LangGraph powered ReAct agent wrapper."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .router import RouteDecision, ThreeTierRouter
from .tool_registry import ToolRegistry

try:
    from langgraph.graph import START, END, StateGraph
except ImportError:  # pragma: no cover - langgraph optional during tests
    START = END = None
    StateGraph = None


URL_RE = re.compile(r"https?://\S+")

WEATHER_TAIL_RE = re.compile(
    r"\b(?:in|for|at)\s+(?P<location>[^?.!]+)",
    re.IGNORECASE,
)

WEATHER_PREFIX_RE = re.compile(
    r"^(?P<location>[^?.!]+?)\s+(?:weather|forecast|temperature)\b",
    re.IGNORECASE,
)

WEATHER_TRAILING_PHRASES = (
    "right now",
    "this evening",
    "this afternoon",
    "this morning",
    "this weekend",
    "this week",
    "tonight",
    "tomorrow",
    "today",
    "now",
)


def _clean_location_candidate(fragment: str) -> str:
    fragment = fragment.strip(" \t\n\r,.:;!?-\"")
    if not fragment:
        return ""
    fragment = re.sub(r"\s+", " ", fragment)
    lowered = fragment.lower()
    # Remove trailing temporal phrases like "today" or "tomorrow"
    trimmed = True
    while trimmed and fragment:
        trimmed = False
        for phrase in WEATHER_TRAILING_PHRASES:
            suffix = " " + phrase
            if lowered.endswith(suffix):
                fragment = fragment[: -len(suffix)].rstrip(" ,.:;!?-\"")
                lowered = fragment.lower()
                trimmed = True
                break
    return fragment.strip(" \t\n\r,.:;!?-\"")


def _extract_weather_location(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    candidates = []

    tail_matches = list(WEATHER_TAIL_RE.finditer(prompt))
    if tail_matches:
        candidates.append(tail_matches[-1].group("location"))

    prefix_match = WEATHER_PREFIX_RE.search(prompt)
    if prefix_match:
        candidates.append(prefix_match.group("location"))

    for candidate in candidates:
        cleaned = _clean_location_candidate(candidate)
        if cleaned:
            return cleaned

    return prompt


class LangGraphReActAgent:
    """Coordinates routing, tool execution, and LLM reasoning."""

    def __init__(
        self,
        router: ThreeTierRouter,
        registry: ToolRegistry,
        llm_client: OpenAI,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.router = router
        self.registry = registry
        self.llm_client = llm_client
        self.system_prompt = system_prompt or (
            "You are LifePilot, a ReAct-style assistant that reasons over tool outputs before answering."
        )
        self._graph = self._build_graph() if StateGraph else None

    # ------------------------------------------------------------------
    # LangGraph plumbing
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("route", self._route_node)
        graph.add_node("execute", self._execute_node)
        graph.add_edge(START, "route")
        graph.add_edge("route", "execute")
        graph.add_edge("execute", END)
        return graph.compile()

    def _route_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query")
        decision = self.router.route(query)
        return {**state, "decision": decision}

    def _execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision: RouteDecision = state.get("decision")
        query = state.get("query")
        outputs = self._execute_plan(query, decision)
        return {**state, "tool_outputs": outputs}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, prompt: str) -> Dict[str, Any]:
        if self._graph:
            return self._graph.invoke({"query": prompt})
        decision = self.router.route(prompt)
        return {
            "decision": decision,
            "tool_outputs": self._execute_plan(prompt, decision),
        }

    def run(self, prompt: str) -> Dict[str, Any]:
        plan_state = self.plan(prompt)
        decision: RouteDecision = plan_state.get("decision")
        tool_outputs = plan_state.get("tool_outputs", [])
        final_response = self._render_final(prompt, decision, tool_outputs)
        return {
            "decision": decision,
            "tool_outputs": tool_outputs,
            "final_response": final_response,
        }

    def stream_final(self, prompt: str, decision: RouteDecision, tool_outputs: List[Dict[str, Any]]):
        toollog = self._format_tool_log(tool_outputs)
        base_messages = self._build_messages(prompt, toollog)
        stream = self.llm_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=base_messages,
            stream=True,
        )
        for chunk in stream:
            yield chunk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _execute_plan(self, prompt: str, decision: Optional[RouteDecision]) -> List[Dict[str, Any]]:
        if not decision or not decision.tool_sequence:
            return []
        observations = []
        for tool_name in decision.tool_sequence:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue
            try:
                payload = self._guess_payload(tool_name, prompt)
                result = self.registry.execute_tool(tool_name, payload)
                observations.append(
                    {
                        "tool": tool_name,
                        "payload": payload,
                        "result": result,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                observations.append(
                    {
                        "tool": tool_name,
                        "payload": {},
                        "error": str(exc),
                    }
                )
        return observations

    def _guess_payload(self, tool_name: str, prompt: str) -> Dict[str, Any]:
        if "search" in tool_name:
            return {"query": prompt}
        if "extract" in tool_name:
            match = URL_RE.search(prompt)
            if not match:
                raise ValueError("No URL detected for extraction request")
            return {"url": match.group(0), "prompt": f"Extract facts relevant to: {prompt}"}
        if tool_name == "open_metro_weather_lookup":
            location = _extract_weather_location(prompt)
            return {"location": location}
        return {"query": prompt}

    def _render_final(
        self,
        prompt: str,
        decision: Optional[RouteDecision],
        tool_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        toollog = self._format_tool_log(tool_outputs)
        messages = self._build_messages(prompt, toollog)
        completion = self.llm_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
        )
        message = completion.choices[0].message
        return {
            "content": message.content,
            "reasoning_content": getattr(message, "reasoning_content", None),
        }

    def _build_messages(self, prompt: str, tool_log: str) -> List[Dict[str, str]]:
        system = {
            "role": "system",
            "content": self.system_prompt + "\nUse the supplied tool log when answering.",
        }
        tool_summary = {
            "role": "system",
            "name": "tool_logger",
            "content": tool_log,
        }
        user = {"role": "user", "content": prompt}
        return [system, tool_summary, user]

    def _format_tool_log(self, tool_outputs: List[Dict[str, Any]]) -> str:
        if not tool_outputs:
            return "No tools executed."
        blocks = []
        for obs in tool_outputs:
            tool = obs.get("tool")
            payload = obs.get("payload")
            result = obs.get("result") or obs.get("error")
            blocks.append(
                json_dumps(
                    {
                        "tool": tool,
                        "payload": payload,
                        "result": result,
                    }
                )
            )
        return "\n".join(blocks)


def json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
